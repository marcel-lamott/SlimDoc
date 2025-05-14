import argparse
from rich.progress import (
    Progress,
)
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch
import torch.optim
import transformers
import wandb
from typing import Union, get_args

from slimdoc import (
    CONSTS,
    DEFAULTS,
    SUPPORTED_DATASET,
    TASKS,
    DATASET_CONF,
    DUModel,
    DistillMode,
)
from slimdoc.data.hf_dataset import load_dataset
from slimdoc.data.utils import create_dataloader, load_vocab_map
from slimdoc.train import ExperimentDefinition
from slimdoc.model import get_model, forward
from slimdoc.train.utils import (
    extract_text_logits,
    get_device,
    accuracy_ser,
    accuracy_vqa,
    get_linear_schedule_with_min_lr,
)

TRAIN_STEP_LOG_INTERVAL = 10  # Accumulate train losses and accuracy over this number of batches and log then (does not impact training, only logging)

# Define loss functions
mse_loss_fn = nn.MSELoss()
ce_loss_fn = nn.CrossEntropyLoss()  # probably mean? becuase two outputs
kl_loss_fn = nn.KLDivLoss(reduction="batchmean")


def eval_model(
    model,
    teacher_model,
    vocab_lookup,
    eval_dataloader,
    cur_train_steps,
    device,
    is_first_distillation_phase,
    ed: ExperimentDefinition,
    debug_run: bool,
    progress: Progress,
):
    is_distillation = (
        ed.mode == DistillMode.DISTILLATION
        or ed.mode == DistillMode.DISTILLATION_IN_1PHASE
    )
    is_1phase_distillation = ed.mode == DistillMode.DISTILLATION_IN_1PHASE

    # EVALUATING
    model.eval()
    running_acc_eval = 0.0
    running_total_loss_eval = 0.0
    running_attention_mse_loss_eval = 0.0
    running_hidden_states_mse_loss_eval = 0.0
    running_embedding_mse_loss_eval = 0.0
    running_logits_kl_loss_eval = 0.0
    running_logits_ce_loss_eval = 0.0

    eval_task = progress.add_task("[cyan]Eval...", total=len(eval_dataloader))

    for inputs in eval_dataloader:
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Convert the original token IDs to the new token IDs for the student vocab
        student_input_ids = (
            vocab_lookup[inputs["input_ids"]]
            if vocab_lookup is not None
            else inputs["input_ids"]
        )
        text_seq_length = student_input_ids.shape[1]

        outputs: dict[str, torch.Tensor] = forward(
            model=model,
            model_type=ed.model,
            output_internals=is_distillation,
            input_ids=student_input_ids,
            bbox=inputs["bbox"],
            attention_mask=inputs["attention_mask"],
            pixel_values=(
                inputs["pixel_values"]
                if ed.model == DUModel.LayoutLMv3_TextAndImage
                else None
            ),
        )

        labels = inputs["labels"]

        teacher_outputs = None
        if is_distillation:
            with torch.no_grad():
                teacher_outputs: dict[str, torch.Tensor] = forward(
                    model=teacher_model,
                    model_type=ed.model,
                    output_internals=True,
                    input_ids=inputs["input_ids"],
                    bbox=inputs["bbox"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=(
                        inputs["pixel_values"]
                        if ed.model == DUModel.LayoutLMv3_TextAndImage
                        else None
                    ),
                )

        total_loss, accuracy, losses = calculate_loss_and_accuracy(
            outputs=outputs,
            teacher_outputs=teacher_outputs,
            labels=labels,
            text_seq_length=text_seq_length,
            model_type=ed.model,
            task=ed.task,
            num_labels=ed.num_labels,
            is_distillation=is_distillation,
            is_first_distillation_phase=is_first_distillation_phase,
            is_1phase_distillation=is_1phase_distillation,
            student_layer_mapping=ed.student_layer_map,
            alpha=ed.alpha,
            beta=ed.beta,
            gamma=ed.gamma,
            delta=ed.delta,
            device=device,
            use_ce_loss_for_logits_distillation=ed.use_ce_loss_for_logits_distillation,
        )

        (
            attention_mse_loss,
            hidden_states_mse_loss,
            embedding_mse_loss,
            logits_kl_loss,
            logits_ce_loss,
        ) = losses
        running_attention_mse_loss_eval += attention_mse_loss.cpu().item()
        running_hidden_states_mse_loss_eval += hidden_states_mse_loss.cpu().item()
        running_embedding_mse_loss_eval += embedding_mse_loss.cpu().item()
        running_logits_kl_loss_eval += logits_kl_loss.cpu().item()
        running_logits_ce_loss_eval += logits_ce_loss.cpu().item()

        running_total_loss_eval += total_loss.cpu().item()
        running_acc_eval += accuracy

        progress.update(eval_task, advance=1)

    progress.remove_task(eval_task)

    avg_attention_mse_loss_eval = running_attention_mse_loss_eval / len(eval_dataloader)
    avg_hidden_states_mse_loss_eval = running_hidden_states_mse_loss_eval / len(
        eval_dataloader
    )
    avg_embedding_mse_loss_eval = running_embedding_mse_loss_eval / len(eval_dataloader)
    avg_logits_kl_loss_eval = running_logits_kl_loss_eval / len(eval_dataloader)
    avg_logits_ce_loss_eval = running_logits_ce_loss_eval / len(eval_dataloader)

    avg_total_loss_eval = running_total_loss_eval / len(eval_dataloader)
    avg_acc_eval = running_acc_eval / len(eval_dataloader)

    if not debug_run:
        wandb.log(
            {
                "eval/total_loss": avg_total_loss_eval,
                "eval/attention_mse_loss": avg_attention_mse_loss_eval,
                "eval/hidden_states_mse_loss": avg_hidden_states_mse_loss_eval,
                "eval/embedding_mse_loss": avg_embedding_mse_loss_eval,
                "eval/logits_kl_loss": avg_logits_kl_loss_eval,
                "eval/logits_ce_loss": avg_logits_ce_loss_eval,
                "eval/accuracy": avg_acc_eval,
            },
            step=cur_train_steps,
        )

    return avg_total_loss_eval, avg_acc_eval


def calculate_loss_and_accuracy(
    outputs: Union[
        dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]
    ],
    teacher_outputs: Union[
        dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]
    ],
    labels: torch.Tensor,
    text_seq_length: int,
    model_type: DUModel,
    task: TASKS,
    num_labels: int,
    is_distillation: bool,
    is_first_distillation_phase: bool,
    is_1phase_distillation: bool,
    student_layer_mapping: list[int],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    device,
    use_ce_loss_for_logits_distillation: bool = False,
) -> tuple[
    torch.Tensor,
    float,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:

    attention_mse_loss = torch.tensor([0.0], device=device)
    hidden_states_mse_loss = torch.tensor([0.0], device=device)
    embedding_mse_loss = torch.tensor([0.0], device=device)
    logits_kl_loss = torch.tensor([0.0], device=device)
    logits_ce_loss = torch.tensor([0.0], device=device)

    def unpack_outputs(vals, has_seperate_flows):
        if not has_seperate_flows:
            return (vals,)
        else:
            # ( (T,L), (T,L), (T,L), ...)
            flows_count = len(vals[0])
            flow_vals = [[] for _ in range(flows_count)]
            for layer_outputs in vals:
                for i_flow in range(flows_count):
                    flow_vals[i_flow].append(layer_outputs[i_flow])

            for i_flow in range(flows_count):
                flow_vals[i_flow] = tuple(flow_vals[i_flow])

            flow_vals = tuple(flow_vals)
            return flow_vals

    #
    # Feature based loss
    #
    if is_distillation and (is_first_distillation_phase or is_1phase_distillation):
        # For LiLT_TextAndLayoutFlow the outputs maps to a tuple of tuples instead to a tuple of tensors.
        # We unpack these such that we have a tuple in the shape ((TextFlow,TextFlow,...), (LayoutFlow,LayoutFlow,...)).
        # For all other models we have a tuple in the shape ((DefaultFlow, DefaultFlow, ...), )
        # The length of the outer tuple determines the number of flows (for all models 1, except for LiLT_TextAndLayoutFlow where it is 2)
        has_seperate_flows: bool = model_type == DUModel.LiLT_TextAndLayoutFlow
        student_flows_hidden_states = unpack_outputs(
            outputs["hidden_states"], has_seperate_flows
        )
        student_flows_attentions = unpack_outputs(
            outputs["attentions"], has_seperate_flows
        )
        teacher_flows_hidden_states = unpack_outputs(
            teacher_outputs["hidden_states"], has_seperate_flows
        )
        teacher_flows_attentions = unpack_outputs(
            teacher_outputs["attentions"], has_seperate_flows
        )
        flows_count = len(student_flows_hidden_states)

        for i_flow in range(flows_count):  # For LiLT_TextAndLayoutFlow 2, else 1
            student_hidden_states = student_flows_hidden_states[i_flow]
            student_attentions = student_flows_attentions[i_flow]
            teacher_hidden_states = teacher_flows_hidden_states[i_flow]
            teacher_attentions = teacher_flows_attentions[i_flow]

            # Distillation: First one complete training only on hidden_states and attentions...
            for student_layer_idx, teacher_layer_idx in enumerate(
                student_layer_mapping
            ):
                # We distill the attention scores...
                attention_mse_loss += mse_loss_fn(
                    student_attentions[student_layer_idx],
                    teacher_attentions[teacher_layer_idx],
                )
                # ...and we distill the hidden_states, where layer indices are offset by 1 due to the embedding layer
                hidden_states_mse_loss += mse_loss_fn(
                    student_hidden_states[student_layer_idx + 1],
                    teacher_hidden_states[teacher_layer_idx + 1],
                )

            # Divide by number of layers
            # attention_mse_loss = attention_mse_loss / len(student_layer_mapping)
            # hidden_states_mse_loss = hidden_states_mse_loss / len(student_layer_mapping)

            # Finally we also distill the embedding layer
            embedding_mse_loss += mse_loss_fn(
                student_hidden_states[0], teacher_hidden_states[0]
            )

        # Divide by number of layers and of flows
        attention_mse_loss = attention_mse_loss / (
            len(student_layer_mapping) * flows_count
        )
        hidden_states_mse_loss = hidden_states_mse_loss / (
            len(student_layer_mapping) * flows_count
        )
        embedding_mse_loss = embedding_mse_loss / flows_count

    #
    # Logits based loss
    #
    # Flatten the tokens and labels for CrossEntropyLoss
    if task == TASKS.SER:
        # SER shape: batch_size, seq_len
        logits: torch.Tensor = extract_text_logits(
            model_type=model_type,
            logits=outputs["logits"],
            text_seq_length=text_seq_length,
            task=task,
        )

        if is_distillation:
            if (not is_first_distillation_phase) or is_1phase_distillation:
                teacher_logits = extract_text_logits(
                    model_type=model_type,
                    logits=teacher_outputs["logits"],
                    text_seq_length=text_seq_length,
                    task=task,
                )
                if use_ce_loss_for_logits_distillation:
                    # Comparison to TinyBERT
                    loss_val = ce_loss_fn(
                        logits.reshape(-1, num_labels),
                        teacher_logits.reshape(-1, num_labels),
                    )
                    logits_ce_loss += loss_val
                else:
                    # Distillation: ...afterwards one complete training only on logits
                    # Have to use log_softmax for predictions here, see: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                    loss_val = kl_loss_fn(
                        F.log_softmax(logits, dim=-1), F.softmax(teacher_logits, dim=-1)
                    )
                    logits_kl_loss += loss_val
        else:
            # Fine-tuning
            # We need to call reshape(...) instead of view(...) on the logits, as for LayoutLMv3 the logits are non-contigous and view does not work then
            loss_val = ce_loss_fn(logits.reshape(-1, num_labels), labels.view(-1))
            logits_ce_loss += loss_val

        accuracy = accuracy_ser(logits=logits, labels=labels)

    elif task == TASKS.VQA:
        # VQA shape: batch_size, seq_len
        start_positions, end_positions = labels[:, 0], labels[:, 1]
        start_logits = extract_text_logits(
            model_type=model_type,
            logits=outputs["start_logits"],
            text_seq_length=text_seq_length,
            task=task,
        )
        end_logits = extract_text_logits(
            model_type=model_type,
            logits=outputs["end_logits"],
            text_seq_length=text_seq_length,
            task=task,
        )

        if is_distillation:
            if (not is_first_distillation_phase) or is_1phase_distillation:
                teacher_start_logits = extract_text_logits(
                    model_type=model_type,
                    logits=teacher_outputs["start_logits"],
                    text_seq_length=text_seq_length,
                    task=task,
                )
                teacher_end_logits = extract_text_logits(
                    model_type=model_type,
                    logits=teacher_outputs["end_logits"],
                    text_seq_length=text_seq_length,
                    task=task,
                )

                if use_ce_loss_for_logits_distillation:
                    # Comparison to TinyBERT
                    start_loss = ce_loss_fn(start_logits, teacher_start_logits)
                    end_loss = ce_loss_fn(end_logits, teacher_end_logits)
                    logits_ce_loss += (start_loss + end_loss) / 2
                else:
                    # Distillation: ...afterwards one complete training only on logits
                    # Have to use log_softmax for predictions here, see: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
                    start_loss = kl_loss_fn(
                        F.log_softmax(start_logits, dim=-1),
                        F.softmax(teacher_start_logits, dim=-1),
                    )
                    end_loss = kl_loss_fn(
                        F.log_softmax(end_logits, dim=-1),
                        F.softmax(teacher_end_logits, dim=-1),
                    )
                    logits_kl_loss += (start_loss + end_loss) / 2
        else:
            # Fine-tuning
            # Calculate the loss for start and end separately
            start_loss = ce_loss_fn(start_logits, start_positions)
            end_loss = ce_loss_fn(end_logits, end_positions)
            # Combine the start and end losses
            loss_val = (start_loss + end_loss) / 2
            logits_ce_loss += loss_val

        accuracy = accuracy_vqa(
            start_logits, end_logits, start_positions, end_positions
        )

    losses = (
        attention_mse_loss * gamma,
        hidden_states_mse_loss * alpha,
        embedding_mse_loss * beta,
        logits_kl_loss * delta,
        logits_ce_loss,
    )
    total_loss = torch.tensor([0.0], device=device)
    total_loss += sum(losses)

    return total_loss, accuracy, losses


def train_loop(
    model,
    teacher_model,  # only required for distillation
    vocab_lookup,  # only for distillation but optional there
    is_first_distillation_phase: bool,  # only required for distillation
    device,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    ed: ExperimentDefinition,
    optimizer,
    progress: Progress,
    start_epoch=0,
    debug_run=False,
):
    # Prepare
    is_distillation = (
        ed.mode == DistillMode.DISTILLATION
        or ed.mode == DistillMode.DISTILLATION_IN_1PHASE
    )
    is_1phase_distillation = ed.mode == DistillMode.DISTILLATION_IN_1PHASE

    if is_distillation:
        assert (
            teacher_model is not None
        ), "Teacher model required in DistillMode.DISTILLATION"

    # Setup LR scheduler
    num_training_steps = ed.num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=int(
            num_training_steps * DEFAULTS.LR_SCHEDULE_WARMUP_STEPS_FRAC_OF_TOTAL
        ),
        num_training_steps=num_training_steps,
        initial_lr=DEFAULTS.LR_START,
        final_lr=DEFAULTS.LR_END,
    )

    steps = start_epoch * len(train_dataloader)
    running_total_loss_train = 0.0
    running_attention_mse_loss_train = 0.0
    running_hidden_states_mse_loss_train = 0.0
    running_embedding_mse_loss_train = 0.0
    running_logits_kl_loss_train = 0.0
    running_logits_ce_loss_train = 0.0
    running_acc_train = 0.0
    best_accuracy = 0

    # These two lines are only needed for small datasets, where an epoch finished before TRAIN_STEP_LOG_INTERVAL is reached
    avg_total_loss_train = 0
    avg_acc_train = 0
    prev_avg_total_loss_train = 0
    prev_avg_acc_train = 0

    prev_eval_loss = 0
    prev_eval_acc = 0
    cur_eval_loss = 0
    cur_eval_acc = 0

    epoch_task = progress.add_task("[blue]Epoch", total=ed.num_epochs)

    for epoch in range(ed.num_epochs):
        train_task = progress.add_task("[cyan]Train...", total=len(train_dataloader))

        # TRAINING
        model.train()
        for inputs in train_dataloader:
            # FORWARD PASS
            optimizer.zero_grad()

            # Move inputs to device
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            # Convert the original token IDs to the new token IDs for the student vocab
            student_input_ids = (
                vocab_lookup[inputs["input_ids"]]
                if vocab_lookup is not None
                else inputs["input_ids"]
            )
            text_seq_length = student_input_ids.shape[1]

            # SER shape: batch_size, seq_len, num_classes
            # VQA shape: batch_size, seq_len
            # Model is either teacher (DistillMode.FINETUNE_TEACHER) or student (DistillMode.FINETUNE_STUDENT or DistillMode.DISTILLATION)
            outputs: dict[str, torch.Tensor] = forward(
                model=model,
                model_type=ed.model,
                output_internals=is_distillation,
                input_ids=student_input_ids,
                bbox=inputs["bbox"],
                attention_mask=inputs["attention_mask"],
                pixel_values=(
                    inputs["pixel_values"]
                    if ed.model == DUModel.LayoutLMv3_TextAndImage
                    else None
                ),
            )

            labels = inputs["labels"]

            teacher_outputs = None
            if is_distillation:
                with torch.no_grad():
                    # This is always teacher model
                    teacher_outputs: dict[str, torch.Tensor] = forward(
                        model=teacher_model,
                        model_type=ed.model,
                        output_internals=True,
                        input_ids=inputs["input_ids"],
                        bbox=inputs["bbox"],
                        attention_mask=inputs["attention_mask"],
                        pixel_values=(
                            inputs["pixel_values"]
                            if ed.model == DUModel.LayoutLMv3_TextAndImage
                            else None
                        ),
                    )

            total_loss, accuracy, losses = calculate_loss_and_accuracy(
                outputs=outputs,
                teacher_outputs=teacher_outputs,
                labels=labels,
                text_seq_length=text_seq_length,
                model_type=ed.model,
                task=ed.task,
                num_labels=ed.num_labels,
                is_distillation=is_distillation,
                is_first_distillation_phase=is_first_distillation_phase,
                is_1phase_distillation=is_1phase_distillation,
                student_layer_mapping=ed.student_layer_map,
                alpha=ed.alpha,
                beta=ed.beta,
                gamma=ed.gamma,
                delta=ed.delta,
                device=device,
                use_ce_loss_for_logits_distillation=ed.use_ce_loss_for_logits_distillation,
            )

            total_loss.backward()

            optimizer.step()

            (
                attention_mse_loss,
                hidden_states_mse_loss,
                embedding_mse_loss,
                logits_kl_loss,
                logits_ce_loss,
            ) = losses
            running_attention_mse_loss_train += attention_mse_loss.cpu().item()
            running_hidden_states_mse_loss_train += hidden_states_mse_loss.cpu().item()
            running_embedding_mse_loss_train += embedding_mse_loss.cpu().item()
            running_logits_kl_loss_train += logits_kl_loss.cpu().item()
            running_logits_ce_loss_train += logits_ce_loss.cpu().item()

            running_total_loss_train += total_loss.cpu().item()
            running_acc_train += accuracy

            if (steps + 1) % TRAIN_STEP_LOG_INTERVAL == 0:
                avg_attention_mse_loss_train = (
                    running_attention_mse_loss_train / TRAIN_STEP_LOG_INTERVAL
                )
                avg_hidden_states_mse_loss_train = (
                    running_hidden_states_mse_loss_train / TRAIN_STEP_LOG_INTERVAL
                )
                avg_embedding_mse_loss_train = (
                    running_embedding_mse_loss_train / TRAIN_STEP_LOG_INTERVAL
                )
                avg_logits_kl_loss_train = (
                    running_logits_kl_loss_train / TRAIN_STEP_LOG_INTERVAL
                )
                avg_logits_ce_loss_train = (
                    running_logits_ce_loss_train / TRAIN_STEP_LOG_INTERVAL
                )

                avg_total_loss_train = (
                    running_total_loss_train / TRAIN_STEP_LOG_INTERVAL
                )
                avg_acc_train = running_acc_train / TRAIN_STEP_LOG_INTERVAL

                # desc = f'[blue]Epoch {start_epoch + epoch + 1} - loss_train={prev_avg_total_loss_train:.4f}->{avg_total_loss_train:.4f}, acc_train={prev_avg_acc_train:.4f}->{avg_acc_train:.4f}, loss_eval={prev_eval_loss:.4f}->{cur_eval_loss:.4f}, acc_eval={prev_eval_acc:.4f}->{cur_eval_acc:.4f}'
                # progress.update(epoch_task, description=desc, advance=0, refresh=True)
                print(
                    f"Epoch {start_epoch + epoch + 1} - loss_train={prev_avg_total_loss_train:.4f}->{avg_total_loss_train:.4f}, acc_train={prev_avg_acc_train:.4f}->{avg_acc_train:.4f}, loss_eval={prev_eval_loss:.4f}->{cur_eval_loss:.4f}, acc_eval={prev_eval_acc:.4f}->{cur_eval_acc:.4f}"
                )

                prev_avg_total_loss_train = avg_total_loss_train
                prev_avg_acc_train = avg_acc_train

                if not debug_run:
                    wandb.log(
                        {
                            "train/total_loss": avg_total_loss_train,
                            "train/attention_mse_loss": avg_attention_mse_loss_train,
                            "train/hidden_states_mse_loss": avg_hidden_states_mse_loss_train,
                            "train/embedding_mse_loss": avg_embedding_mse_loss_train,
                            "train/logits_kl_loss": avg_logits_kl_loss_train,
                            "train/logits_ce_loss": avg_logits_ce_loss_train,
                            "train/accuracy": avg_acc_train,
                            "LR": scheduler.get_last_lr()[0],
                        },
                        step=steps,
                    )

                # Reset metrics for next interval
                running_attention_mse_loss_train = 0.0
                running_hidden_states_mse_loss_train = 0.0
                running_embedding_mse_loss_train = 0.0
                running_logits_kl_loss_train = 0.0
                running_logits_ce_loss_train = 0.0

                running_total_loss_train = 0.0
                running_acc_train = 0.0

            steps += 1
            scheduler.step()
            progress.update(train_task, advance=1)
            progress.refresh()

        progress.remove_task(train_task)

        # EVALUATING
        eval_loss, eval_acc = eval_model(
            model,
            teacher_model=teacher_model,
            vocab_lookup=vocab_lookup,
            eval_dataloader=eval_dataloader,
            cur_train_steps=steps,
            device=device,
            is_first_distillation_phase=is_first_distillation_phase,
            ed=ed,
            debug_run=debug_run,
            progress=progress,
        )
        prev_eval_loss, prev_eval_acc = cur_eval_loss, cur_eval_acc
        cur_eval_loss, cur_eval_acc = eval_loss, eval_acc

        # desc = f'[blue]Epoch {start_epoch + epoch + 1} - loss_train={prev_avg_total_loss_train:.4f}->{avg_total_loss_train:.4f}, acc_train={prev_avg_acc_train:.4f}->{avg_acc_train:.4f}, loss_eval={prev_eval_loss:.4f}->{cur_eval_loss:.4f}, acc_eval={prev_eval_acc:.4f}->{cur_eval_acc:.4f}'
        # progress.update(epoch_task, advance=1, description=desc, refresh=True)
        # progress.refresh()
        progress.update(epoch_task, advance=1)
        print(
            f"Epoch {start_epoch + epoch + 1} - loss_train={prev_avg_total_loss_train:.4f}->{avg_total_loss_train:.4f}, acc_train={prev_avg_acc_train:.4f}->{avg_acc_train:.4f}, loss_eval={prev_eval_loss:.4f}->{cur_eval_loss:.4f}, acc_eval={prev_eval_acc:.4f}->{cur_eval_acc:.4f}"
        )

        # Check if this is the best accuracy we've seen so far
        if eval_acc > best_accuracy and not debug_run:
            best_accuracy = eval_acc

            # Save the model checkpoint
            print(
                f"New best model found at epoch {start_epoch+epoch+1} with accuracy: {eval_acc:.4f}. Saving model."
            )
            save_model(start_epoch + epoch, model, optimizer, best_accuracy, ed, "best")

    progress.remove_task(epoch_task)

    # -- All epochs finished
    return model, best_accuracy


def main(
    ed: ExperimentDefinition,
    dataloader_num_workers: int,
    progress: Progress,
    debug_run=False,
    passed_device=None,
    set_teacher_eval: bool = False,  # previously we didn't set teacher model to eval in distillation, this is for tests
):
    if debug_run:
        print(f"-" * 10)
        print(f"DEBUG RUN")
        print(f"-" * 10)

    # Set seed for reproducibility
    transformers.set_seed(0)

    # Initialize WandB
    if not debug_run:
        wandb.login()
        wandb.init(
            project=CONSTS.WANDB_PROJECT_ID,
            name=ed.run_name,
            config={
                "model": ed.model.name,
                "mode": ed.mode.name,
                "datasets": ed.dataset,
                "chatgpt_labels": ed.use_chatgpt_labels,
                "num_epochs": ed.num_epochs,
                "batch_size": DEFAULTS.BATCH_SIZE,
                "teacher_run_name": ed.teacher_run_name,
                "student_layer_map": ed.student_layer_map,
                "learning_rate_start": DEFAULTS.LR_START,
                "learning_rate_start": DEFAULTS.LR_END,
                "lr_schedule_warmup_steps_frac_of_total": DEFAULTS.LR_SCHEDULE_WARMUP_STEPS_FRAC_OF_TOTAL,
            },
        )

    device = passed_device or get_device()

    # Load datasets and set task config
    print(f"Loading dataset...")
    dataset = load_dataset(
        [ed.dataset], use_cache=True, use_chatgpt_labels=ed.use_chatgpt_labels
    )
    ds_conf = DATASET_CONF[ed.dataset]
    task = ds_conf.task
    num_labels = ds_conf.num_labels

    # Load vocab if given
    vocab_map, vocab_lookup = load_vocab_map(ed.vocab_name, device=device)

    # Create dataloaders
    train_dataloader = create_dataloader(
        dataset["train"],
        num_workers=dataloader_num_workers,
        batch_size=DEFAULTS.BATCH_SIZE,
        shuffle=True,
    )
    eval_dataloader = create_dataloader(
        dataset["eval"],
        num_workers=dataloader_num_workers,
        batch_size=DEFAULTS.BATCH_SIZE,
        shuffle=True,
    )

    # Load models
    is_distillation = (
        ed.mode == DistillMode.DISTILLATION
        or ed.mode == DistillMode.DISTILLATION_IN_1PHASE
    )
    is_1phase_distillation = ed.mode == DistillMode.DISTILLATION_IN_1PHASE

    print(f"Loading models...")
    model = get_model(
        model_type=ed.model,
        task=task,
        is_student=ed.mode != DistillMode.FINETUNE_TEACHER,
        num_labels=num_labels,
        vocab_map=vocab_map,
        device=device,
        teacher_run_name=ed.teacher_run_name,
        student_layer_map=ed.student_layer_map,
    )
    teacher_model = None
    if is_distillation:
        teacher_model = get_model(
            model_type=ed.model,
            task=task,
            is_student=False,
            num_labels=num_labels,
            vocab_map=None,
            device=device,
            teacher_run_name=ed.teacher_run_name,
            student_layer_map=ed.student_layer_map,
        )
        if set_teacher_eval:
            print(f"#### Setting teacher to eval mode")
            teacher_model.eval()

    # Make model directory
    ed.checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=DEFAULTS.LR_START,
    )

    # Training, best model on eval split is saved after each epoch.
    # For distillation training is split into two phases:
    # 	- Phase 1: calculate loss only based on attention scores and hidden states
    # 	- Phase 2: calculate loss only based on logits
    #
    print(f"Training...")
    model, best_accuracy = train_loop(
        model=model,
        teacher_model=teacher_model,
        vocab_lookup=vocab_lookup,
        is_first_distillation_phase=True,
        device=device,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        ed=ed,
        debug_run=debug_run,
        progress=progress,
    )
    final_epoch = ed.num_epochs - 1

    if is_distillation and not is_1phase_distillation:
        print(f"-- PHASE 2 Distillation Training --")
        model, best_accuracy = train_loop(
            model=model,
            teacher_model=teacher_model,
            vocab_lookup=vocab_lookup,
            is_first_distillation_phase=False,
            device=device,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            optimizer=optimizer,
            ed=ed,
            start_epoch=ed.num_epochs,
            debug_run=debug_run,
            progress=progress,
        )
        final_epoch = (2 * ed.num_epochs) - 1

    # Save model
    if not debug_run:
        save_model(final_epoch, model, optimizer, best_accuracy, ed, "final")

    # Finish WandB logger
    wandb.finish()


def save_model(
    epoch, model, optimizer, best_accuracy, ed: ExperimentDefinition, name: str
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "is_student": ed.mode != DistillMode.FINETUNE_TEACHER,
            "student_layer_map": ed.student_layer_map,
            "vocab_name": ed.vocab_name,
        },
        ed.checkpoint_path / f"{name}.pth",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SlimDoc Distillation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "run_name",
        type=str,
        help=f"Speficy a run name.",
    )

    parser.add_argument(
        "mode",
        type=int,
        help=f"Speficy a mode via number: {[v for v in DistillMode]}",
    )

    parser.add_argument(
        "dataset",
        type=str,
        # nargs='+',
        help=f"Speficy a datasets to use: {get_args(SUPPORTED_DATASET)}",
    )

    parser.add_argument(
        "--use_chatgpt_labels",
        action="store_true",
        default=False,
        help=(
            "If set, use the ground truth established by the teacher LLM (ChatGPT). "
            "By default, this is not set and the ground truth delivered with the datasets is used."
        ),
    )

    parser.add_argument(
        "--teacher_run_name",
        type=str,
        help=f"Speficy a run name to load the teacher model from. (Required for DistillMode.DISTILLATION)",
    )

    parser.add_argument(
        "--student_layer_map",
        type=str,
        nargs="+",
        default=DEFAULTS.DEFAULT_LAYER_MAP,
        help=f"Speficy how the encoder layers of the student map to the teacher (position in list is student layer, value is teachr layer, number of entries defines number of layers in student)",
    )

    parser.add_argument(
        "--single_phase_distillation",
        action="store_true",
        default=False,
        help="If set, both distillation phases are compressed into one (Logits-, Hidden states- and Attention scores-distillation)",
    )

    parser.add_argument(
        "--vocab",
        type=str,
        help=f"Speficy the filename of the vocabulary to use (without file ending). If none is given, the vocabulary of the student willnot be changed. Call liltdistill/data/vocab.py to create vocabs, these are saved under data/vocab",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help=f"Speficy the number of epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=f"Speficy a batch size.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "If set, model inputs and outputs will be printed and the batch_size will be set to 1. "
        ),
    )

    args = parser.parse_args()

    assert args.dataset in get_args(
        SUPPORTED_DATASET
    ), f"Unknown dataset: {args.dataset}"

    if args.debug:
        print(f"--- DEBUG MODE ---")
        args.batch_size = 1

    if args.mode == DistillMode.DISTILLATION.value:
        assert (
            args.teacher_run_name
        ), "Teacher run name needs to be specified for DistillMode.DISTILLATION"

    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    ed = ExperimentDefinition(
        dataset=args.dataset,
        mode=DistillMode(args.mode),
        use_chatgpt_labels=args.use_chatgpt_labels,
        teacher_run_name=args.teacher_run_name,
        student_layer_map=tuple([int(x) for x in args.student_layer_map]),
        vocab_name=args.vocab,
    )
    main(ed, args.debug)
