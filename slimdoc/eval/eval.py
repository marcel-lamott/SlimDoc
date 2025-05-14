"""
Evaluates a single model on a given dataset(-combination)
"""

import argparse
from collections import defaultdict
from typing import get_args
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LayoutLMv3Processor
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


from slimdoc import (
    CONSTS,
    DATASET_CONF,
    DEFAULTS,
    ENV,
    SUPPORTED_DATASET,
    TASKS,
    DUModel,
)
from slimdoc.data.hf_dataset import load_dataset
from slimdoc.data.utils import create_dataloader, load_vocab_map
from slimdoc.eval.due_eval import evaluate_due_results
from slimdoc.eval.utils import run_name_to_experiment_definition
from slimdoc.model import get_model, forward
from slimdoc.train import ExperimentDefinition
from slimdoc.train.utils import (
    extract_text_logits,
    get_device,
)


def calc_f1(predictions, labels):
    # Labels are of same shape, calculate F1
    from sklearn.metrics import f1_score

    # Assuming predictions and labels are torch tensors of shape (batch_size, seq_len)
    predictions_flat = (
        predictions.view(-1).cpu().numpy()
    )  # Flatten and move to CPU if necessary
    labels_flat = labels.view(-1).cpu().numpy()  # Flatten and move to CPU if necessary

    # print(f'predictions_flat:', predictions_flat)
    # print(f'labels_flat:', labels_flat)

    # Mask out padding tokens (assuming padding token label is -100)
    mask = labels_flat != -100

    # Calculate F1 score
    f1 = f1_score(
        labels_flat[mask], predictions_flat[mask], average="weighted"
    )  # or 'micro' for token-level score

    # print(f'labels_flat[mask]:', labels_flat[mask])
    # print(f'predictions_flat[mask]:', predictions_flat[mask])
    # print(f1)
    # input()

    return f1


def extract_predictions(inputs, outputs, task, tokenizer, model_type, text_seq_length):
    # Flatten the tokens and labels for CrossEntropyLoss
    if task == TASKS.SER:
        # SER shape: batch_size, seq_len
        logits: torch.Tensor = extract_text_logits(
            model_type=model_type,
            logits=outputs["logits"],
            text_seq_length=text_seq_length,
            task=task,
        )

        # Apply softmax and argmax along the class dimension to get the predicted class index
        predictions = torch.argmax(logits, dim=-1)  # shape: (batch_size, seq_len)
        return predictions

    elif task == TASKS.VQA:
        # VQA shape: batch_size, seq_len
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
        predicted_start_positions = torch.argmax(
            start_logits, dim=1
        )  # Shape: [batch_size]
        predicted_end_positions = torch.argmax(end_logits, dim=1)  # Shape: [batch_size]
        selected_ranges = [
            inputs["input_ids"][i, start:end]
            for i, (start, end) in enumerate(
                zip(predicted_start_positions, predicted_end_positions)
            )
        ]
        decoded_answers = tokenizer.batch_decode(selected_ranges)
        return decoded_answers


def evaluate_model(
    run_name: str,
    split: str = "test",
    batch_size: int = 16,
    complete_due_datasets: bool = False,
    dataloader_num_workers: int = DEFAULTS.DATALOADER_NUM_WORKERS,
    progress: any = None,
) -> dict[str, float]:
    #
    # Step 1: prepare (load datasets and model)
    #
    device = get_device()

    # Parse run_name to infert model type and dataset
    ed: ExperimentDefinition = run_name_to_experiment_definition(run_name)
    dataset_name: SUPPORTED_DATASET = ed.dataset
    model_type: DUModel = ed.model

    # Load datasets and set task config
    ds_conf = DATASET_CONF[dataset_name]
    # We set use_chatgpt_labels to false because we want to evaluate against the GT labels
    dataset = load_dataset([dataset_name], use_cache=True, use_chatgpt_labels=False)
    task = ds_conf.task
    num_labels = ds_conf.num_labels

    # Load tokenizer
    processor = LayoutLMv3Processor.from_pretrained(
        CONSTS.LAYOUTLMV3_HF_MODEL_ID, apply_ocr=False
    )
    tokenizer = processor.tokenizer

    # Load checkpoint for access to metadata
    chk_path = ENV.MODELS_DIR / run_name / "best.pth"
    checkpoint = torch.load(chk_path, map_location=device)
    vocab_name = checkpoint["vocab_name"]
    is_student = checkpoint["is_student"]
    student_layer_map = checkpoint["student_layer_map"]

    # Load vocab and data
    vocab_map, vocab_lookup = load_vocab_map(vocab_name, device)
    test_dataloader = create_dataloader(
        dataset["test"],
        num_workers=dataloader_num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    # Load model and checkpoint
    model = get_model(
        model_type=model_type,
        task=task,
        is_student=is_student,
        num_labels=num_labels,
        vocab_map=vocab_map,
        device=device,
        teacher_run_name=None,  # we dont intialize the student, we load checkpoint
        student_layer_map=student_layer_map,
    )

    # Load finetuned checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    #
    # Step 2: get predictions on data
    #
    sample_ids_preds = []
    with progress or Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        "[yellow]({task.completed}/{task.total})",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        eval_task = progress.add_task(
            f"[cyan]Evaluating {run_name} on {split}", total=len(test_dataloader)
        )

        with torch.no_grad():
            for inputs in test_dataloader:
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

                # Forward pass
                outputs: dict[str, torch.Tensor] = forward(
                    model=model,
                    model_type=model_type,
                    output_internals=False,
                    input_ids=student_input_ids,
                    bbox=inputs["bbox"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=(
                        inputs["pixel_values"]
                        if model_type == DUModel.LayoutLMv3_TextAndImage
                        else None
                    ),
                )

                labels = inputs["labels"]

                predictions = extract_predictions(
                    inputs=inputs,
                    outputs=outputs,
                    task=task,
                    tokenizer=tokenizer,
                    model_type=model_type,
                    text_seq_length=text_seq_length,
                )

                sample_ids_preds.append(
                    (inputs["sample_id"], inputs["dataset_name"], predictions, labels)
                )
                progress.update(eval_task, advance=1)

        progress.remove_task(eval_task)

    #
    # Step 3: evaluate predictions
    #
    score = __eval_final(sample_ids_preds, task, split, complete_due_datasets)
    return score


def __eval_final(
    sample_ids_preds_labels, task, split, complete_due_datasets: bool = False
):
    results = dict()
    # dataset name to sample id to answer span
    due_predictions = dict()
    ser_scores = list()

    # task should be same for all evaluated samples
    dsnames = set()

    # For each batch
    for sample_ids, dataset_names, predictions, labels in sample_ids_preds_labels:
        # For each sample in batch
        for i, (sample_id, dataset_name, prediction, label) in enumerate(
            zip(sample_ids, dataset_names, predictions, labels)
        ):
            dsnames.add(dataset_name)

            if task == TASKS.SER:
                # Predictions is batch of token class predictions of shape (batch_size, seq_len)
                f1 = calc_f1(prediction, label)
                # print(f'F1 for batch: {f1}')

                # these datasets should never be mixed
                assert len(set([ds for ds in dataset_names])) == 1

                ser_scores.append(f1)
            elif task == TASKS.VQA:
                # Prediction is answer span
                due_predictions[sample_id] = prediction

    assert len(dsnames) == 1

    if task == TASKS.SER:
        score = np.mean(ser_scores)
    elif task == TASKS.VQA:
        score = evaluate_due_results(
            dataset_name=list(dsnames)[0],
            split=split,
            predictions=due_predictions,
            only_our_samples=not complete_due_datasets,
        )

    return score


def _parse_args():
    parser = argparse.ArgumentParser(
        description="SlimDoc Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "run_name",
        type=str,
        help=f"Speficy a run name.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help=f"Speficy the split of the dataset on which to evaluate. The dataset is infered automatically from the run_name.",
    )

    parser.add_argument(
        "--batch_size", type=int, default=16, help=f"Speficy the batch size."
    )

    parser.add_argument(
        "--complete_due_datasets",
        action="store_true",
        default=False,
        help="If set, we don't restrict evaluation of DUE datasets to our filtered subset but instead use the whole dataset.",
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=DEFAULTS.DATALOADER_NUM_WORKERS,
        help=f"Speficy the number of workers for parallel dataloading to speed up training. When set to 0, no parallelism is used and no new processes are spawned.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    score = evaluate_model(
        args.run_name,
        args.split,
        args.batch_size,
        args.complete_due_datasets,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    print(score)
