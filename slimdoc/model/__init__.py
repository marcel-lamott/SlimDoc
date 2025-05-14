from typing import Union

import torch

from slimdoc import (
    ENV,
    TASKS,
    DUModel,
)
from slimdoc.model.head_models import (
    DUModelForExtractiveQA,
    DUModelForTokenClassification,
)
from slimdoc.model.student import create_student


def get_model(
    model_type: DUModel,
    task: TASKS,
    is_student: bool,
    num_labels: Union[int, None],
    vocab_map,
    device: Union[int, torch.device],
    teacher_run_name: Union[str, None],
    student_layer_map: Union[dict[int, int], None],
):
    load_pretrained = teacher_run_name is None

    msg = f"LOAD MODEL: {model_type=} {task=} {load_pretrained=} {is_student=} {teacher_run_name=}"
    if task == TASKS.SER:
        model = DUModelForTokenClassification(
            num_labels=num_labels,
            model_type=model_type,
            load_pretrained_hf=load_pretrained,
        )
    elif task == TASKS.VQA:
        model = DUModelForExtractiveQA(
            model_type=model_type, load_pretrained_hf=load_pretrained
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    if teacher_run_name:
        # Load pretrained teacher -> load best checkpoint from training
        chk_path = ENV.MODELS_DIR / teacher_run_name / "best.pth"
        checkpoint = torch.load(chk_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    # create student model
    if is_student:
        model = create_student(
            model_type=model_type,
            teacher=model,
            layer_map=student_layer_map,
            vocab_map=vocab_map,
        )

    model.to(device)
    print(msg + f" {model.config.num_hidden_layers=}")
    return model


def forward(
    model,
    model_type: DUModel,
    output_internals: bool,
    input_ids,
    bbox,
    attention_mask,
    pixel_values,
):
    # LayoutLMv3
    if (
        model_type == DUModel.LayoutLMv3_TextAndImage
        or model_type == DUModel.LayoutLMv3_TextOnly
        or model_type == DUModel.LayoutLMv3_NoVisionNoLayout
    ):
        # override Bboxes for DUModel.LiLT_TextFlow_NoLayout (inplace)
        if model_type == DUModel.LayoutLMv3_NoVisionNoLayout:
            bbox = torch.zeros_like(bbox)

        # Only LayoutLMv3 accepts image modality (parameter pixel_values)
        outputs: dict[str, torch.Tensor] = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=(
                pixel_values if model_type == DUModel.LayoutLMv3_TextAndImage else None
            ),
            output_attentions=output_internals,
            output_hidden_states=output_internals,
        )
    # LiLT
    elif (
        model_type == DUModel.LiLT_TextFlow
        or model_type == DUModel.LiLT_TextAndLayoutFlow
        or model_type == DUModel.LiLT_TextFlow_NoLayout
    ):
        # override Bboxes for DUModel.LiLT_TextFlow_NoLayout (inplace)
        if model_type == DUModel.LiLT_TextFlow_NoLayout:
            bbox = torch.zeros_like(bbox)

        outputs: dict[str, torch.Tensor] = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            output_attentions=output_internals,
            output_hidden_states=output_internals,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type=}")

    return outputs
