import argparse
import multiprocessing
from typing import Union, get_args

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LayoutLMv3Tokenizer
from torch.utils.data import DataLoader
from datasets import DatasetDict

from slimdoc import (
    CONSTS,
    DEFAULTS,
    ENV,
    SUPPORTED_DATASET,
    TASKS,
    DUModel,
    DistillMode,
)
from slimdoc.data.vocab import read_vocab
from slimdoc.model.head_models import (
    DUModelForExtractiveQA,
    DUModelForTokenClassification,
)
from slimdoc.model.student import create_student


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def get_linear_schedule_with_min_lr(
    optimizer, num_warmup_steps, num_training_steps, initial_lr, final_lr
):
    # Calculate the decay factor based on the desired final learning rate
    decay_factor = (initial_lr - final_lr) / initial_lr

    # Adjusted lambda function for custom linear decay with final LR
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # Linearly decrease to final_lr instead of 0
        return max(final_lr / initial_lr, 1.0 - decay_factor * progress)

    # Use the LambdaLR scheduler with this custom lambda function
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def accuracy_ser(logits: torch.Tensor, labels: torch.Tensor):
    # Step 1: Get predicted labels by taking argmax over the class dimension
    predicted_labels = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_len)

    # Step 2: If using padding tokens, mask them out
    # Assuming padding tokens are labeled as -100 in labels (standard for Hugging Face)
    mask = labels != -100  # Creates a mask of shape (batch_size, seq_len)

    # Step 3: Calculate accuracy only on non-padding tokens
    correct = (predicted_labels == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()

    # print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def accuracy_vqa(start_logits, end_logits, start_positions, end_positions):
    # Get predicted start and end positions
    predicted_start_positions = torch.argmax(start_logits, dim=1)  # Shape: [batch_size]
    predicted_end_positions = torch.argmax(end_logits, dim=1)  # Shape: [batch_size]

    # Calculate correct predictions
    start_correct = predicted_start_positions == start_positions
    end_correct = predicted_end_positions == end_positions

    # Both start and end must be correct for a span to be counted as accurate
    span_correct = start_correct & end_correct  # Logical AND

    # Calculate accuracy
    accuracy = span_correct.float().mean().item()
    return accuracy


# For LayoutLMv3, the model appends image patch embedding tokens to the output sequence, so we have to retrieve only
# the part of the output sequence which correspondes to the text tokens
def extract_text_logits(
    model_type: DUModel, logits: torch.Tensor, text_seq_length: int, task: TASKS
) -> torch.Tensor:
    text_only_logits = logits[:, :text_seq_length]
    # VQA: output: 	[16, 2] 				input: [16, 512]
    # 				[BATCH, CLASS] 			input: [BATCH, SEQ]
    # SER: output: 	[16, 512, 5] 			input: [16, 512, 5]
    # 				[BATCH, SEQ, CLASS] 	input: [BATCH, SEQ, CLASS]
    if model_type == DUModel.LayoutLMv3_TextAndImage:
        assert (
            text_only_logits.shape[1] != logits.shape[1]
        ), f"Model output sequence seems to be missing image patch embeddings."
    else:
        assert (
            text_only_logits.shape[1] == logits.shape[1]
        ), f'Model output sequence is a different size than the input sequence ({text_only_logits.shape} vs {logits.shape}). This is not expected for DU model "{model_type}".'
    return text_only_logits
