"""
Wie die ChatGPT labels evaluieren?
- Müssen einfach die Labels auslesen (siehe chatgptlabels.py) und den samples zuordnen
- Dann auf train evaluieren
"""

import argparse
from collections import defaultdict
from typing import get_args
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LayoutLMv3Processor

from slimdoc import CONSTS, DATASET_CONF, DEFAULTS, ENV, SUPPORTED_DATASET, TASKS
from slimdoc.data.chatgptlabels import get_label_value, read_chatgpt_labels
from slimdoc.data.hf_dataset import load_dataset
from slimdoc.data.models import DUEDatasetRow, SROIERow
from slimdoc.data.reader.due import read_due_dataset
from slimdoc.data.reader.sroie import read_sroie
from slimdoc.data.utils import create_dataloader
from slimdoc.data.vocab import read_vocab
from slimdoc.eval.due_eval import evaluate_due_results
from slimdoc.eval.chatgptlabels.sroie_eval import (
    evaluate_address,
    evaluate_company,
    evaluate_currency,
    evaluate_date,
)
from slimdoc.train.utils import extract_text_logits


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


def evaluate_chatpgt_labels(
    dataset: SUPPORTED_DATASET, split: str = "train", only_custom_due: bool = False
) -> dict[str, float]:

    assert (
        split == "train" or split == "test"
    ), f"ChatGPT labels are only available for train or test splits"

    # Preparation
    ds_conf = DATASET_CONF[dataset]
    device = torch.device("cpu")

    # We set use_chatgpt_labels to false because we want to evaluate against the GT labels
    dataset_gt = load_dataset([dataset], use_cache=True, use_chatgpt_labels=False)
    dataset_preds = load_dataset([dataset], use_cache=True, use_chatgpt_labels=True)
    task = ds_conf.task
    num_labels = ds_conf.num_labels

    # Load data
    BATCH_SIZE = 1
    dataloader_gt = create_dataloader(
        dataset_gt[split],
        num_workers=DEFAULTS.DATALOADER_NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    dataloader_preds = create_dataloader(
        dataset_preds[split],
        num_workers=DEFAULTS.DATALOADER_NUM_WORKERS,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    #
    # Step 2: collect predictions on data
    #
    sample_ids_preds = []
    for i_batch, (inputs_gt, inputs_preds) in tqdm(
        enumerate(zip(dataloader_gt, dataloader_preds)),
        total=len(dataloader_gt),
        desc=f"Evaluate {split}",
        leave=False,
    ):
        labels = inputs_gt["labels"]
        predictions = inputs_preds["labels"]

        torch.set_printoptions(threshold=10_000)
        print(f"{labels=}")
        print(f"{predictions=}")
        f1 = calc_f1(predictions, labels)
        print(f"{f1=}")
        input()

        # For SROIE we can take the predictions from the dataloader
        # For DUE...

        # predictions = extract_predictions(inputs=inputs, outputs=outputs, task=task, tokenizer=tokenizer, model_type=model_type, text_seq_length=text_seq_length)
        sample_ids_preds.append(
            (inputs_gt["sample_id"], inputs_gt["dataset_name"], predictions, labels)
        )

    #
    # Step 3: evaluate predictions
    #
    score = __eval_final(sample_ids_preds, task, split, only_custom_due)
    return score


def __eval_final(sample_ids_preds_labels, task, split, only_custom_due: bool = False):
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
            only_our_samples=only_custom_due,
        )

    return score


def eval_sroie(split):
    """
    {
            "company": "BOOK TA .K(TAMAN DAYA) SDN BND",
            "date": "25/12/2018 8:13:39 PM",
            "address": "NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR",
            "total": "9.00"
    }
    """
    chatgptlabels = read_chatgpt_labels(dataset_name="SROIE", split=split)

    data = read_sroie(split=split, use_chatgpt_labels=False)
    r: SROIERow

    accuracy_sum = 0
    for r in data:

        gt_company = r.entities["company"]
        pred_company = get_label_value(
            docid_to_labels=chatgptlabels, doc_id=r.id, key="company"
        )
        res_company = evaluate_company(gt_company, pred_company)
        # print(res_company, gt_company, pred_company)

        gt_date = r.entities["date"]
        pred_date = get_label_value(
            docid_to_labels=chatgptlabels, doc_id=r.id, key="date"
        )
        res_date = evaluate_date(gt_date, pred_date)
        # print(res_date, gt_date, pred_date)

        gt_address = r.entities["address"]
        pred_address = get_label_value(
            docid_to_labels=chatgptlabels, doc_id=r.id, key="address"
        )
        res_address = evaluate_address(gt_address, pred_address)
        # print(res_address, gt_address, pred_address)

        gt_total = r.entities["total"]
        pred_total = get_label_value(
            docid_to_labels=chatgptlabels, doc_id=r.id, key="total"
        )
        res_total = evaluate_currency(gt_total, pred_total)
        # print(res_total, gt_total, pred_total)

        # K3-113, JL IBRAHIM SULTAN 80300 JOHOR BAHRU JOHOR ---- K3-113,JL IBRAHIM SULTAN, 80300 JOHOR BAHRU, JOHOR
        # NO. 53, JALAN BESAR, 45600 BATANG BERJUNTAI SELANGOR DARUL EHSAN ----
        # NO. 53, JALAN BESAR, 45600 BATANG BERJUNTAI SELANGOR DARUL EHSAN

        sample_accuracy = sum(
            [
                1 if res_company else 0,
                1 if res_date else 0,
                1 if res_address else 0,
                1 if res_total else 0,
            ]
        ) / float(4)
        accuracy_sum += sample_accuracy
        # input()
    print(f"SROIE {split=} Accuracy: {accuracy_sum / len(data)}")


def eval_due(dataset_name, split):
    ds = read_due_dataset(
        dataset_name=dataset_name,
        split=split,
        ocr_engines=["microsoft_cv"],
        use_chatgpt_labels=True,
        max_image_side_length=None,
        skip_none=True,
    )
    r: DUEDatasetRow
    due_predictions = {r.annotation_id: r.annotation_values[0] for r in ds}
    score_custom = evaluate_due_results(
        dataset_name=dataset_name,
        split=split,
        predictions=due_predictions,
        only_our_samples=True,
    )
    score_all = evaluate_due_results(
        dataset_name=dataset_name,
        split=split,
        predictions=due_predictions,
        only_our_samples=False,
    )
    print(f"{dataset_name=} {split=} only_custom_due=True -> {score_custom}")
    print(f"{dataset_name=} {split=} only_custom_due=False -> {score_all}")


if __name__ == "__main__":
    # score = evaluate_chatpgt_labels(args.run_name, args.dataset, args.split, args.batch_size, args.only_custom_due)
    # print(score)
    # eval_sroie()
    eval_sroie("train")
    eval_sroie("test")
    eval_due("DocVQA", "train")
    eval_due("DocVQA", "test")
    eval_due("InfographicsVQA", "train")
    eval_due("InfographicsVQA", "test")
    eval_due("WikiTableQuestions", "train")
    eval_due("WikiTableQuestions", "test")
    # score = evaluate_chatpgt_labels(dataset='SROIE', split='test', only_custom_due=True)
    # print(score)
