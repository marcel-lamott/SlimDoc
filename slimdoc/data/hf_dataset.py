#
# Set cache path before importing HF
# (set it here because of trouble with VS Code SSH not reading env variables correctly)
#
import os
from typing import Literal, Optional

# os.environ['HF_HOME'] = '/data/dok/marcel/.cache/huggingface/'

import numpy as np
import torch
from transformers import AutoTokenizer
import datasets
from datasets import Features, Sequence, Value, Array3D
from transformers import LayoutLMv3Processor

from slimdoc import ENV, CONSTS, DATASET_CONF, SUPPORTED_DATASET, TASKS
from slimdoc.data.dataset import SERDataset, VQADataset
from slimdoc.data import get_dataset_descriptor

datasets.disable_caching()

EVAL_SIZE = 0.05  # this much training samples will be taken into eval split if there is no given with the dataset
SEED = 0
PREPROCESS_BATCH_SIZE = 32


"""
This class is responsible for:
    - ensuring there is an eval-split
    - possibly combining multiple datasets into one
"""


def _load_torch_dataset(
    dataset_name, split, use_chatgpt_labels, max_samples: Optional[int] = None
):
    dsconf = DATASET_CONF[dataset_name]
    task = dsconf.task
    if task == TASKS.SER:
        return SERDataset(dataset_name, split, use_chatgpt_labels=use_chatgpt_labels)
    elif task == TASKS.VQA:
        return VQADataset(
            dataset_name,
            split,
            use_chatgpt_labels=use_chatgpt_labels,
            max_samples=max_samples,
        )
    else:
        raise ValueError(f'Unknown dataset: "{dataset_name}"')


def _create_hf_dataset(
    base_datasets: list[SUPPORTED_DATASET],
    use_chatgpt_labels: bool = False,
    max_samples: Optional[int] = None,
):

    # 1 Load Tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(CONSTS.LILT_HF_MODEL_ID)
    processor = LayoutLMv3Processor.from_pretrained(
        CONSTS.LAYOUTLMV3_HF_MODEL_ID, apply_ocr=False
    )

    # 2 Load datasets
    # We need to build a datasets.DatasetDict from the torch.Dataset
    datasets_train = []
    datasets_test = []
    datasets_eval = []

    for ds in base_datasets:
        dsconf = DATASET_CONF[ds]
        train_ds = _load_torch_dataset(
            ds, "train", use_chatgpt_labels=use_chatgpt_labels, max_samples=max_samples
        )

        if dsconf.eval_split_name is not None:
            # dont use chatgpt labels for eval
            eval_ds = _load_torch_dataset(
                ds,
                dsconf.eval_split_name,
                use_chatgpt_labels=False,
                max_samples=max_samples,
            )
            datasets_eval.append(datasets.Dataset.from_list([x for x in eval_ds]))
            datasets_train.append(datasets.Dataset.from_list([x for x in train_ds]))
        else:
            # Split train into eval and train
            hf_ds = datasets.Dataset.from_list([x for x in train_ds])
            train_eval = hf_ds.train_test_split(
                test_size=EVAL_SIZE, seed=SEED, keep_in_memory=True
            )
            hf_dataset_train = train_eval["train"]
            hf_dataset_eval = train_eval["test"]

            datasets_eval.append(hf_dataset_eval)
            datasets_train.append(hf_dataset_train)

        if dsconf.has_test_split:
            # dont use chatgpt labels for test
            test_ds = _load_torch_dataset(
                ds, "test", use_chatgpt_labels=False, max_samples=max_samples
            )
            datasets_test.append(datasets.Dataset.from_list([x for x in test_ds]))

    # TODO: assert all have same task, same num labels etc

    hf_dataset_train = datasets.concatenate_datasets(datasets_train)
    hf_dataset_test = (
        datasets.concatenate_datasets(datasets_test) if any(datasets_test) else None
    )
    hf_dataset_eval = datasets.concatenate_datasets(datasets_eval)

    # Build HF dataset
    if hf_dataset_test is not None:
        dataset = datasets.DatasetDict(
            {
                "train": hf_dataset_train,
                "eval": hf_dataset_eval,
                "test": hf_dataset_test,
            }
        )
    else:
        dataset = datasets.DatasetDict(
            {"train": hf_dataset_train, "eval": hf_dataset_eval}
        )

    dataset = dataset.shuffle(
        seed=SEED, load_from_cache_file=False, keep_in_memory=True
    )  # we need the map function
    print(f"dataset.cleanup_cache_files(): {dataset.cleanup_cache_files()}")

    # 3 Preprocess dataset: tokenize, truncate, padding, align labels
    task = DATASET_CONF[base_datasets[0]].task

    def preprocess_function(batch_dict: dict[str, list]):
        if task == TASKS.SER:
            images = batch_dict["image"]
            texts = batch_dict["words"]
            word_labels = batch_dict["word_labels"]
            boxes = batch_dict["boxes"]
            sample_ids = batch_dict["sample_id"]
            dataset_names = batch_dict["dataset_name"]

            encodings = processor(
                images=images,
                text=texts,
                boxes=boxes,
                word_labels=word_labels,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # word_ids = [encodings.word_ids(i) for i in range(len(batch_dict['words']))]

            labels = encodings["labels"]

        elif task == TASKS.VQA:
            images = batch_dict["image"]
            questions = batch_dict["question"]
            texts = batch_dict["words"]
            answer_span_indices = batch_dict["answer_span_indices"]
            boxes = batch_dict["boxes"]
            sample_ids = batch_dict["sample_id"]
            dataset_names = batch_dict["dataset_name"]

            encodings = processor(
                images=images,
                text=questions,
                text_pair=texts,
                boxes=boxes,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            # word_ids = [encodings.word_ids(i) for i in range(len(batch_dict['words']))]

            # DUE datasets have multiple answer variants, but we only take the first one
            # Also, we only take the first two elements of each answer_span_indices, as the third element defines the mode with which they were retrieved: either exact or similar
            labels = torch.vstack(
                [
                    torch.tensor(all_answers_indices[0][:2])
                    for all_answers_indices in answer_span_indices
                ]
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        model_inputs = {
            "sample_id": sample_ids,
            "dataset_name": dataset_names,
            #'word_ids': word_ids,
            "input_ids": encodings["input_ids"],
            "bbox": encodings["bbox"],
            "attention_mask": encodings["attention_mask"],
            "pixel_values": encodings[
                "pixel_values"
            ],  # if 'pixel_values' in encodings else None,
            "labels": labels,
            #'answer_span_extraction_modes': answer_span_extraction_modes
        }

        # if not use_processor:
        #     del model_inputs['pixel_values']

        # print(model_inputs)
        # input()

        return model_inputs

    columns_to_remove = ["words", "boxes", "image"]
    if task == TASKS.SER:
        columns_to_remove.extend(["word_labels"])
    elif task == TASKS.VQA:
        columns_to_remove.extend(["question", "answers", "answer_span_indices"])

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=PREPROCESS_BATCH_SIZE,
        remove_columns=columns_to_remove,
        keep_in_memory=True,
    )  # , remove_columns=["label", "dataset", "verbalizer"]
    print(f"Keys of processed dataset: {list(tokenized_dataset['train'].features)}")
    print(dataset)
    return tokenized_dataset


def load_dataset(
    base_datasets: list[SUPPORTED_DATASET],
    use_chatgpt_labels=False,
    use_cache=True,
    max_samples: Optional[int] = None,
):
    assert (
        max_samples is None or use_cache is False
    ), f"max_samples!=None is not compatible with use_cache=True"

    # Get dataset descriptor
    ds_id = get_dataset_descriptor(base_datasets)
    ds_id = ds_id + ("-ChatGPTLabels" if use_chatgpt_labels else "")

    path = ENV.PROCESSED_DATASETS_DIR / ds_id

    # Load from disk if exists
    if path.exists() and use_cache:
        print(
            f"Load cached preprocessed dataset from disk for datasets: {base_datasets}..."
        )
        tokenized_dataset = datasets.load_from_disk(str(path))
        print(tokenized_dataset)
        return tokenized_dataset

    # Else create
    print(f"Create dataset: {ds_id}")
    tokenized_dataset = _create_hf_dataset(
        base_datasets=base_datasets,
        use_chatgpt_labels=use_chatgpt_labels,
        max_samples=max_samples,
    )

    # save datasets to disk for later easy loading
    tokenized_dataset.save_to_disk(str(path))
    return tokenized_dataset


if __name__ == "__main__":
    # #os.environ['HF_HOME'] = '/data/dok/marcel/.cache/huggingface/'
    # ds = load_dataset(['DocVQA'], use_cache=False, use_chatgpt_labels=False)
    # for sample in ds['train']:
    #     print(sample.keys())
    #     input()
    #     break

    ds = load_dataset(["SROIE"], use_cache=False, use_chatgpt_labels=False)
    for sample in ds["train"]:
        print(sample.keys())
        input()
        break
