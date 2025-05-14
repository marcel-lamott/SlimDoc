import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import LayoutLMv3Tokenizer
import Levenshtein as lev

from slimdoc import CONSTS, SUPPORTED_DATASET, DEFAULTS
from slimdoc.data.vocab import read_vocab


def __find_answer_indices_exact(
    tokenizer, question, word_list, boxes, answer, _encoding=None
) -> tuple[int, int]:
    def find_sub_seq(context, pattern):
        pattern_length = len(pattern)
        context_length = len(context)

        for i in range(context_length - pattern_length + 1):
            if context[i : i + pattern_length] == pattern:
                return i, i + pattern_length

        return -1, -1

    encoding = _encoding or tokenizer(question, word_list, boxes=boxes, truncation=True)

    ans_enc = tokenizer(answer, [""], boxes=[[0, 0, 0, 0]])
    ans_enc = [
        a for a, b in zip(ans_enc.input_ids, ans_enc.sequence_ids()) if b is not None
    ]

    start_index, end_index = find_sub_seq(encoding.input_ids, ans_enc)

    return start_index, end_index


def __find_most_similar_span_indices(
    tokenizer, question, word_list, boxes, answer, _encoding=None
):
    # Step 1: Tokenize the document with original case)
    encoding = _encoding or tokenizer(
        question, word_list, boxes=boxes, truncation=True, return_offsets_mapping=True
    )
    doc_offsets = encoding["offset_mapping"]

    # Step 2: Initialize variables to track the best match
    min_edit_distance = float("inf")
    answer_len = len(answer)

    # Step 3: Iterate over possible start positions in the document text
    document = " ".join(word_list)
    for i in range(len(document) - answer_len + 1):
        # Extract a candidate span from document of the same length as the answer
        candidate_span = document[i : i + answer_len]

        # Compute edit distance between the candidate and the answer (case-insensitive)
        distance = lev.distance(candidate_span.lower(), answer.lower())

        # Update if this span has a lower edit distance
        if distance < min_edit_distance:
            min_edit_distance = distance
            # Record character start and end positions of the best span
            best_start_char_idx = i
            best_end_char_idx = i + answer_len

    # Step 4: Map character indices of best match to token indices
    start_token_idx = -1
    end_token_idx = -1
    for j, (start, end) in enumerate(doc_offsets):
        if start <= best_start_char_idx < end:
            start_token_idx = j
        if start < best_end_char_idx <= end:
            end_token_idx = j
            break

    return start_token_idx, end_token_idx


def find_answer_indices(
    tokenizer, question, word_list, boxes, answer, allow_similarity_based
):
    """First tries to find the answer exact, if this does not work the span with the lowest edit distance is selected as answer to the question"""
    # encoding = tokenizer(question, word_list, boxes=boxes, truncation=True, return_offsets_mapping=True)
    # start_idx, end_idx = __find_answer_indices_exact(tokenizer, question, word_list, boxes, answer, encoding)
    # if start_idx != -1 and end_idx != -1:
    #     return start_idx, end_idx, 'exact'
    # elif allow_similarity_based:
    #     start_idx, end_idx = __find_most_similar_span_indices(tokenizer, question, word_list, boxes, answer, encoding)
    #     return start_idx, end_idx, 'similar'

    # return start_idx, end_idx, 'exact'

    encoding = tokenizer(
        question, word_list, boxes=boxes, truncation=True, return_offsets_mapping=True
    )
    start_idx, end_idx = __find_answer_indices_exact(
        tokenizer, question, word_list, boxes, answer, encoding
    )
    if start_idx != -1 and end_idx != -1:
        return start_idx, end_idx
    elif allow_similarity_based:
        start_idx, end_idx = __find_most_similar_span_indices(
            tokenizer, question, word_list, boxes, answer, encoding
        )
        return start_idx, end_idx

    return start_idx, end_idx


def _collate_fn(batch):
    valid_fields = {
        "input_ids",
        "bbox",
        "attention_mask",
        "pixel_values",
        "labels",
        "sample_id",
        "dataset_name",
    }
    inputs = {
        k: (
            torch.tensor(data=[sample[k] for sample in batch])
            if not isinstance(batch[0][k], str)
            else [sample[k] for sample in batch]
        )
        for k in valid_fields
    }
    return inputs


def create_dataloader(
    dataset: Dataset, num_workers: int, batch_size: int, shuffle: bool = True
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
        persistent_workers=False,
    )


def load_vocab_map(vocab_name: str, device):
    # Load vocab if given
    vocab_map = None
    vocab_lookup = None
    if vocab_name is not None:
        vocab_map = read_vocab(vocab_name)

        # Create lookup
        tokenizer = LayoutLMv3Tokenizer.from_pretrained(
            CONSTS.LAYOUTLMV3_HF_MODEL_ID, apply_ocr=False
        )
        unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
        vocab_lookup = torch.full((max(vocab_map) + 1,), unk_token_id, device=device)
        vocab_lookup[vocab_map] = torch.arange(0, len(vocab_map), device=device)

    return vocab_map, vocab_lookup
