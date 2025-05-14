import argparse
from collections import Counter
import json
from typing import get_args
from transformers import AutoTokenizer

from slimdoc import ENV, SUPPORTED_DATASET
from slimdoc.data import get_dataset_descriptor

# from slimdoc.data.hf_dataset import load_dataset


def generate_vocab(
    datasets: list[SUPPORTED_DATASET], split: str, vocab_sizes: list[int], tokenizer
):
    # Preparation
    dataset = load_dataset(datasets, use_chatgpt_labels=False, use_cache=True)

    # Collect input IDs
    all_input_ids = []
    for sample in dataset[split]:
        input_ids = sample["input_ids"]
        all_input_ids.extend(input_ids)

    # Counte tokens and sort by frequency
    token_freq = Counter(all_input_ids)
    tokens_sorted_by_freq = [
        i[0] for i in sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    ]

    # Ensure special tokens are in new vocab
    for token_id, token in tokenizer.added_tokens_decoder.items():
        if token_id in tokens_sorted_by_freq:
            tokens_sorted_by_freq.remove(token_id)
        tokens_sorted_by_freq.insert(0, token_id)

    # Create vocabs of different sizes
    for vocab_size in vocab_sizes:
        this_vocab_tokens = tokens_sorted_by_freq[:vocab_size]
        ds_id = get_dataset_descriptor(datasets)
        vocab_fname = f"{ds_id}-{split}-{vocab_size}"
        vocab_path = ENV.VOCABS_DIR / f"{vocab_fname}.json"

        with open(vocab_path, "w") as f:
            json.dump(this_vocab_tokens, f)

        print(
            f"Generated vocab with name {vocab_fname}: Unique tokens in datasets {datasets} {split}={len(tokens_sorted_by_freq)}, saved {len(this_vocab_tokens)} tokens, dropped {len(tokens_sorted_by_freq) - len(this_vocab_tokens)} tokens"
        )


def read_vocab(vocab_fname: str) -> list[int]:
    with open(ENV.VOCABS_DIR / f"{vocab_fname}.json", "r") as f:
        vocab = json.load(f)
        print(f'Read vocabulary "{vocab_fname}" with {len(vocab)} entries.')
        return vocab


def parse_args():
    parser = argparse.ArgumentParser(
        description="LiLT Distillation - Vocab Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        help=f"Speficy one or more datasets to create the vocabulary from: {get_args(SUPPORTED_DATASET)}",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help=f"Speficy the size of the vocabulary.",
    )

    parser.add_argument(
        "--split", type=str, default="train", help=f"Specify the split to use."
    )

    args = parser.parse_args()

    for ds in args.datasets:
        assert ds in get_args(SUPPORTED_DATASET), f"Unknown dataset: {ds}"

    print(args)
    return args


if __name__ == "__main__":
    # generate_vocab(datasets=['SROIE'], split='train', vocab_sizes=[15000, 10000])
    # vocab = read_vocab('SROIE-train-10000')
    # print(vocab)
    args = parse_args()
    generate_vocab(datasets=args.datasets, split=args.split, vocab_sizes=args.sizes)
