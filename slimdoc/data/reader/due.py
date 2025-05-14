"""
All datasets of the DUE benchmark share a common format:
https://github.com/due-benchmark/du-schema/
https://duebenchmark.com/data
"""

import itertools
import json
import pathlib
import shelve

from typing import Counter, Literal, Optional, Union
from dataclasses import dataclass

import jsonlines
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from slimdoc import ENV, CONSTS
from slimdoc.data.chatgptlabels import get_label_value, read_chatgpt_labels
from slimdoc.data.imageutils import draw_bounding_boxes
from slimdoc.data.models import DatasetRowBase, OCRBox, DUEDatasetRow
from slimdoc.data.utils import find_answer_indices
from slimdoc.data.pdfutils import pdf_to_image


def __get_page_or_line_idx(ocr_results: dict, key: str, i_token: int):
    for i, startStopTokenIdxs in enumerate(
        ocr_results["structures"][key]["structure_value"]
    ):
        if i_token >= startStopTokenIdxs[0] and i_token < startStopTokenIdxs[1]:
            return i
    return None


def resize_bboxes(bboxes: list[OCRBox], original_size, new_size):
    """
    Rescale bounding boxes to match the resized image.

    :param bboxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
    :param original_size: Tuple (orig_width, orig_height)
    :param new_size: Tuple (new_width, new_height)
    :return: List of rescaled bounding boxes
    """
    orig_w, orig_h = original_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    resized_bboxes = [
        OCRBox(
            x0=int(b.x0 * scale_x),
            y0=int(b.y0 * scale_y),
            x2=int(b.x2 * scale_x),
            y2=int(b.y2 * scale_y),
            text=b.text,
        )
        for b in bboxes
    ]

    return resized_bboxes


def read_due_dataset(
    dataset_name: Literal["DocVQA", "InfographicsVQA", "TabFact", "WikiTableQuestions"],
    split: Literal["train", "dev", "test"],
    ocr_engines: list[Literal["djvu", "microsoft_cv", "tesseract"]],
    use_chatgpt_labels: bool = False,
    max_image_side_length: Union[
        int, None
    ] = 1000,  # downscale insanely large documents (some InfoVQA docs have a side length of 54k pixels..), if set to None no image will be loaded
    max_samples: Optional[int] = None,
    verbose: bool = True,
    skip_none: bool = False,  # only for testing purposes: if true, no samples are skipped
) -> list[DUEDatasetRow]:
    print(
        f"Reading DUE dataset {dataset_name} {split}, ChatGPT labels: {use_chatgpt_labels}"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONSTS.LILT_HF_MODEL_ID)

    base_path = ENV.DATASETS_DIR / dataset_name
    split_path: pathlib.Path = base_path / "data" / split
    document_path: pathlib.Path = split_path / "document.jsonl"
    documents_content_path: pathlib.Path = split_path / "documents_content.jsonl"

    # First read all document contents into a dict
    document_contents: dict = dict()
    with jsonlines.open(documents_content_path) as reader:
        for doc in tqdm(reader, leave=False, desc=f"Reading {dataset_name} {split}"):
            document_contents[doc["name"]] = doc["contents"]

    # Read ChatGPT generated labels if applicable
    docid_to_chatgpt_labels = None
    if use_chatgpt_labels:
        docid_to_chatgpt_labels = read_chatgpt_labels(dataset_name, split)

    # Now parse the document.jsonl file
    result: list[DUEDatasetRow] = []
    docs_count = 0
    samples_count = 0
    skipped_samples_count = 0
    answer_indices_extraction_mode = Counter()
    with jsonlines.open(document_path) as reader:
        for doc in tqdm(
            reader,
            leave=False,
            desc=f"Parsing {dataset_name} {split}",
            total=len(document_contents),
        ):
            docs_count += 1
            document_name = doc["name"]

            # Parse contents
            tool_name_to_contents = {
                c["tool_name"]: c for c in document_contents[document_name]
            }
            selected_ocr_engine = None
            for ocr_engine in ocr_engines:
                if ocr_engine in tool_name_to_contents:
                    selected_ocr_engine = ocr_engine
                    break

            if selected_ocr_engine is None:
                raise Exception(
                    f'No OCR engine from selection "{ocr_engines}" available for dataset {dataset_name}'
                )

            contents = tool_name_to_contents[selected_ocr_engine]
            del tool_name_to_contents  # free up memory
            ocr_results_key = (
                "common_format" if "common_format" in contents else "tokens_layer"
            )
            if ocr_results_key not in contents:
                print(f"{ocr_results_key} NOT FOUND, keys: {list(contents.keys())}")
            ocr_results = contents[ocr_results_key]

            # This returned the original PDF dimensions as given in the OCR results,
            # but since we convert it to an image, the dimensions change and we need to rescale the bboxes
            # # Get page dimensions
            # _, _, page_width, page_height = tuple(ocr_results['structures']['pages']['positions'][0])

            # read PDF as image
            _, _, orig_width, orig_height = tuple(
                ocr_results["structures"]["pages"]["positions"][0]
            )

            # Downscale the image to be input to LayoutLMv3
            image = None
            if max_image_side_length is not None:
                scale = max_image_side_length / max(orig_width, orig_height)
                scale = min(scale, 1)  # dont upscale images, only downscale
                new_w = int(orig_width * scale)
                new_h = int(orig_height * scale)

                # print(f'{scale=} orig:({orig_width}, {orig_height}) -> ({new_w}, {new_h})')
                # input()

                pdf_file_path: pathlib.Path = (
                    ENV.DATASETS_DIR / dataset_name / "pdfs" / f"{document_name}.pdf"
                )
                # image = pdf_to_image(pdf_file_path, target_size=(orig_width, orig_height))
                # width, height = image.size
                # assert orig_width == width
                # assert orig_height == height
                image = pdf_to_image(pdf_file_path, target_size=(new_w, new_h))

            # Collect BBoxes
            bboxes: list[(OCRBox, int, int)] = []
            for i_token, t in enumerate(ocr_results["tokens"]):
                bbox = ocr_results["positions"][i_token]

                # Get page/line index
                page_idx = __get_page_or_line_idx(
                    ocr_results=ocr_results, key="pages", i_token=i_token
                )
                line_idx = __get_page_or_line_idx(
                    ocr_results=ocr_results, key="lines", i_token=i_token
                )

                # Due errors in the dataset, for the last tokens this info is sometimes not given
                if page_idx is None:
                    page_idx = bboxes[-1][1] if len(bboxes) > 0 else 0
                if line_idx is None:
                    line_idx = bboxes[-1][2] if len(bboxes) > 0 else 0

                bbox = OCRBox(
                    x0=bbox[0], y0=bbox[1], x2=bbox[2], y2=bbox[3], text=t
                )  # (OCRBox(x0=bbox[0], y0=bbox[1], x2=bbox[2], y2=bbox[3], text=t), page_idx, line_idx)
                bboxes.append(bbox)

            # Rescale the bounding boxes to fit on our new image
            if max_image_side_length is not None:
                bboxes = resize_bboxes(
                    bboxes=bboxes,
                    original_size=(orig_width, orig_height),
                    new_size=(new_w, new_h),
                )
            # debug_img = draw_bounding_boxes(image, bboxes, show_text=True)
            # debug_img.show()

            doc_text_lower = contents["text"].lower()
            for i_annotation, a in enumerate(doc["annotations"]):
                samples_count += 1

                annotation_id = a["id"]
                annotation_key = a["key"]

                # incorporate ChatGPT labels
                vals = []
                if use_chatgpt_labels:
                    value = get_label_value(
                        docid_to_chatgpt_labels,
                        doc_id=document_name,
                        key=str(i_annotation),
                    )
                    # Use ChatGPT generated labels
                    if value is None:
                        # ChatGPT returned invalid json or did not return an answer or whatever..
                        skipped_samples_count += 1
                        continue
                    else:
                        # Here we can be sure it's a string
                        vals.append([value])
                else:
                    # Use normal labels
                    for x in a["values"]:
                        vals.append([x["value"]])

                        # DocVQA, InfoVQA
                        if "value_variants" in x:
                            vals.append(x["value_variants"])

                annotation_values = list(itertools.chain.from_iterable(vals))
                annotation_values = list(
                    set(annotation_values)
                )  # Filter potential duplicates

                # Filter to only the answers present in the text, such that we can identify start and end tokens
                found_answers = [
                    a for a in annotation_values if a.lower() in doc_text_lower
                ]
                b: OCRBox
                answer_indices = [
                    (
                        i_a,
                        find_answer_indices(
                            tokenizer=tokenizer,
                            question=annotation_key,
                            word_list=[b.text for b in bboxes],
                            boxes=[[b.x0, b.y0, b.x2, b.y2] for b in bboxes],
                            answer=a,
                            allow_similarity_based=False,
                        ),
                    )
                    for i_a, a in enumerate(found_answers)
                ]

                found_answers = [
                    found_answers[i]
                    for i, (start, stop) in answer_indices
                    if start != -1 and stop != -1
                ]
                answer_indices = [
                    (start, stop)
                    for i, (start, stop) in answer_indices
                    if start != -1 and stop != -1
                ]

                answer_indices_extraction_mode["exact"] += len(answer_indices)

                if max_samples is not None and len(result) >= max_samples:
                    return result

                if any(found_answers) or skip_none:
                    result.append(
                        DUEDatasetRow(
                            dataset_name=dataset_name,
                            document_name=document_name,
                            language=doc["language"],
                            # document_text=contents['text'],
                            annotation_id=annotation_id,
                            annotation_key=annotation_key,
                            annotation_values=(
                                found_answers if not skip_none else annotation_values
                            ),
                            answer_indices=answer_indices,
                            bboxes=bboxes,
                            image=image,
                        )
                    )
                else:
                    skipped_samples_count += 1

    if verbose:
        tqdm.write(
            f"Loaded DUE dataset {dataset_name}, {split}, use_chatgpt_labels={use_chatgpt_labels} -> {docs_count} Documents, {samples_count-skipped_samples_count} Samples ({samples_count} total, skipped {skipped_samples_count}), Answer Indices Extraction succeeded with modes: {answer_indices_extraction_mode}"
        )

    return result


def __get_max_document_size(
    dataset_name: Literal["DocVQA", "InfographicsVQA", "TabFact", "WikiTableQuestions"],
    split: Literal["train", "dev", "test"],
    ocr_engines: list[Literal["djvu", "microsoft_cv", "tesseract"]],
) -> list[DUEDatasetRow]:
    base_path = ENV.DATASETS_DIR / dataset_name
    split_path: pathlib.Path = base_path / "data" / split
    document_path: pathlib.Path = split_path / "document.jsonl"
    documents_content_path: pathlib.Path = split_path / "documents_content.jsonl"

    # First read all document contents into a dict
    document_contents: dict = dict()
    with jsonlines.open(documents_content_path) as reader:
        for doc in tqdm(reader, leave=False, desc=f"Reading {dataset_name} {split}"):
            document_contents[doc["name"]] = doc["contents"]

    # Now parse the document.jsonl file
    docs_count = 0
    widths = []
    heigths = []
    with jsonlines.open(document_path) as reader:
        for doc in tqdm(
            reader,
            leave=False,
            desc=f"Parsing {dataset_name} {split}",
            total=len(document_contents),
        ):
            docs_count += 1
            document_name = doc["name"]

            # Parse contents
            tool_name_to_contents = {
                c["tool_name"]: c for c in document_contents[document_name]
            }
            selected_ocr_engine = None
            for ocr_engine in ocr_engines:
                if ocr_engine in tool_name_to_contents:
                    selected_ocr_engine = ocr_engine
                    break

            if selected_ocr_engine is None:
                raise Exception(
                    f'No OCR engine from selection "{ocr_engines}" available for dataset {dataset_name}'
                )

            contents = tool_name_to_contents[selected_ocr_engine]
            del tool_name_to_contents  # free up memory
            ocr_results_key = (
                "common_format" if "common_format" in contents else "tokens_layer"
            )
            if ocr_results_key not in contents:
                print(f"{ocr_results_key} NOT FOUND, keys: {list(contents.keys())}")
            ocr_results = contents[ocr_results_key]

            _, _, orig_width, orig_height = tuple(
                ocr_results["structures"]["pages"]["positions"][0]
            )

            widths.append(orig_width)
            heigths.append(orig_height)

    print(
        f"{dataset_name}-{split} Max width: {max(widths)}, Max height: {max(heigths)}"
    )


if __name__ == "__main__":
    read_due_dataset(dataset_name="DocVQA", split="train", ocr_engines=["microsoft_cv"])
    for ds in ["DocVQA", "InfographicsVQA", "WikiTableQuestions"]:
        for split in ["train", "test", "dev"]:
            __get_max_document_size(
                dataset_name=ds, split=split, ocr_engines=["microsoft_cv"]
            )
