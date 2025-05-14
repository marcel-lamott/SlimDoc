from dataclasses import dataclass
from pathlib import Path
import json
from typing import Union, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from slimdoc import ENV
from slimdoc.data.chatgptlabels import get_label_value, read_chatgpt_labels
from slimdoc.data.models import DatasetRowBase, OCRBox, SROIERow


def _read_bbox_and_words(fpath: Path):
    bbox_and_words_list = []

    with open(fpath, "r", errors="ignore") as f:
        for line in f.read().splitlines():
            if len(line) == 0:
                continue

            split_lines = line.split(",")

            bbox = np.array(split_lines[0:8], dtype=np.int32)
            text = ",".join(split_lines[8:])

            # From the splited line we save (filename, [bounding box points], text line).
            # The filename will be useful in the future
            # bbox_and_words_list.append([fpath.stem, *bbox, text])
            bbox_and_words_list.append([*bbox, text])

        cols = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "line"]
        dataframe = pd.DataFrame(bbox_and_words_list, columns=cols)  # , dtype=np.int16
        dataframe = dataframe.drop(columns=["x1", "y1", "x3", "y3"])
        values = dataframe.values.tolist()
        ocr_lines = [OCRBox(*v) for v in values]

    return ocr_lines


def _read_entities(fpath: Path):
    with open(fpath, "r", errors="ignore") as f:
        return json.loads(f.read())


# Function to assign labels considering fragments and mixed text
def assign_labels(ocr_boxes, entities):
    label_ids = []
    box: OCRBox
    for box in ocr_boxes:
        text = box.text
        assigned_label = None  # Default label (Outside)

        # Check each entity to see if the bounding box contains any relevant entity part
        for entity, entity_text in entities.items():
            # Check if the entire entity text is in the bounding box
            if entity_text in text:
                assigned_label = entity
                break

            if text in entity_text:
                assigned_label = entity
                break

            if assigned_label != None:
                break  # No need to check further if label is assigned

        # Assign label to the bounding box
        label_ids.append(assigned_label)
    return label_ids


def read_sroie(
    split: Literal["train", "test"], use_chatgpt_labels: bool = False
) -> list[SROIERow]:
    path = ENV.DATASETS_DIR / "SROIE" / split
    p = (path / "box").glob("**/*")
    files = [x for x in p if x.is_file()]
    samples = list()

    # Read ChatGPT generated labels if applicable
    docid_to_chatgpt_labels = None
    if use_chatgpt_labels:
        docid_to_chatgpt_labels = read_chatgpt_labels("SROIE", split)

    skipped_samples = 0

    for f in tqdm(files):
        lines = _read_bbox_and_words(f)

        _id = f.stem

        # Read labels
        if use_chatgpt_labels:
            ents = dict()
            for ent in ["company", "date", "address", "total"]:
                value = get_label_value(docid_to_chatgpt_labels, doc_id=_id, key=ent)
                if value is not None:
                    ents[ent] = value
        else:
            ent_path = path / "entities" / f"{_id}.txt"
            ents = _read_entities(ent_path)

        img_path = path / "img" / f"{_id}.jpg"
        # assert img_path.exists()
        image = Image.open(img_path).convert("RGB")
        # width, height = image.size

        # Check all entities are given
        if any([1 for k, v in ents.items() if len(v) == 0]) or len(ents) < 4:
            skipped_samples += 1
            continue

        labels = assign_labels(lines, ents)
        if len(set(labels)) != 5:
            skipped_samples += 1
            continue

        samples.append(
            SROIERow(_id=_id, bboxes=lines, entities=ents, labels=labels, image=image)
        )

    print(
        f"Read SROIE {split}: {len(samples)=}, use_chatgpt_labels={use_chatgpt_labels}, skipped {skipped_samples} samples"
    )
    return samples


if __name__ == "__main__":
    data = read_sroie(split="train", use_chatgpt_labels=True)
    data = read_sroie(split="train", use_chatgpt_labels=False)
    data = read_sroie(split="test", use_chatgpt_labels=True)
    data = read_sroie(split="test", use_chatgpt_labels=False)
