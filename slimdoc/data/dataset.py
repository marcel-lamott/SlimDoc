import json
import pathlib
from typing import Literal, Optional

from torch.utils.data import Dataset

from slimdoc.data.models import DUEDatasetRow, DatasetRowBase, OCRBox
from slimdoc.data.reader.funsd import read_funsd
from slimdoc.data.reader.sroie import read_sroie
from slimdoc.data.reader.due import read_due_dataset

# from slimdoc.data.reader.websrc import read_websrc

"""
- Dokumentbilder einbinden
- Token Indices der Labels rausfinden
- ChatGPT labels einbinden --> aber wie in Token Indices verwandeln?

DocVQA InfoVQA WTQ SROIE FUNSD
"""


def _unsupported_dataset(ds_name: str):
    return ValueError(f"Unsupported dataset '{ds_name}'")


def _normalize_bboxes(
    bboxes: list[list[int]],
    page_width: int,
    page_height: int,
    new_min: int = 0,
    new_max: int = 1000,
):
    def normalize(value, old_min, old_max):
        return int(
            ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
        )

    min_x0 = min([box[0] for box in bboxes])
    min_y0 = min([box[1] for box in bboxes])
    # max_x2 = max([box[2] for box in bboxes])
    # max_y2 = max([box[3] for box in bboxes])
    max_x2 = page_width
    max_y2 = page_height

    for box in bboxes:
        x0, y0, x2, y2 = tuple(box)
        yield [
            normalize(value=x0, old_min=min_x0, old_max=max_x2),
            normalize(value=y0, old_min=min_y0, old_max=max_y2),
            normalize(value=x2, old_min=min_x0, old_max=max_x2),
            normalize(value=y2, old_min=min_y0, old_max=max_y2),
        ]


class SERDataset(Dataset):
    data: list[DatasetRowBase]
    label_map: dict[str, int]
    dataset_name: str
    split: str

    def __init__(
        self,
        dataset_name: Literal["SROIE", "FUNSD", "CORD"],
        split: Literal["train", "test"],
        use_chatgpt_labels: bool = False,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.use_chatgpt_labels = use_chatgpt_labels

        if self.dataset_name == "SROIE":
            self.data = read_sroie(
                self.split, use_chatgpt_labels=self.use_chatgpt_labels
            )
            self.label_map = {
                None: 0,
                "company": 1,
                "address": 2,
                "date": 3,
                "total": 4,
            }
        elif self.dataset_name == "FUNSD":
            self.data = read_funsd(
                split=self.split, use_chatgpt_labels=self.use_chatgpt_labels
            )
            self.label_map = {"answer": 0, "header": 1, "other": 2, "question": 3}
        else:
            raise _unsupported_dataset(self.dataset_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row: DatasetRowBase = self.data[idx]

        # It seems LayoutLMv3 Processor does NOT normalize the bounding boxes
        boxes = [[b.x0, b.y0, b.x2, b.y2] for b in row.bounding_boxes]
        normalized_bboxes = list(
            _normalize_bboxes(boxes, page_width=row.width, page_height=row.height)
        )

        word_labels = None
        if self.label_map is not None:
            word_labels = [self.label_map[label] for label in row.GT]
        else:
            word_labels = row.GT

        b: OCRBox
        return {
            "sample_id": row.id,
            "dataset_name": self.dataset_name,
            "words": [b.text for b in row.bounding_boxes],
            "boxes": normalized_bboxes,  # [[b.x0, b.y0, b.x2, b.y2] for b in row.bounding_boxes],
            "word_labels": word_labels,
            "image": row.image,
        }


# For extractive QA
class VQADataset(Dataset):
    LABEL_NONE = 0
    LABEL_START = 1
    LABEL_END = 2

    data: list[DatasetRowBase]
    dataset_name: str
    split: str

    def __init__(
        self,
        dataset_name: Literal["DocVQA", "InfographicsVQA", "WikiTableQuestions"],
        split: Literal["train", "dev", "test"],
        use_chatgpt_labels: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.use_chatgpt_labels = use_chatgpt_labels

        if self.dataset_name in ("DocVQA", "InfographicsVQA", "WikiTableQuestions"):
            self.data = read_due_dataset(
                dataset_name=dataset_name,
                split=split,
                ocr_engines=["microsoft_cv"],
                use_chatgpt_labels=use_chatgpt_labels,
                max_samples=max_samples,
            )
        else:
            raise _unsupported_dataset(self.dataset_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.dataset_name in ("DocVQA", "InfographicsVQA", "WikiTableQuestions"):
            row: DUEDatasetRow = self.data[idx]

            # It seems LayoutLMv3 Processor does NOT normalize the bounding boxes
            boxes = [[b.x0, b.y0, b.x2, b.y2] for b in row.bounding_boxes]
            normalized_bboxes = list(
                _normalize_bboxes(boxes, page_width=row.width, page_height=row.height)
            )

            b: OCRBox
            return {
                "sample_id": row.annotation_id,
                "dataset_name": self.dataset_name,
                "words": [b.text for b in row.bounding_boxes],
                "question": row.annotation_key,
                "answers": row.annotation_values,
                "answer_span_indices": row.answer_indices,
                "boxes": normalized_bboxes,  # [[b.x0, b.y0, b.x2, b.y2] for b in row.bounding_boxes],
                "image": row.image,
            }
        else:
            raise _unsupported_dataset(self.dataset_name)


if __name__ == "__main__":
    ds = SERDataset("SROIE", "train")
    print(ds)
    print(ds[0])
