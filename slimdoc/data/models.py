from __future__ import annotations
from dataclasses import dataclass
import pathlib
from typing import Optional, Union

from PIL.Image import Image


@dataclass(frozen=True)
class OCRBox:
    x0: int
    y0: int
    x2: int
    y2: int
    text: str

    @property
    def width(self):
        return self.x2 - self.x0

    @property
    def height(self):
        return self.y2 - self.y0

    @property
    def top(self):
        return self.y0

    @property
    def left(self):
        return self.x0

    @property
    def bottom(self):
        return self.y2

    @property
    def right(self):
        return self.x2

    @staticmethod
    def union(box_a: OCRBox, box_b: OCRBox, text: Optional[str] = None) -> OCRBox:
        left = min(box_a.left, box_b.left)
        top = min(box_a.top, box_b.top)
        right = max(box_a.right, box_b.right)
        bottom = max(box_a.bottom, box_b.bottom)
        text = text or f"{box_a.text} {box_b.text}"
        return OCRBox(left, top, right, bottom, text)

    @staticmethod
    def union_all(boxes: list[OCRBox], text: Optional[str] = None) -> Optional[OCRBox]:
        boxes = iter(boxes)
        total = next(boxes, None)
        if not total:
            return None
        for box in boxes:
            total = OCRBox.union(total, box)

        text = text or total.text
        return OCRBox(total.left, total.top, total.right, total.bottom, text)


class DatasetRowBase:
    image: Image

    @property
    def GT(self) -> dict[Union[str, int], list[str]]:
        pass

    @property
    def id(self) -> str:
        pass

    @property
    def bounding_boxes(self) -> list[OCRBox]:
        pass

    # @property
    # def image(self) -> Image:
    #     pass

    @property
    def width(self) -> float:
        width, height = self.image.size
        return width

    @property
    def height(self) -> float:
        width, height = self.image.size
        return height


@dataclass(kw_only=True)
class DUEDatasetRow(DatasetRowBase):
    """
    Represents a single sample in a DUE dataset, e.g. a single question
    """

    dataset_name: str
    document_name: str
    language: str
    # document_text: str
    annotation_id: str
    annotation_key: str  # e.g. the question
    annotation_values: list[str]  # all answer variants
    answer_indices: list[
        tuple[int, int]
    ]  # answer token indices for each annotation_value
    bboxes: list[OCRBox]
    image: Image

    @property
    def GT(self) -> dict[Union[str, int], list[str]]:
        return self.annotation_values

    @property
    def id(self) -> str:
        return self.annotation_id

    # @property
    # def page_index(self, bbox_i) -> int:
    #     #print(f'page_indices: {len(self.page_indices)} bbox_i: {bbox_i}')
    #     return self.page_indices[bbox_i]

    @property
    def bounding_boxes(self) -> list[OCRBox]:
        return self.bboxes


@dataclass(kw_only=True)
class SROIERow(DatasetRowBase):
    _id: str
    bboxes: list[OCRBox]
    entities: dict[str, str]
    labels: list[str]  # assigns each OCR box a label
    image: Image

    @property
    def GT(self) -> dict[Union[str, int], list[str]]:
        # return {
        #     'company': ([self.entities['company']], 'string'),
        #     'date': ([self.entities['date']], 'date'),
        #     'address': ([self.entities['address']], 'string'),
        #     'total': ([self.entities['total']], 'currency'),
        # }
        return self.labels

    @property
    def id(self) -> str:
        return self._id

    @property
    def bounding_boxes(self) -> list[OCRBox]:
        return self.bboxes


@dataclass(kw_only=True)
class FUNSDRow(DatasetRowBase):
    _id: str
    bboxes: list[OCRBox]
    labels: list[str]  # assigns each OCR box a label
    image: Image

    @property
    def GT(self) -> dict[Union[str, int], list[str]]:
        return self.labels

    @property
    def id(self) -> str:
        return self._id

    @property
    def bounding_boxes(self) -> list[OCRBox]:
        return self.bboxes

    # @property
    # def image(self) -> Image:
    #     pass
