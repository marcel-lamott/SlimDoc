import json
from typing import Literal
from PIL import Image
from tqdm import tqdm

from slimdoc import ENV
from slimdoc.data.models import FUNSDRow, OCRBox


def read_funsd(
    split: Literal["train", "test"], use_chatgpt_labels: bool = False
) -> list[FUNSDRow]:

    assert (
        use_chatgpt_labels is False
    ), f"ChatGPT labels are not supported yet for FUNSD"

    path = ENV.DATASETS_DIR / "funsd" / f"{split}ing_data" / "annotations"
    images_path = ENV.DATASETS_DIR / "funsd" / f"{split}ing_data" / "images"

    result = []
    all_f1 = []
    for file in tqdm(path.iterdir(), desc="Read FUNSD2", leave=False):
        text = file.read_text(encoding="utf-8")
        data = json.loads(text)
        content = data["form"]

        sample_id = file.stem
        boxes = []
        labels = []
        for x in content:
            txt = x["text"]
            box = x["box"]
            label = x["label"]
            if len(txt) > 0:
                x0, y0, x2, y2 = tuple(box)
                b = OCRBox(x0, y0, x2, y2, txt)
                boxes.append(b)
                labels.append(label)

        if any(boxes):
            img_path = images_path / f"{sample_id}.png"
            img = Image.open(img_path).convert("RGB")
            row = FUNSDRow(_id=sample_id, bboxes=boxes, labels=labels, image=img)
            result.append(row)

    print(f"Read FUNSD {split}, {len(result)} entries found.")
    return result


if __name__ == "__main__":
    ds = read_funsd(split="train")
