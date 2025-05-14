import pathlib
from pdf2image import convert_from_path
from PIL import Image

from slimdoc import ENV


def pdf_to_image(pdf_path: pathlib.Path, target_size: tuple[int, int] = None):
    images: list = convert_from_path(pdf_path, size=target_size)
    assert len(images) == 1, "Multi-page document are not supported "
    image = images[0]
    return image


if __name__ == "__main__":
    file_path = ENV.DATASETS_DIR / "DocVQA" / "pdfs" / f"ffbf0023_4.pdf"
    image = pdf_to_image(file_path)
    image.show()
