from PIL import Image, ImageDraw, ImageFont

from slimdoc.data.models import OCRBox


def draw_bounding_boxes(image, bboxes, color="red", width=3, show_text=True):
    """
    Draws bounding boxes on a given Pillow image.

    :param image: Pillow Image object
    :param bboxes: List of bounding boxes [(x0, y0, x1, y1), ...]
    :param color: Color of the bounding box (default: red)
    :param width: Line width (default: 3)
    :return: Image with bounding boxes
    """
    draw = ImageDraw.Draw(image)

    bbox: OCRBox
    print(f"Drawing {len(bboxes)} bounding boxes")
    for bbox in bboxes:
        box = (bbox.x0, bbox.y0, bbox.x2, bbox.y2)
        draw.rectangle(box, outline=color, width=width)

        # font = ImageFont.truetype("sans-serif.ttf", 16)
        if show_text:
            font = ImageFont.load_default(32)
            draw.text(box, bbox.text, (255, 0, 255), font=font)

    return image
