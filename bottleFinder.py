from yolo import YOLO
from PIL import Image

yolo = YOLO()


def get_bottles(path):
    image = Image.open(path)
    unprocessed_image = Image.open(path)
    boxes = yolo.detect_image(image)
    bottle_boxes = [box for box in boxes if box[0] == 'bottle']
    bottle_images = [unprocessed_image.crop((bottle[2][0], bottle[2][1], bottle[3][0], bottle[3][1])) for bottle in bottle_boxes]

    unprocessed_image.show()

    for img in bottle_images:
        img.show()


get_bottles("./polka.jpg")



