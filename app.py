from flask import Flask
from yolo import YOLO
from PIL import Image

app = Flask(__name__)

@app.route('/')
def hello_world():
    image = Image.open("./test_data/farmstead.7.jpeg")
    boxes = detect_img(image)
    print(boxes)
    return('Got boxes!')

def detect_img(image):
    return yolo.detect_image(image)


if __name__ == '__main__':
    app.run()
    yolo = YOLO()