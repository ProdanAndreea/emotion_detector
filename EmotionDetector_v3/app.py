"""
A sample Hello World server.
"""
import os
import requests

from flask import Flask, render_template, request, jsonify, abort

import tensorflow as tf
from tf2_yolov4.anchors import YOLOV4_ANCHORS
from tf2_yolov4.model import YOLOv4

HEIGHT, WIDTH = (416, 416)

image_name = ''

# customize your API through the following parameters
weights_path = './weights/yolov4_3.h5'
tiny = False  # set to True if using a Yolov3 Tiny model
size = 416  # size images are resized to for model
num_classes = 3  # number of classes in model

# load in weights and classes
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = YOLOv4(
    input_shape=(HEIGHT, WIDTH, 3),
    anchors=YOLOV4_ANCHORS,
    num_classes=3,
    training=False,
    yolo_max_boxes=100,
    yolo_iou_threshold=0.5,
    yolo_score_threshold=0.5,
    weights=weights_path
)
# model.load_weights(weights_path)

# pylint: disable=C0103
app = Flask(__name__)


# API that returns JSON with classes found in images
@app.route('/detect', methods=['POST'])
def detect_t():
    print('starting image_t')

    image = request.files["images"]

    global image_name
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))

    image_file = tf.io.read_file(image_name)

    print('image 2')
    img = tf.image.decode_image(image_file)
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    images = tf.expand_dims(img, axis=0) / 255.0

    print('image 5')
    boxes, scores, classes, valid_detections = model.predict(images)

    # COCO classes
    CLASSES = ['happy', 'neutral', 'shocked']

    # colors for visualization
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125]]

    # plot results
    pil_img = images[0]
    boxes = boxes[0] * [WIDTH, HEIGHT, WIDTH, HEIGHT]
    scores = scores[0]
    classes = classes[0].astype(int)
    print('image 7')

    # create list of responses for image
    # this will allow in future to process multiple images
    responses = []
    # create list for final response
    response = []
    for (xmin, ymin, xmax, ymax), score, cl in zip(boxes.tolist(), scores.tolist(), classes.tolist()):
        if score > 0:
            responses.append({
                "class": CLASSES[cl],
                "confidence": float("{0:.2f}".format(score))
            })
    response.append({
        "image": image_name,
        "detections": responses
    })

    # remove temporary image
    os.remove(image_name)

    try:
        return jsonify({"response": response}), 200
    except FileNotFoundError:
        abort(404)


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    message = "It's running!"

    """Get Cloud Run environment variables."""
    # service = os.environ.get('K_SERVICE', 'Unknown service')
    # revision = os.environ.get('K_REVISION', 'Unknown revision')

    return render_template('index.html',
                           message=message)


if __name__ == '__main__':
    server_port = os.environ.get('PORT', '8080')
    app.run(debug=False, port=server_port, host='0.0.0.0')
