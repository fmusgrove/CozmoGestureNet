import sys
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf

from HandDetector.utils import label_map_util

detection_graph = tf.Graph()
sys.path.append("..")

# Path to frozen detection graph used in object detection
HAND_DETECT_CKPT = 'Models/hand_detect_graph/ssd5_optimized_inference_graph.pb'
# List of the strings that is used to add correct label for each box
DETECT_LABELS = 'Models/hand_detect_graph/hand_label_map.pbtxt'

NUM_CLASSES = 1
# Load label map
label_map = label_map_util.load_labelmap(DETECT_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_detect_graph():
    # Load frozen TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(HAND_DETECT_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print('>  ====== Hand detection graph loaded.')
    return detection_graph, sess


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(image_np, bounds):
    for i in range(len(bounds)):
        p1, p2 = bounds[i][:2]
        cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)


def get_segmentation_bounds(num_hands_detect, score_thresh, scores, boxes, im_width, im_height):
    """
    Return the segmentation bounding box coordinates
    :param num_hands_detect: number of hands to detect
    :param score_thresh:
    :param scores:
    :param boxes:
    :param im_width: width of the image
    :param im_height: height of the image
    :return: list containing coordinate pair tuples
    """
    bounds_scores = []
    for i in range(num_hands_detect):
        if scores[i] > score_thresh:
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            bounds_scores.append((p1, p2, scores[i]))
    return bounds_scores


# Show fps value on image.
def draw_fps_on_image(fps_text, image_np):
    cv2.putText(image_np, fps_text, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


def draw_score_on_image(score_text, image_np):
    cv2.putText(image_np, score_text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input. For testing on a webcam instead of Cozmo
# Source : Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
