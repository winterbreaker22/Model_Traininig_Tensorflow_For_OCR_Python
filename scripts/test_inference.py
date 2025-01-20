import tensorflow as tf
import cv2
import os
import numpy as np


def draw_bounding_boxes(image, boxes, scores, classes, threshold=0.2):
    """
    Draws bounding boxes on the image based on detected objects.

    Parameters:
    - image: The input image.
    - boxes: Detected bounding boxes (normalized coordinates).
    - scores: Confidence scores for each box.
    - classes: Detected class labels for each box.
    - threshold: Minimum confidence threshold for displaying a box.
    """
    image_height, image_width, _ = image.shape

    for i in range(len(scores[0])):
        if scores[0][i] >= threshold:
            box = boxes[0][i]
            print("box: ", box)
            ymin, xmin, ymax, xmax = box

            xmin = int(xmin * image_width)
            xmax = int(xmax * image_width)
            ymin = int(ymin * image_height)
            ymax = int(ymax * image_height)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)  

            label = f"Class: {int(classes[0][i])}, Score: {scores[0][i]:.2f}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


def simple_test(image_path, model_path, threshold=0.2):
    """
    Loads the model and performs object detection on the input image.

    Parameters:
    - image_path: Path to the input image.
    - model_path: Path to the exported SavedModel.
    - threshold: Minimum confidence threshold for displaying detections.
    """
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (300, 300))
    image_resized = np.expand_dims(image_resized, axis=0) 
    model = tf.saved_model.load(model_path)
    model_fn = model.signatures['serving_default']

    input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)
    detections = model_fn(input_tensor)

    detection_boxes = detections['detection_boxes'].numpy()
    detection_scores = detections['detection_scores'].numpy()
    detection_classes = detections['detection_classes'].numpy()

    image_with_boxes = draw_bounding_boxes(image, detection_boxes, detection_scores, detection_classes, threshold)

    cv2.imshow("Detections", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    IMAGE_PATH = f"{os.getcwd()}/test/test.jpg" 
    MODEL_PATH = f"{os.getcwd()}/exported_model/saved_model" 

    simple_test(IMAGE_PATH, MODEL_PATH)
