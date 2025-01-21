import tensorflow as tf
import cv2
import numpy as np
import pytesseract
import os
from object_detection.utils import label_map_util

# Ensure the Tesseract executable path is correctly set
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_label_map(label_map_path):
    """
    Loads the label map from a label_map.pbtxt file.

    Parameters:
    - label_map_path: Path to the label map file (label_map.pbtxt).

    Returns:
    - A dictionary with class_id as keys and field names as values.
    """
    label_map = label_map_util.create_categories_from_labelmap(label_map_path, use_display_name=True)
    return {item['id']: item['name'] for item in label_map}


def draw_bounding_boxes(image, boxes, scores, classes, label_map, threshold):
    """
    Draws bounding boxes on the image based on detected objects and prints field names and text.

    Parameters:
    - image: The input image.
    - boxes: Detected bounding boxes (normalized coordinates).
    - scores: Confidence scores for each box.
    - classes: Detected class labels for each box.
    - label_map: A dictionary with class_id as keys and field names as values.
    - threshold: Minimum confidence threshold for displaying a box.
    """
    image_height, image_width, _ = image.shape

    highest_score_boxes = {}

    for i in range(len(scores[0])):
        if scores[0][i] >= threshold:
            box = boxes[0][i]
            class_id = int(classes[0][i])
            score = scores[0][i]

            if class_id not in highest_score_boxes or highest_score_boxes[class_id]['score'] < score:
                highest_score_boxes[class_id] = {
                    'box': box,
                    'score': score
                }

    for class_id, data in highest_score_boxes.items():
        box = data['box']
        ymin, xmin, ymax, xmax = box
        score = data['score']

        xmin = int(xmin * image_width)
        xmax = int(xmax * image_width)
        ymin = int(ymin * image_height)
        ymax = int(ymax * image_height)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        roi = image[ymin:ymax, xmin:xmax]
        text = pytesseract.image_to_string(roi, config='--psm 6')

        field_name = label_map.get(class_id, f"Unknown Field {class_id}")

        print(f"{field_name}: {text.strip()}")

        label = f"{field_name}: {text.strip()}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


def simple_test(image_path, model_path, label_map, threshold):
    """
    Loads the model and performs object detection on the input image.

    Parameters:
    - image_path: Path to the input image.
    - model_path: Path to the exported SavedModel.
    - label_map: A dictionary with class_id as keys and field names as values.
    - threshold: Minimum confidence threshold for displaying detections.
    """
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (512, 512))
    image_resized = np.expand_dims(image_resized, axis=0)
    model = tf.saved_model.load(model_path)
    model_fn = model.signatures['serving_default']

    input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)
    detections = model_fn(input_tensor)

    detection_boxes = detections['detection_boxes'].numpy()
    detection_scores = detections['detection_scores'].numpy()
    detection_classes = detections['detection_classes'].numpy()

    image_with_boxes = draw_bounding_boxes(image, detection_boxes, detection_scores, detection_classes, label_map, threshold)

    cv2.imshow("Detections", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    IMAGE_PATH = f"{os.getcwd()}/testing/img.jpg"
    MODEL_PATH = f"{os.getcwd()}/exported_model/saved_model"
    LABEL_MAP_PATH = f"{os.getcwd()}/dataset/label_map.pbtxt"  

    label_map = load_label_map(LABEL_MAP_PATH)

    simple_test(IMAGE_PATH, MODEL_PATH, label_map, threshold=0.1)
