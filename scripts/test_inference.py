import tensorflow as tf
import cv2
import numpy as np
import pytesseract
import os
from object_detection.utils import label_map_util

def load_label_map(label_map_path):
    """
    Load the label map file and return a dictionary mapping class_id to class names.
    """
    label_map = label_map_util.create_categories_from_labelmap(label_map_path, use_display_name=True)
    return {item['id']: item['name'] for item in label_map}

def resize_and_pad_image(image, target_size=(512, 512)):
    """
    Resize and pad the image to the target size while maintaining aspect ratio.
    """
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate new dimensions preserving aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:  # Landscape orientation
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Portrait orientation
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image to new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a blank padded image
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate padding offsets
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    # Place the resized image in the center of the padded image
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    return padded_image, (x_offset, y_offset), (original_width, original_height)

def draw_bounding_boxes(image, boxes, scores, classes, label_map, threshold, original_size, padding_offsets):
    """
    Draw bounding boxes on the image, extract text using Tesseract, and display class names.
    """
    padded_height, padded_width, _ = image.shape
    original_width, original_height = original_size
    x_offset, y_offset = padding_offsets

    highest_score_boxes = {}

    # Loop through detections and store only the highest-scoring box for each class
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

    # Draw only the highest-scoring boxes
    for class_id, data in highest_score_boxes.items():
        box = data['box']
        ymin, xmin, ymax, xmax = box

        # Adjust box coordinates from normalized to padded image size
        xmin_padded = int(xmin * padded_width)
        xmax_padded = int(xmax * padded_width)
        ymin_padded = int(ymin * padded_height)
        ymax_padded = int(ymax * padded_height)

        # Map padded image coordinates back to original image size
        xmin_original = int((xmin_padded - x_offset) * (original_width / (padded_width - 2 * x_offset)))
        xmax_original = int((xmax_padded - x_offset) * (original_width / (padded_width - 2 * x_offset)))
        ymin_original = int((ymin_padded - y_offset) * (original_height / (padded_height - 2 * y_offset)))
        ymax_original = int((ymax_padded - y_offset) * (original_height / (padded_height - 2 * y_offset)))

        # Ensure the bounding box fits within the original image boundaries
        xmin_original = max(0, xmin_original)
        xmax_original = min(original_width, xmax_original)
        ymin_original = max(0, ymin_original)
        ymax_original = min(original_height, ymax_original)

        # Draw the bounding box
        cv2.rectangle(image, (xmin_original, ymin_original), (xmax_original, ymax_original), (0, 0, 255), 2)

        # Extract ROI and preprocess for better OCR
        roi = image[ymin_original:ymax_original, xmin_original:xmax_original]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_thresh = cv2.threshold(roi_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(roi_thresh, config='--psm 6').strip()

        field_name = label_map.get(class_id, f"Unknown Field {class_id}")

        print(f"{field_name}: {text}")

        label = f"{field_name}: {text}"
        cv2.putText(image, label, (xmin_original, ymin_original - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

def simple_test(image_path, model_path, label_map, threshold):
    """
    Perform object detection on the input image and display results with bounding boxes and extracted text.
    """
    image = cv2.imread(image_path)
    padded_image, padding_offsets, original_size = resize_and_pad_image(image)

    # Expand dimensions for the model input
    image_resized = np.expand_dims(padded_image, axis=0)
    model = tf.saved_model.load(model_path)
    model_fn = model.signatures['serving_default']

    input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.uint8)
    detections = model_fn(input_tensor)

    detection_boxes = detections['detection_boxes'].numpy()
    detection_scores = detections['detection_scores'].numpy()
    detection_classes = detections['detection_classes'].numpy()

    # Draw bounding boxes on the original image
    image_with_boxes = draw_bounding_boxes(image, detection_boxes, detection_scores, detection_classes, label_map, threshold, original_size, padding_offsets)

    cv2.imshow("Detections", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    IMAGE_PATH = f"{os.getcwd()}/testing/img.jpg"
    MODEL_PATH = f"{os.getcwd()}/exported_model/saved_model"
    LABEL_MAP_PATH = f"{os.getcwd()}/dataset/label_map.pbtxt"

    label_map = load_label_map(LABEL_MAP_PATH)

    simple_test(IMAGE_PATH, MODEL_PATH, label_map, threshold=0.12)
