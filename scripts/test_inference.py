import tensorflow as tf
import cv2
import numpy as np
import pytesseract
import os
from object_detection.utils import label_map_util

def load_label_map(label_map_path):
    label_map = label_map_util.create_categories_from_labelmap(label_map_path, use_display_name=True)
    return {item['id']: item['name'] for item in label_map}

def preprocess_image(image):
    """
    Preprocess image by resizing to 512x512 with padding while keeping aspect ratio.
    Returns the resized image, scale factors, and padding offsets.
    """
    original_height, original_width = image.shape[:2]
    target_size = 512
    scale = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height))
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

    return padded_image, (original_width, original_height), (x_offset, y_offset)

def extract_text_from_box(image, xmin, ymin, xmax, ymax):
    """
    Extract text using pytesseract from a region within the image.
    """
    roi = image[ymin:ymax, xmin:xmax]
    text = pytesseract.image_to_string(roi, config='--psm 6')  # Use psm 6 for sparse text layout
    return text.strip()

def draw_highest_score_boxes_with_text(image, boxes, scores, classes, label_map, threshold):
    """
    Draw highest-scoring bounding boxes and extract text for each class.
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
        xmin = int(xmin * image_width)
        xmax = int(xmax * image_width)
        ymin = int(ymin * image_height)
        ymax = int(ymax * image_height)

        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Extract text from the bounding box
        text = extract_text_from_box(image, xmin, ymin, xmax, ymax)
        print ("512: ", text)

        # Display label and text
        field_name = label_map.get(class_id, f"Unknown Field {class_id}")
        label = f"{field_name}: {text}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

def draw_highest_score_boxes_original(image, boxes, scores, classes, label_map, threshold, original_size, padding_offsets):
    """
    Draw highest-scoring bounding boxes mapped to the original image and extract text.
    """
    padded_height, padded_width, _ = 512, 512, 3  # Padded size is always 512x512
    original_width, original_height = original_size
    x_offset, y_offset = padding_offsets

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
        xmin_padded = int(xmin * padded_width)
        xmax_padded = int(xmax * padded_width)
        ymin_padded = int(ymin * padded_height)
        ymax_padded = int(ymax * padded_height)

        xmin_no_padding = xmin_padded - x_offset
        xmax_no_padding = xmax_padded - x_offset
        ymin_no_padding = ymin_padded - y_offset
        ymax_no_padding = ymax_padded - y_offset

        xmin_original = int(xmin_no_padding * (original_width / (padded_width - 2 * x_offset)))
        xmax_original = int(xmax_no_padding * (original_width / (padded_width - 2 * x_offset)))
        ymin_original = int(ymin_no_padding * (original_height / (padded_height - 2 * y_offset)))
        ymax_original = int(ymax_no_padding * (original_height / (padded_height - 2 * y_offset)))

        xmin_original = max(0, xmin_original)
        xmax_original = min(original_width, xmax_original)
        ymin_original = max(0, ymin_original)
        ymax_original = min(original_height, ymax_original)

        # Draw bounding box
        cv2.rectangle(image, (xmin_original, ymin_original), (xmax_original, ymax_original), (255, 0, 0), 2)

        # Extract text from the bounding box
        text = extract_text_from_box(image, xmin_original, ymin_original, xmax_original, ymax_original)
        print ("origin: ", text)

        # Display label and text
        field_name = label_map.get(class_id, f"Unknown Field {class_id}")
        label = f"{field_name}: {text}"
        cv2.putText(image, label, (xmin_original, ymin_original - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def simple_test(image_path, model_path, label_map, threshold):
    """
    Load model, preprocess image, and perform detection.
    """
    image = cv2.imread(image_path)
    padded_image, original_size, padding_offsets = preprocess_image(image)
    model = tf.saved_model.load(model_path)
    model_fn = model.signatures['serving_default']

    input_tensor = tf.convert_to_tensor(np.expand_dims(padded_image, axis=0), dtype=tf.uint8)
    detections = model_fn(input_tensor)

    detection_boxes = detections['detection_boxes'].numpy()
    detection_scores = detections['detection_scores'].numpy()
    detection_classes = detections['detection_classes'].numpy()

    # Draw boxes on 512x512 resized image
    resized_image_with_boxes = draw_highest_score_boxes_with_text(
        padded_image.copy(), detection_boxes, detection_scores, detection_classes, label_map, threshold
    )
    cv2.imshow("Resized Image with Boxes", resized_image_with_boxes)

    # Draw boxes on original image
    original_image_with_boxes = draw_highest_score_boxes_original(
        image.copy(), detection_boxes, detection_scores, detection_classes, label_map, threshold, original_size, padding_offsets
    )
    cv2.imshow("Original Image with Boxes", original_image_with_boxes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    IMAGE_PATH = f"{os.getcwd()}/testing/img.jpg"
    MODEL_PATH = f"{os.getcwd()}/exported_model/saved_model"
    LABEL_MAP_PATH = f"{os.getcwd()}/dataset/label_map.pbtxt"

    label_map = load_label_map(LABEL_MAP_PATH)
    simple_test(IMAGE_PATH, MODEL_PATH, label_map, threshold=0.12)
