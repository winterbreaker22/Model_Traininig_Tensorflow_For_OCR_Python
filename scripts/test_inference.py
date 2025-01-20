import tensorflow as tf
import numpy as np
import cv2
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

model = tf.saved_model.load(f'{os.getcwd()}/exported_model/saved_model')

label_map_path = f'{os.getcwd()}/dataset/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

image_path = f'{os.getcwd()}/test/test.jpg'
image_np = cv2.imread(image_path)
image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

input_tensor = tf.convert_to_tensor(image_np_rgb)
input_tensor = input_tensor[tf.newaxis,...]

detections = model(input_tensor)
print("detections: ", detections)

vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    detections['detection_boxes'][0].numpy(),
    detections['detection_classes'][0].numpy().astype(np.int32),
    detections['detection_scores'][0].numpy(),
    category_index,
    instance_masks=detections.get('detection_masks', None),
    use_normalized_coordinates=True,
    line_thickness=8)

cv2.imshow('Inference Result', image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
