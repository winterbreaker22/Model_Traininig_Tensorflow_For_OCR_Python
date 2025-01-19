import os
import glob
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import defaultdict
import xml.etree.ElementTree as ET

def create_example(xml_file, image_dir, label_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = os.path.join(image_dir, filename)

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpg' 

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_names = []
    classes = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = label_map.get(class_name, None)
        if class_id is None:
            continue  

        xmin = float(obj.find('bndbox/xmin').text) / 1024 
        ymin = float(obj.find('bndbox/ymin').text) / 1024  
        xmax = float(obj.find('bndbox/xmax').text) / 1024
        ymax = float(obj.find('bndbox/ymax').text) / 1024

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        classes.append(class_id)
        class_names.append(class_name)

    tf_example = dataset_util.create_example(
        image_data=encoded_image_data,
        image_format=image_format,
        xmins=xmins,
        xmaxs=xmaxs,
        ymins=ymins,
        ymaxs=ymaxs,
        classes=classes,
        classes_text=class_names,
        filename=filename
    )

    return tf_example

def write_tfrecord(xml_dir, image_dir, output_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    for xml_file in xml_files:
        tf_example = create_example(xml_file, image_dir, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    image_dir = f'{os.getcwd()}/dataset/images' 
    label_map_path = f'{os.getcwd()}/dataset/label_map.pbtxt'  

    label_map = label_map_util.get_label_map_dict(label_map_path)

    train_output_path = f'{os.getcwd()}/dataset/train/train.tfrecord'
    eval_output_path = f'{os.getcwd()}/dataset/eval/eval.tfrecord'

    write_tfrecord(f'{os.getcwd()}/dataset/train/labels', image_dir, train_output_path, label_map)
    write_tfrecord(f'{os.getcwd()}/dataset/eval/labels', image_dir, eval_output_path, label_map)

    print("TFRecord files created successfully!")
