import os
import tensorflow as tf
from object_detection.utils import dataset_util
from lxml import etree
import glob
import pathlib

# Function to convert XML annotation into a dict
def create_example(annotation_file, image_dir):
    tree = etree.parse(annotation_file)
    root = tree.getroot()

    # Get image filename
    filename = root.find("filename").text
    image_path = os.path.join(image_dir, filename)
    
    # Read image
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    # Get image dimensions
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    # Get the annotations (bounding boxes and labels)
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes = []
    classes_text = []
    
    for obj in root.findall("object"):
        xmin = float(obj.find("bndbox/xmin").text) / width
        ymin = float(obj.find("bndbox/ymin").text) / height
        xmax = float(obj.find("bndbox/xmax").text) / width
        ymax = float(obj.find("bndbox/ymax").text) / height
        
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        class_name = obj.find("name").text
        classes.append(1)  # Assuming all objects are of class 1 (customize as needed)
        classes_text.append(class_name.encode('utf8'))

    # Create the TFRecord example
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


# Function to write the TFRecord file
def write_tfrecord(xml_dir, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    
    # Loop through all XML files in the annotation directory
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    
    for xml_file in xml_files:
        tf_example = create_example(xml_file, image_dir)
        writer.write(tf_example.SerializeToString())
    
    writer.close()


if __name__ == "__main__":
    xml_dir = '/path/to/dataset/labels'  # Path to your XML annotations
    image_dir = '/path/to/dataset/images'  # Path to your images
    output_path = '/path/to/dataset/train.tfrecord'  # Path to save the TFRecord
    
    write_tfrecord(xml_dir, image_dir, output_path)
    print("TFRecord file created:", output_path)
