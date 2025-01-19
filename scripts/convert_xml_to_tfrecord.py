import os
import glob
import tensorflow as tf
from object_detection.utils import label_map_util
import xml.etree.ElementTree as ET

# Function to create a TensorFlow Example from an XML file
def create_example(xml_file, image_dir, label_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image file name
    filename = root.find("filename").text
    image_path = os.path.join(image_dir, filename)

    # Read image data
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    # Get image format (assumes JPG, adjust if needed)
    image_format = b'jpg'

    # Get bounding box and labels
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_names = []
    classes = []

    for obj in root.findall('object'):
        # Get label name and map it to the label map ID
        class_name = obj.find('name').text
        class_id = label_map.get(class_name, None)
        if class_id is None:
            continue  # Skip if the class is not in the label map

        # Get bounding box coordinates
        xmin = float(obj.find('bndbox/xmin').text) / 1024  # normalize based on image width (adjust as needed)
        ymin = float(obj.find('bndbox/ymin').text) / 1024  # normalize based on image height (adjust as needed)
        xmax = float(obj.find('bndbox/xmax').text) / 1024
        ymax = float(obj.find('bndbox/ymax').text) / 1024

        # Append data
        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        classes.append(class_id)
        class_names.append(class_name)

    # Create the TensorFlow Example
    feature_dict = {
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode('utf8') for name in class_names])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    
    return example

# Function to write TFRecord file
def write_tfrecord(xml_dir, image_dir, output_path, label_map):
    writer = tf.io.TFRecordWriter(output_path)
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    for xml_file in xml_files:
        tf_example = create_example(xml_file, image_dir, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    # Set paths for training and evaluation data
    train_image_dir = f'{os.getcwd()}/dataset/train/images'  # Path to your training image folder
    train_label_dir = f'{os.getcwd()}/dataset/train/labels'  # Path to your training label folder
    eval_image_dir = f'{os.getcwd()}/dataset/eval/images'    # Path to your evaluation image folder
    eval_label_dir = f'{os.getcwd()}/dataset/eval/labels'    # Path to your evaluation label folder

    # Load label map
    label_map_path = f'{os.getcwd()}/dataset/label_map.pbtxt'  # Path to your label map
    label_map = label_map_util.get_label_map_dict(label_map_path)

    # Set output paths for TFRecord files
    train_output_path = f'{os.getcwd()}/dataset/train/train.tfrecord'
    eval_output_path = f'{os.getcwd()}/dataset/eval/eval.tfrecord'

    # Write TFRecord files for training and evaluation
    write_tfrecord(train_label_dir, train_image_dir, train_output_path, label_map)
    write_tfrecord(eval_label_dir, eval_image_dir, eval_output_path, label_map)

    print("TFRecord files created successfully!")
