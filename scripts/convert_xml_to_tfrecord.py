import os
import glob
import tensorflow as tf
from object_detection.utils import label_map_util
import xml.etree.ElementTree as ET

def create_example(xml_file, image_dir, label_map):
    """
    Creates a TensorFlow Example from a single XML annotation file and corresponding image,
    using normalized bounding box values.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image file name and path
    filename = root.find("filename").text
    image_path = os.path.join(image_dir, filename)

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = b'jpg'  # Adjust if your images are PNG or other formats

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    class_names = []
    classes = []

    for obj in root.findall('object'):
        # Get class name and ID
        class_name = obj.find('name').text
        class_id = label_map.get(class_name, None)
        if class_id is None:
            continue  # Skip unknown classes

        # Use normalized bounding box values
        xmin = float(obj.find('bndbox/normalized_xmin').text)
        ymin = float(obj.find('bndbox/normalized_ymin').text)
        xmax = float(obj.find('bndbox/normalized_xmax').text)
        ymax = float(obj.find('bndbox/normalized_ymax').text)

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        classes.append(class_id)
        class_names.append(class_name)

    # Prepare feature dictionary for TFRecord
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

def write_tfrecord(xml_dir, image_dir, output_path, label_map):
    """
    Writes multiple XML annotations and their corresponding images to a TFRecord file.
    """
    writer = tf.io.TFRecordWriter(output_path)
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    for xml_file in xml_files:
        tf_example = create_example(xml_file, image_dir, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == "__main__":
    # Paths to image and annotation directories
    train_image_dir = f'{os.getcwd()}/dataset/train/images' 
    train_label_dir = f'{os.getcwd()}/dataset/train/labels' 
    eval_image_dir = f'{os.getcwd()}/dataset/eval/images'    
    eval_label_dir = f'{os.getcwd()}/dataset/eval/labels'  
    # image_dir = f'{os.getcwd()}/dataset/images'    
    # label_dir = f'{os.getcwd()}/dataset/labels'  

    # Label map path
    label_map_path = f'{os.getcwd()}/dataset/label_map.pbtxt'  
    label_map = label_map_util.get_label_map_dict(label_map_path)

    # TFRecord output paths
    train_output_path = f'{os.getcwd()}/dataset/train/train.tfrecord'
    eval_output_path = f'{os.getcwd()}/dataset/eval/eval.tfrecord'
    # output_path = f'{os.getcwd()}/dataset/data.tfrecord'

    # Create TFRecord files
    write_tfrecord(train_label_dir, train_image_dir, train_output_path, label_map)
    write_tfrecord(eval_label_dir, eval_image_dir, eval_output_path, label_map)
    # write_tfrecord(label_dir, image_dir, output_path, label_map)

    print("TFRecord files created successfully!")
