import os
import tensorflow as tf
from object_detection.utils import dataset_util
from lxml import etree
import glob

def create_example(annotation_file, image_dir):
    tree = etree.parse(annotation_file)
    root = tree.getroot()

    filename = root.find("filename").text
    image_path = os.path.join(image_dir, filename)
    
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

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
        classes.append(1)  # Assuming all objects are of class 1
        classes_text.append(class_name.encode('utf8'))

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


def write_tfrecord(xml_dir, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    
    for xml_file in xml_files:
        tf_example = create_example(xml_file, image_dir)
        writer.write(tf_example.SerializeToString())
    
    writer.close()


if __name__ == "__main__":
    xml_dir = f'{os.getcwd()}/dataset/labels'
    image_dir = f'{os.getcwd()}/dataset/images'
    output_path = f'{os.getcwd()}/dataset/train.tfrecord'
    
    write_tfrecord(xml_dir, image_dir, output_path)
    print("TFRecord file created:", output_path)
