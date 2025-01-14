import os
import glob
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

LABEL_MAP = {
    'BOL_Number': 1,
    'Load_Date': 2,
    'Load_Time': 3,
    'Terminal': 4,
    'Supplier': 5,
    'Product_Name_List': 6,
    'Product_Gross_List': 7,
    'Product_Net_List': 8,
}

def xml_to_csv(path):
    """Converts XML annotation files to a CSV file."""
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text.strip(), 
                int(root.find('size/width').text),
                int(root.find('size/height').text),
                member[0].text.strip(),
                int(member.find('bndbox/xmin').text),
                int(member.find('bndbox/ymin').text),
                int(member.find('bndbox/xmax').text),
                int(member.find('bndbox/ymax').text)
            )
            xml_list.append(value)
    
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def create_tf_example(group, path):
    """Creates a TFRecord example from the image and annotation."""
    image_path = os.path.join(path, group.filename.iloc[0])
    if not os.path.exists(image_path):
        print(f"❌ Warning: Image file not found - {image_path}")
        return None

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    
    width, height = group.width.iloc[0], group.height.iloc[0]
    filename = group.filename.iloc[0].encode('utf8')
    image_format = b'jpg'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for _, row in group.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(LABEL_MAP.get(row['class'], 0)) 

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    
    return tf_example

def main():
    """Main function to convert XML annotations to TFRecord."""
    annotations_path = 'dataset/labels' 
    images_path = 'dataset/images' 
    csv_output = 'dataset/train_labels.csv'
    tfrecord_output = 'dataset/train.record'

    xml_df = xml_to_csv(annotations_path)
    xml_df.to_csv(csv_output, index=False)
    print(f"✅ CSV file saved at: {csv_output}")

    writer = tf.io.TFRecordWriter(tfrecord_output)
    grouped = xml_df.groupby('filename')

    for filename, group in grouped:
        tf_example = create_tf_example(group, images_path)
        if tf_example:
            writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"✅ TFRecord file saved at: {tfrecord_output}")

if __name__ == '__main__':
    main()
