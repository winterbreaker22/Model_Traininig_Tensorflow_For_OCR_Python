import os
import xml.etree.ElementTree as ET
from PIL import Image

def normalize_bboxes(xml_file, image_path):
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        xmin_normalized = xmin / image_width
        ymin_normalized = ymin / image_height
        xmax_normalized = xmax / image_width
        ymax_normalized = ymax / image_height

        bndbox.find('xmin').text = str(xmin_normalized)
        bndbox.find('ymin').text = str(ymin_normalized)
        bndbox.find('xmax').text = str(xmax_normalized)
        bndbox.find('ymax').text = str(ymax_normalized)

    tree.write(xml_file) 

def process_annotations(image_folder, xml_folder):
    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            image_path = os.path.join(image_folder, xml_file.replace('.xml', '.jpg'))
            if os.path.exists(image_path):
                normalize_bboxes(os.path.join(xml_folder, xml_file), image_path)

def preprocess_dataset(dataset_dir):
    train_image_folder = os.path.join(dataset_dir, 'train', 'images')
    train_xml_folder = os.path.join(dataset_dir, 'train', 'labels')
    
    eval_image_folder = os.path.join(dataset_dir, 'eval', 'images')
    eval_xml_folder = os.path.join(dataset_dir, 'eval', 'labels')
    
    print("Processing training annotations...")
    process_annotations(train_image_folder, train_xml_folder)
    
    print("Processing evaluation annotations...")
    process_annotations(eval_image_folder, eval_xml_folder)

dataset_dir = f'{os.getcwd()}/dataset'
preprocess_dataset(dataset_dir)
