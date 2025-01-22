import os
import cv2
import numpy as np

def update_annotations_with_padding(xml_path, original_width, original_height, resized_width, resized_height, x_offset, y_offset, target_width, target_height):
    """
    Update bounding box annotations in the XML file to align with resized and padded images.
    Converts bounding box coordinates to normalized values (0-1) based on the final target size.
    """
    import xml.etree.ElementTree as ET

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')

        # Read original bounding box coordinates
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Step 1: Scale bounding box coordinates to the resized image dimensions
        scaled_xmin = (xmin / original_width) * resized_width
        scaled_ymin = (ymin / original_height) * resized_height
        scaled_xmax = (xmax / original_width) * resized_width
        scaled_ymax = (ymax / original_height) * resized_height

        # Step 2: Adjust bounding box coordinates for padding
        padded_xmin = scaled_xmin + x_offset
        padded_ymin = scaled_ymin + y_offset
        padded_xmax = scaled_xmax + x_offset
        padded_ymax = scaled_ymax + y_offset

        # Step 3: Normalize the bounding box coordinates to the target dimensions
        norm_xmin = padded_xmin / target_width
        norm_ymin = padded_ymin / target_height
        norm_xmax = padded_xmax / target_width
        norm_ymax = padded_ymax / target_height

        # Update XML with absolute and normalized bounding box coordinates
        bndbox.find('xmin').text = str(int(padded_xmin))  # Absolute value
        bndbox.find('ymin').text = str(int(padded_ymin))  # Absolute value
        bndbox.find('xmax').text = str(int(padded_xmax))  # Absolute value
        bndbox.find('ymax').text = str(int(padded_ymax))  # Absolute value

        # If your XML has normalized values, add or update them:
        if bndbox.find('xcenter') is not None:  # Example of normalized keys
            bndbox.find('xcenter').text = str(round((norm_xmin + norm_xmax) / 2, 6))
            bndbox.find('ycenter').text = str(round((norm_ymin + norm_ymax) / 2, 6))
        else:
            # Add normalized values to XML if not present
            ET.SubElement(bndbox, 'normalized_xmin').text = str(round(norm_xmin, 6))
            ET.SubElement(bndbox, 'normalized_ymin').text = str(round(norm_ymin, 6))
            ET.SubElement(bndbox, 'normalized_xmax').text = str(round(norm_xmax, 6))
            ET.SubElement(bndbox, 'normalized_ymax').text = str(round(norm_ymax, 6))

    # Write the updated XML back to the file
    tree.write(xml_path)


def preprocess_image(image_path, xml_path, target_size=(512, 512)):
    """
    Preprocess a single image by resizing, padding, and normalizing.
    Ensures padding is centered, avoiding distortion.
    Updates bounding box annotations to align with resized and padded images.
    """
    import xml.etree.ElementTree as ET

    # Load the image
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]

    # Resize the image while maintaining aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:  # Wide image
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Tall or square image
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Create a blank padded canvas
    padded_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Center the resized image on the padded canvas
    y_offset = (target_size[1] - new_height) // 2
    x_offset = (target_size[0] - new_width) // 2
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

    # Normalize bounding boxes in the XML file
    if xml_path:
        update_annotations_with_padding(xml_path, original_width, original_height, new_width, new_height, x_offset, y_offset, target_size[0], target_size[1])

    # Save the preprocessed image
    cv2.imwrite(image_path, padded_image)


def preprocess_multiple_images(image_folder, xml_folder, target_size=(512, 512)):
    """
    Preprocess multiple images in a folder by resizing, padding, and normalizing.
    Also processes the corresponding XML annotations.
    """
    files = os.listdir(image_folder)

    for filename in files:
        image_path = os.path.join(image_folder, filename)
        xml_path = os.path.join(xml_folder, filename.replace('.jpg', '.xml'))

        if os.path.isfile(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            if os.path.exists(xml_path):  # Ensure XML exists
                bboxes = preprocess_image(image_path, xml_path, target_size)
                print(f"Processed {filename} with bounding boxes: {bboxes}")
            else:
                print(f"Warning: No XML file found for {filename}, skipping bounding box normalization.")


# Define paths
image_folder = 'dataset/images'
xml_folder = 'dataset/annotations'

# Run preprocessing  
preprocess_multiple_images(image_folder, xml_folder)  
