import os
import cv2
import numpy as np

def rename_images(image_folder, prefix="img"):
    """
    Rename all images in the folder to the format img1.jpg, img2.jpg, etc.
    """
    files = os.listdir(image_folder)
    
    img_count = 1
    
    for filename in files:
        file_path = os.path.join(image_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            new_name = f"{prefix}{img_count}.jpg"
            new_file_path = os.path.join(image_folder, new_name)
            
            os.rename(file_path, new_file_path)
            print(f"Renamed {filename} to {new_name}")
            
            img_count += 1

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Preprocess a single image by resizing, padding, and normalizing.
    Replaces the original image with the preprocessed one, using higher-quality interpolation.
    """
    img = cv2.imread(image_path)
    
    aspect_ratio = img.shape[1] / img.shape[0]
    if aspect_ratio > 1:
        width = target_size[0]
        height = int(width / aspect_ratio)
    else:
        height = target_size[1]
        width = int(height * aspect_ratio)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    padded_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    padded_image[:img_resized.shape[0], :img_resized.shape[1]] = img_resized

    img_normalized = padded_image / 255.0

    cv2.imwrite(image_path, img_normalized * 255)  

def preprocess_multiple_images(image_folder, target_size=(512, 512)):
    """
    Preprocess multiple images in a folder by resizing, padding, and normalizing,
    replacing the original images with preprocessed ones.
    """
    files = os.listdir(image_folder)
    
    for filename in files:
        image_path = os.path.join(image_folder, filename)
        if os.path.isfile(image_path) and image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            preprocess_image(image_path, target_size) 

image_folder = 'dataset/images'
rename_images(image_folder)  
preprocess_multiple_images(image_folder) 
