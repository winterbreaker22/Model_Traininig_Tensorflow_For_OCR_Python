import os

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
            
# Define path
image_folder = 'dataset/images'

# Rename
rename_images(image_folder)  