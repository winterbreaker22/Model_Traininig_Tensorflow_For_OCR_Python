import os

def rename_images(directory):
    # Get a list of image files in the directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]
    
    # Sort files to maintain order
    files.sort()
    
    # Rename files sequentially and change extension to .jpg
    for index, file in enumerate(files, start=1):
        old_path = os.path.join(directory, file)
        new_filename = f"img{index}.jpg"  # Change extension to .jpg
        new_path = os.path.join(directory, new_filename)
        
        os.rename(old_path, new_path)
        print(f'Renamed: {file} -> {new_filename}')

if __name__ == "__main__":
    directory = input("Enter the path to the directory: ")
    directory = os.path.normpath(directory)  # Normalize the input path
    if os.path.isdir(directory):
        rename_images(directory)
    else:
        print("Invalid directory path.")
