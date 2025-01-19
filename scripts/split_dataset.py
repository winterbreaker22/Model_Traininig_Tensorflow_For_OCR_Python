import os
import random
import shutil

def split_dataset(image_dir, label_dir, train_dir, eval_dir, train_percentage=0.8):
    images = os.listdir(image_dir)
    random.shuffle(images)

    train_size = int(len(images) * train_percentage)
    train_images = images[:train_size]
    eval_images = images[train_size:]

    # Create directories for training and evaluation data
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'labels'), exist_ok=True)

    for img in train_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, 'images', img))
        shutil.move(os.path.join(label_dir, img.replace('.jpg', '.xml')), os.path.join(train_dir, 'labels', img.replace('.jpg', '.xml')))

    for img in eval_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(eval_dir, 'images', img))
        shutil.move(os.path.join(label_dir, img.replace('.jpg', '.xml')), os.path.join(eval_dir, 'labels', img.replace('.jpg', '.xml')))

if __name__ == "__main__":
    split_dataset(f'{os.getcwd()}/dataset/images', f'{os.getcwd()}/dataset/labels', f'{os.getcwd()}/dataset/train', f'{os.getcwd()}/dataset/eval')
