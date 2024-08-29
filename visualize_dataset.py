import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_yolo_annotations(label_path):
    """
    Load YOLO format annotations from a .txt file.
    Args:
        label_path (str): Path to the YOLO format label file.
    Returns:
        List of bounding boxes and class labels.
    """
    with open(label_path, 'r') as file:
        annotations = file.readlines()

    boxes = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.strip().split())
        boxes.append((class_id, x_center, y_center, width, height))

    return boxes

def plot_image_with_annotations(img_path, label_path, class_mapping, padding=50):
    """
    Plot an image with YOLO annotations and add padding to display labels correctly.
    Args:
        img_path (str): Path to the image file.
        label_path (str): Path to the YOLO format label file.
        class_mapping (dict): Mapping from class index to class name.
        padding (int): Padding to add around the image to ensure labels are visible.
    """
    # Check if the image path exists
    if not os.path.exists(img_path):
        print(f"Image file {img_path} not found.")
        return
    
    img = cv2.imread(img_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Failed to load image {img_path}.")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate a color map with as many colors as there are categories
    colormap = plt.get_cmap('hsv', len(class_mapping)+1)
    colors = colormap(range(len(class_mapping)))

    # Add padding around the image
    img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=255)

    # Check if the label path exists
    if not os.path.exists(label_path):
        print(f"Label file {label_path} not found.")
        return

    boxes = load_yolo_annotations(label_path)

    for class_id, x_center, y_center, width, height in boxes:
        # Convert YOLO format to box coordinates
        h, w, _ = img.shape
        x1 = int((x_center - width / 2) * (w - 2 * padding) + padding)
        y1 = int((y_center - height / 2) * (h - 2 * padding) + padding)
        x2 = int((x_center + width / 2) * (w - 2 * padding) + padding)
        y2 = int((y_center + height / 2) * (h - 2 * padding) + padding)

        # Draw bounding box and label
        color = colors[int(class_id)] 
        # Remove the alpha channel and convert each RGB value to the range 0-255
        color = tuple(int(channel * 255) for channel in color[:3])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = cv2.putText(img, class_mapping[int(class_id)], (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)

    plt.imshow(img)
    plt.axis('off')

def plot_random_images_grid(dataset_dir, class_mapping, num_images=18):
    """
    Plot a grid of random images with YOLO format ground truth labels.
    Args:
        dataset_dir (str): Directory containing images and labels.
        class_mapping (dict): Mapping from class index to class name.
        num_images (int): Number of images to display in the grid.
    """
    images_dir = os.path.join(dataset_dir, 'images')  # Adjust extension if needed
    img_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]  # Adjust extension if needed
    print("Num images: ", len(img_files))
    random_imgs = random.sample(img_files, num_images)

    plt.figure(figsize=(14, 9))

    for i, img_file in enumerate(random_imgs):
        img_path = os.path.join(images_dir, img_file)
        labels_dir = os.path.join(dataset_dir, 'labels')
        label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))  # Adjust extension if needed

        plt.subplot(3, 6, i + 1)
        plot_image_with_annotations(img_path, label_path, class_mapping)

    plt.tight_layout()
    plt.show()

# Example usage
random.seed(43)
dataset_dir = '/home/alex/Desktop/MRS-YOLO/datasets/trash_detection'  # Update with your dataset images path
class_mapping = {0: 'Not recyclable', 1: 'Food waste', 2: 'Glass', 3: 'Textile', 4: 'Metal', 5: 'Wooden', 6: 'Leather', 7: 'Plastic', 8: 'Ceramic', 9: 'Paper'}
    
plot_random_images_grid(dataset_dir, class_mapping)
