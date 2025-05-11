import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

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
    color_map = plt.get_cmap('hsv', len(class_mapping) + 1)
    colors = color_map(range(len(class_mapping)))

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

def plot_random_images_grid(
    dataset_dir, 
    class_mapping, 
    num_images=18, 
    padding=50, 
    output_dir='dataset_examples',
    filename='trash_detection'):
    """
    Plot a grid of random images with YOLO format ground truth labels.
    Args:
        dataset_dir (str): Directory containing images and labels.
        class_mapping (dict): Mapping from class index to class name.
        num_images (int): Number of images to display in the grid.
    """
    images_dir = os.path.join(dataset_dir, 'images')  # Adjust extension if needed
    img_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]  # Adjust extension if needed
    
    if len(img_files) == 0:
        raise ValueError(f"No images found in {images_dir}. Please check the directory.")
    
    print("Num images: ", len(img_files))
    random_imgs = random.sample(img_files, num_images)

    plt.figure(figsize=(8.27, 11.69))  # A4 size in inches (8.27 x 11.69)

    for i, img_file in enumerate(random_imgs):
        img_path = os.path.join(images_dir, img_file)
        labels_dir = os.path.join(dataset_dir, 'labels')
        label_path = os.path.join(labels_dir, img_file.replace('.jpg', '.txt'))  # Adjust extension if needed

        plt.subplot(int(num_images/3), 3, i + 1, adjustable='box', aspect='auto')
        plot_image_with_annotations(img_path, label_path, class_mapping, padding=padding)

    plt.tight_layout()
    # plt.show()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a filename for the output image
    output_filename = os.path.join(output_dir, f"{filename}.png")

    # Check if the output file already exists -> increment the filename
    base_filename, ext = os.path.splitext(output_filename)
    counter = 1
    while os.path.exists(output_filename):
        output_filename = f"{base_filename}_{counter}{ext}"
        counter += 1
    print(f"Saving to {output_filename}")
    
    # Save the figure
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.axis('off')
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gca().set_frame_on(False)
    plt.gcf().canvas.draw()
    plt.gcf().set_size_inches(8.27, 11.69)  # A4 size in inches (8.27 x 11.69)
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset with annotations.")
    parser.add_argument('-s', '--seed', type=int, default=43, help='Random seed for selecting images.')
    parser.add_argument('-d', '--dataset_dir', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('-p', '--padding', type=int, default=50, help='Padding around the image for labels.')
    parser.add_argument('-n', '--num_images', type=int, default=18, help='Number of images to display in the grid. Must be a multiple of 3.')
    parser.add_argument('-o', '--output_dir', type=str, default='dataset_examples', help='Output directory for saving the images.')
    parser.add_argument('-fn', '--filename', type=str, default='trash_detection', help='Filename prefix for the output images.')
    
    args = parser.parse_args()

    # Validate num_images
    if args.num_images % 3 != 0:
        raise ValueError("num_images must be a multiple of 3.")

    # Set random seed
    random.seed(args.seed)

    # Use dataset directory from arguments
    dataset_dir = os.path.abspath(args.dataset_dir)
    class_mapping = {0: 'Not recyclable', 1: 'Food waste', 2: 'Glass', 3: 'Textile', 4: 'Metal', 5: 'Wooden', 6: 'Leather', 7: 'Plastic', 8: 'Ceramic', 9: 'Paper'}

    plot_random_images_grid(
        dataset_dir, 
        class_mapping, 
        num_images=args.num_images, 
        padding=args.padding)
