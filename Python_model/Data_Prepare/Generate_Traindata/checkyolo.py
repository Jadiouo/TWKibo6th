import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import tkinter as tk
from tkinter import filedialog

def check_yolo_labels(label_folder, image_folder, check_folder, total_class=11):
    """
    Plot YOLO labels on images and save in check_folder
    
    Args:
        label_folder: Path to folder containing YOLO label files (.txt)
        image_folder: Path to folder containing corresponding images
        check_folder: Path to folder where annotated images will be saved
        total_class: Number of classes for color generation
    """
    
    # Create check folder if it doesn't exist
    if not os.path.exists(check_folder):
        os.makedirs(check_folder)
    
    # Generate colors for each class using HSV color space
    colors = []
    for i in range(total_class):
        hue = i / total_class
        saturation = 0.8
        value = 0.9
        rgb = hsv_to_rgb([hue, saturation, value])
        # Convert to BGR format for OpenCV (0-255 range)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)
    
    # Get all label files
    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]
    
    print(f"Found {len(label_files)} label files to process...")
    
    processed_count = 0
    
    for label_file in label_files:
        # Get corresponding image file
        base_name = os.path.splitext(label_file)[0]
        
        # Try common image extensions
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            potential_image = os.path.join(image_folder, base_name + ext)
            if os.path.exists(potential_image):
                image_file = potential_image
                break
        
        if image_file is None:
            print(f"Warning: No corresponding image found for {label_file}")
            continue
        
        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read image {image_file}")
            continue
        
        height, width = image.shape[:2]
        
        # Read YOLO labels
        label_path = os.path.join(label_folder, label_file)
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Draw bounding boxes
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])
                
                # Convert from YOLO format to pixel coordinates
                x_center_px = int(x_center * width)
                y_center_px = int(y_center * height)
                bbox_width_px = int(bbox_width * width)
                bbox_height_px = int(bbox_height * height)
                
                # Calculate top-left and bottom-right coordinates
                x1 = int(x_center_px - bbox_width_px / 2)
                y1 = int(y_center_px - bbox_height_px / 2)
                x2 = int(x_center_px + bbox_width_px / 2)
                y2 = int(y_center_px + bbox_height_px / 2)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                # Get color for this class
                if class_id < len(colors):
                    color = colors[class_id]
                else:
                    color = (0, 255, 0)  # Default green if class_id exceeds available colors
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw class ID label
                label_text = f"Class {class_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 1
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, font_thickness
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    image, 
                    (x1, y1 - text_height - baseline - 5), 
                    (x1 + text_width, y1), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    image, 
                    label_text, 
                    (x1, y1 - 5), 
                    font, 
                    font_scale, 
                    (255, 255, 255), 
                    font_thickness
                )
            
            # Save annotated image
            output_path = os.path.join(check_folder, f"annotated_{base_name}.jpg")
            cv2.imwrite(output_path, image)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} images...")
                
        except Exception as e:
            print(f"Error processing {label_file}: {str(e)}")
            continue
    
    print(f"Successfully processed {processed_count} images.")
    print(f"Annotated images saved in: {check_folder}")


# Main execution
if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    mainfolder = filedialog.askdirectory(title="Select the main folder for Kibo dataset")
    root.destroy()  # Destroy the root window after selection
    # Define paths
    label_folder = os.path.join(mainfolder, 'labels')
    image_folder = os.path.join(mainfolder, 'images')
    check_folder = os.path.join(mainfolder, 'check')
    total_class = 11
    
    # Run the function
    check_yolo_labels(label_folder, image_folder, check_folder, total_class)