from Datahelp import *
import os
import tkinter as tk
from tkinter import filedialog
"""
    KiboDataset if you call it it can autogenerate the dataset for training
    Args:
        datainfo: Dictionary containing dataset information
                  {
                      'Blank_image_size': (width, height),
                      'All_itemclass': [list of item classes],
                      'Treasure_item': path to treasure items,
                      'Landmark_item': path to landmark items,
                      'Lost_treasure_rate': float,
                      'Targetitem_rate': float,
                      'Batch_number': int
                      'Overlap_rate': float,
                  }
        transforminfo: Dictionary containing transformation information
                      {
                          'Augmentation': augmentation parameters,
                          'Max_landmark_items': int,
                          'Outer_effect': float,
                          'Output_transform': transformation parameters
                      }
        In augmentation parameters
        See the dict description bin Datahelp.py for details 
        augment_dict: dictionary with the key structure:
            {
                'inner': {
                    'flip': bool,                           # Enable horizontal/vertical flips
                    'rotate': bool,                         # Enable rotation
                    'scale': bool,                          # Enable scaling
                    'flip_rate': float,                     # Probability of flipping (0.0-1.0)
                    'rotate_range': (min_angle, max_angle), # Rotation range in degrees
                    'scale_range': (min_scale, max_scale)   # Scale factor range
                },
                'outer': {
                    # Probability rates for each transformation (0.0-1.0)
                    'brightness': float,                    # Probability of brightness adjustment
                    'contrast': float,                      # Probability of contrast adjustment
                    'sharpness': float,                     # Probability of sharpness adjustment
                    'gaussian_noise': float,                # Probability of adding Gaussian noise
                    'gaussian_blur': float,                 # Probability of Gaussian blur
                    'motion_blur': float,                   # Probability of motion blur
                    'gradient_brightness': float,           # Probability of gradient brightness
                    'perspective': float,                   # Probability of perspective transform
                    
                    # Parameter ranges for transformations
                    'brightness_range': (min_brightness, max_brightness),     # Brightness adjustment range (e.g., (-0.3, 0.3))
                    'contrast_range': (min_contrast, max_contrast),           # Contrast adjustment range (e.g., (-0.2, 0.2))
                    'sharpness_range': (min_sharpness, max_sharpness),       # Sharpness factor range (e.g., (0.5, 2.0))
                    
                    # Blur parameters
                    'blur_kernel_size': (height, width),                     # Gaussian blur kernel size (odd integers)
                    'blur_sigma': (min_sigma, max_sigma),                     # Gaussian blur sigma range
                    'motion_blur_kernel_size': (min_size, max_size),         # Motion blur kernel size range (odd integers selected)
                    'motion_blur_angle_range': (min_angle, max_angle),       # Motion blur angle range in degrees
                    'motion_blur_distance_range': (min_distance, max_distance), # Motion blur distance range
                    
                    # Gradient brightness parameters (UPDATED)
                    'gradient_brightness_range': (min_factor, max_factor),   # Gradient brightness factor range
                    
                    # Perspective transform
                    'perspective_distortion': float,                         # Perspective distortion scale (0.0-1.0)
                    
                    # Gaussian noise parameters (UPDATED)
                    'gaussian_noise_list': [
                        (mean_min, mean_max),           # Mean range for Gaussian noise
                        (std_min_min, std_min_max),     # Minimum std deviation range  
                        (std_max_min, std_max_max)      # Maximum std deviation range
                    ]
                }
            }

"""


if __name__ == "__main__":
    # mainfolder = r'H:\Postgraudate\KIBORPC\Democode\item_template_images\Kibodataset\train'
    # source_img_folder = r'H:\Postgraudate\KIBORPC\Democode\item_template_images\item_template_images_filter'
    # use tkinter to select the main folder and source image folder
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    mainfolder = filedialog.askdirectory(title="Select the main folder for Kibo dataset")
    source_img_folder = filedialog.askdirectory(title="Select the source image folder for Kibo dataset")
    root.destroy()  # Destroy the root window after selection
    Store_image_folder = os.path.join(mainfolder, 'images')
    Store_label_folder = os.path.join(mainfolder, 'labels')
    Store_class_folder = os.path.join(mainfolder, 'classes_count')
    # Ensure the folders exist
    if not os.path.exists(Store_image_folder):
        os.makedirs(Store_image_folder)
    if not os.path.exists(Store_label_folder):
        os.makedirs(Store_label_folder)
    if not os.path.exists(Store_class_folder):
        os.makedirs(Store_class_folder)

    # Clean the data initialization
    for folder in [Store_image_folder, Store_label_folder, Store_class_folder]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)


    blank_image_size = (480, 640) # height, width
    template_size = (320, 320)
    output_img_size = (320, 320)  # Output image size for the dataset
    
    source_images_files = [f for f in os.listdir(source_img_folder) if f.endswith('.png')]
    All_class = [os.path.basename(f).split('.')[0] for f in source_images_files]
    Treasure_tag = ["crystal", "diamond", "emerald"]
    # use treasure tag to split source to landmarks_item_path and treasure_item_path
    Treasure_item_path, Landmark_item_path = [], []
    for item in source_images_files:
        if any(tag in item for tag in Treasure_tag):
            Treasure_item_path.append(os.path.join(source_img_folder, item))
        else:
            Landmark_item_path.append(os.path.join(source_img_folder, item))

    datainfo = {
        'Blank_image_size': blank_image_size,
        'templateresize': template_size,  # Resize template images to this size
        'All_itemclass': All_class,
        'Treasure_item': Treasure_item_path,
        'Landmark_item': Landmark_item_path,
        'Lost_treasure_rate': 0.2,  # 50% of the images will have lost treasure items
        'Targetitem_rate': 0.2,  # 50% of the images will have target items
        'Batch_number': 10,  # Number of images to generate
        'Overlap_rate': 0.5  # 10% of the images will have overlapping items
    }
    Augmetation_dict = {
        'inner': {
            'flip': True,
            'rotate': True,
            'scale': True,
            'flip_rate': 0.5,
            'rotate_range': (-30, 30),  # degrees
            'scale_range': (0.4, 0.8)  # scale factor
        },
        'outer': {
            'brightness': 0.2,
            'contrast': 0.2,
            'sharpness': 0.2,
            'gaussian_noise': 0.1,
            'gaussian_blur': 0.5,
            'motion_blur': 0.5,
            'gradient_brightness': 0.3,
            'perspective': 0.2,
            'brightness_range': (0.5, 1.5),
            'contrast_range': (0.8, 1.2),
            'sharpness_range': (0.8, 1.2),
            'blur_kernel_size': (7, 7),
            'blur_sigma': (0.5, 3),
            'motion_blur_kernel_size': (7, 27),
            'motion_blur_angle_range': (-90, 90),
            'motion_blur_distance_range': (5, 40),
            'gradient_brightness_range': (0.1, 10),
            'perspective_distortion': 0.3,
            'gaussian_noise_list': [ (-0.25,0.25), (0.01, 0.1), (0.1,0.5)]
        }
    }

    transforminfo = {
        'Augmentation': Augmetation_dict,
        'Max_landmark_items': 7,  # Maximum number of landmark items in an image
        'Outer_effect': 0.4,  # Outer effect strength
        'Output_transform': None  # Placeholder for any output transformation
    }

    # Create the dataset
    Kibo_datasets = KiboDataset(datainfo, transforminfo)
    current_count = 1
    for _ in range(5):  # Generate 30 batches of data
        # Generate the dataset
        Kibo_datasets.initialize_batchdata()
        for count, data in enumerate(Kibo_datasets):
            img, label, classes = data
            # make img map to [0, 255] and convert to uint8
            # img = (img * 255).astype(np.uint8)
            # current count = 0 -> "0000"
            fname_title = f"{current_count:04d}"
            # Save the image
            img_save_path = os.path.join(Store_image_folder, f"{fname_title}.png")
            # Resize image to output_img_size
            # img = cv2.resize(img, output_img_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(img_save_path, img)

            # Save the label
            label_save_path = os.path.join(Store_label_folder, f"{fname_title}.txt")
            with open(label_save_path, 'w') as f:
                for item in label:
                    f.write(f"{item[0]} {item[1]} {item[2]} {item[3]} {item[4]}\n")
            # Save the class count
            class_count_save_path = os.path.join(Store_class_folder, f"{fname_title}.txt")
            # use set to count unique classes
            unique_classes = set(classes)
            # count the number of each class
            class_count = {cls: classes.count(cls) for cls in unique_classes}
            with open(class_count_save_path, 'w') as f:
                for cls, count in class_count.items():
                    f.write(f"{cls} {count}\n")
            current_count += 1

    # write the class names to the source folder
    class_names_path = os.path.join(mainfolder, 'class_names.txt')
    with open(class_names_path, 'w') as f:
        for count, cls in enumerate(All_class):
            f.write(f"{count}:{cls}\n")

     