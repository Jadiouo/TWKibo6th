from Datahelp import *
import numpy as np
import cv2
from PIL import Image
import os



def main():

    png_path = r'\Democode\item_template_images\Kibodataset\train\images\0002.png'

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
            'brightness': 0,
            # 'brightness': 1,
            'contrast': 0,
            # 'contrast': 1,
            'sharpness': 0,
            # 'sharpness': 1,
            'gaussian_noise': 0,
            # 'gaussian_noise': 1,
            # 'gaussian_blur': 0,
            'gaussian_blur': 1,
            'motion_blur': 0,
            # 'motion_blur': 1,
            'gradient_brightness': 0,
            # 'gradient_brightness': 1,
            'perspective': 0,
            # 'perspective': 1,
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
            'gaussian_noise_list': [ (-0.25,0.25), (0.01, 0.1), (0.1,0.5)]# (-0.25,0.25), (0.01, 0.1), (0.1,0.5)
        }
    }
    # Load the image
    print(f"Loading image from: {png_path}")
    img = cv2.imread(png_path)
    if img is None:
        print(f"Error: Could not load image from {png_path}")
        return
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a dummy mask (same height and width as image, single channel)
    mask = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
    
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Create outer augmentation transform
    outer_augmenter = Augmentationtransform2D(Augmetation_dict, aug_type='outer')
    
    # Apply outer augmentation
    print("Applying outer augmentation...")
    augmented_img, augmented_mask = outer_augmenter(img, mask)
    # write original image and augmented image to check the result

    cv2.imwrite('original_image.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # Save the result
    out_dir =r'\Democode\item_template_images'
    output_path = os.path.join(out_dir, 'outer_test.png')
    cv2.imwrite(output_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
    
    print(f"Outer augmentation result saved as: {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()