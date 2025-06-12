import cv2
import numpy as np
import SimpleITK as sitk
import os

# read each image in the folder and print the image unique values

src_folder =r'\item_template_images\item_template_images'
dst_folder = r'\item_template_images\item_template_images_nobackground'

for filename in os.listdir(src_folder):
    
    fpath = os.path.join(src_folder, filename)
    if not os.path.isfile(fpath):
        continue
    # read the image
    ori_image = cv2.imread(fpath)
    # turn to gray
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    # convert to numpy array
    # print(image.shape)
    # print(np.unique(image.flatten()))

    # do binarization
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(np.unique(binary_image.flatten()))
    # turn the binary image to sitk format
    sitk_image = sitk.GetImageFromArray(binary_image)
    seg = sitk.ConnectedComponent(sitk_image == 255)
    new_seg = sitk.GetArrayFromImage(seg)
    uni_value, counts = np.unique(new_seg, return_counts=True)

    # map image as the back ground trasparent png
    new_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2BGRA)
    #  make the new_seg with max counts value as the transparent background
    max_index = np.argmax(counts)
    new_image[new_seg == uni_value[max_index]] = [0, 0, 0, 0]  # set the background to transparent

    # save the new image
    # save the 
    new_fpath = os.path.join(dst_folder, filename)
    
    # color segmentation part with rgb
    # nwe_image = np.zeros_like(ori_image)
    # for i in range(len(uni_value)):
    #     if uni_value[i] == 0:
    #         continue
    #     randomcolor = np.random.randint(0, 255, size=3)
    #     nwe_image[new_seg == uni_value[i]] = randomcolor
    # resize the new image to 128x128
    new_image = cv2.resize(new_image, (128, 128), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(new_fpath, new_image)



    
    

