import copy
import numpy as np
import cv2
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


#Erode the Boundary and return a mask for blank regions
def dilateBlankRegions(image_data, threshold, return_nan = False):

    copy_image = copy.deepcopy(image_data)

    #Dilate White Pixels-->make more concise
    mask_copy = copy.deepcopy(copy_image)
    mask_copy[mask_copy > threshold] = 255
    mask_copy = mask_copy.astype(np.uint8)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(mask_copy, kernel, iterations=30)

    #Dilate nan pixels--> make more concise
    copy_image = copy_image.astype(np.float32)
    copy_image[dilated_image < threshold] = np.nan
    mask = np.isnan(copy_image)
    binary_mask = np.zeros_like(copy_image, dtype=np.uint8)
    binary_mask[mask] = 255
    binary_mask[~mask] = 0
    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=2)

    dilated_image = copy.deepcopy(copy_image)
    dilated_image[dilated_mask==255] = np.nan

    #Plot new image
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('nan_color', [(0, 0, 0), (1, 1, 1)], N=256)
    cmap.set_bad('gray')  
    plt.imshow(np.flipud(dilated_image), cmap=cmap)
    plt.axis('off')  
    plt.title('Image with grey pixels masked')
    plt.show()


    #make a mask based on dilated image, true value indicates pixel should be masked
    mask = np.isnan(dilated_image)

    #masked region in opriginal image can be returned as nan or zero
    if(not return_nan):
        dilated_image[np.isnan(dilated_image)] = 0

    return dilated_image, mask