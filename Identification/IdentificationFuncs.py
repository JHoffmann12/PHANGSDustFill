import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import pandas as pd
import os
import copy
import csv
import importlib
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from reproject import reproject_exact
from scipy.ndimage import zoom
from matplotlib.colors import Normalize
from glob import glob
from scipy.ndimage import gaussian_filter
import time 
import math

def ThresholdSkel(image_data, original_header, fits_out_path):
    image_data = 255 * (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    image_data = image_data.astype(np.uint8)    
    total_thresh = cv2.adaptiveThreshold(image_data, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)
    skeleton_image = skeletonize(total_thresh).astype(np.float32)
    hdu = fits.PrimaryHDU(skeleton_image, header=original_header)
    time.sleep(1)  # Pause for 1 second before writing
    hdu.writeto(fits_out_path, overwrite=True)

def create_composite_image(directory, output_name, output_directory, common_string, cutoff_perc=0):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Find FITS files based on the common_string
    file_pattern = os.path.join(directory, f"*{common_string}*.fits")
    fits_files = glob(file_pattern)

    # Read each FITS file and store the data
    images = []
    header = None

    for file in fits_files:
        with fits.open(file) as hdul:
            image_data = hdul[0].data
            if image_data is not None:
                # Binary presence indicator (values > 0 set to 1, others set to 0)
                images.append((image_data > 0).astype(np.uint8))
                if header is None:
                    header = hdul[0].header
            else:
                print(f"Skipping empty data in file: {file}")

    if not images:
        print("No images were loaded. Exiting function.")
        return

    # Stack images and count the presence of each pixel across files
    stacked_images = np.array(images)
    composite_data = np.nansum(stacked_images, axis=0)

    # Check if composite_data is an array or a scalar
    if np.isscalar(composite_data) or composite_data.size == 0:
        print("Error: composite_data is invalid or empty. Exiting function.")
        return
    else:
        print("Composite data array created successfully.")

    # Apply the cutoff percentage
    max_presence = len(images)
    cutoff_value = (cutoff_perc * max_presence) / 100  # Adjusting cutoff based on total images
    print(f"Cutoff value (presence threshold): {cutoff_value}")
    print(f"Total number of images: {max_presence}")

    composite_data[composite_data < cutoff_value] = 0  # Set pixels below the cutoff to 0

    # Normalize composite data to the range [0, 255] for grayscale representation
    composite_data = (composite_data / max_presence * 255).astype(np.uint8)

    # Save the grayscale composite as PNG
    output_png_path = os.path.join(output_directory, output_name + ".png")
    success = cv2.imwrite(output_png_path, composite_data)
    if success:
        print(f"Composite image saved as PNG: {output_png_path}")
    else:
        print("Failed to save composite image as PNG. Check image data and path.")

    # Save the grayscale composite as FITS
    output_fits_path = os.path.join(output_directory, output_name + ".fits")
    hdu = fits.PrimaryHDU(data=composite_data, header=header)
    time.sleep(1)  # Pause for 1 second before writing
    hdu.writeto(output_fits_path, overwrite=True)
    print(f"Composite image saved as FITS: {output_fits_path}")



#Erode the Boundary and return a mask for blank regions
def dilateBlankRegions(image_data, threshold, iters, kernel_size, return_nan = False):
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
    kernel_size = 11
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=iters)

    dilated_image = copy.deepcopy(copy_image)
    dilated_image[dilated_mask==255] = np.nan

    #make a mask based on dilated image, true value indicates pixel should be masked
    mask = np.isnan(dilated_image)

    #masked region in opriginal image can be returned as nan or zero
    if(not return_nan):
        dilated_image[np.isnan(dilated_image)] = 0

    return dilated_image, mask


def bkgSub(image_data, mask, scalepix, original_header, fits_BkgSub_out_path):
    #copy data
    data = copy.deepcopy(image_data.astype(np.float64)) #photutils should take float64
    #subtract bkg
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, box_size=round(10.*scalepix/2.)*2+1, coverage_mask = mask,filter_size=(3,3), bkg_estimator=bkg_estimator) #Very different RMS with mask. Minimum is MUCH larger
    data -= bkg.background #subtract bkg
    data[data < 0] = 0 #Elimate neg values. This is over estimating the background and messes up fits files

    #bkg sub/RMS map
    noise = bkg.background_rms
    noise[noise == 0] = 10**-10 #replace with small number...if noise = 0, we are in background, and output will be set to 0 anyway in two lines

    divRMS = data/noise
    divRMS[mask] = 0 #masked regions are zero...can change to some reasonable value but doesn't really matter for SOAX

    #scale divRMS
    topval=np.max(divRMS)
    divRMSscl=divRMS*65535/topval
    time.sleep(1)  # Pause for 1 second before writing
    hdu = fits.PrimaryHDU(divRMSscl, header=original_header)
    hdu.writeto(fits_BkgSub_out_path, overwrite=True)

    return divRMSscl


def crop_nan_border(image, expected_shape):
    # Create a mask of non-NaN values
    mask = ~np.isnan(image)
    
    # Find the rows and columns with at least one non-NaN value
    non_nan_rows = np.where(mask.any(axis=1))[0]
    non_nan_cols = np.where(mask.any(axis=0))[0]
    
    # Use the min and max of these indices to slice the image
    cropped_image = image[non_nan_rows.min():non_nan_rows.max() + 1, 
                          non_nan_cols.min():non_nan_cols.max() + 1]
    
    # Get the current shape of the cropped image
    current_shape = cropped_image.shape
    
    # Check if the cropped image needs to be resized
    if current_shape != expected_shape:
        # Pad or crop to reach the expected shape
        padded_image = np.full(expected_shape, np.nan)  # Initialize with NaNs
        
        # If cropped_image is larger than expected_shape, trim it
        trim_rows = min(current_shape[0], expected_shape[0])
        trim_cols = min(current_shape[1], expected_shape[1])
        
        # Center the cropped image within the expected shape
        start_row = (expected_shape[0] - trim_rows) // 2
        start_col = (expected_shape[1] - trim_cols) // 2
        
        # Place the trimmed or centered cropped_image in the padded_image
        padded_image[start_row:start_row + trim_rows, start_col:start_col + trim_cols] = \
            cropped_image[:trim_rows, :trim_cols]
        
        return padded_image
    
    return cropped_image


def upscale_image(image, scale_factor):
    # Use zoom to increase the size by the scale factor
    # Order=1 is bilinear interpolation, which is smooth and often sufficient
    upscaled_image = zoom(image, scale_factor, order=1)
    return upscaled_image


def resample_fits(fits_BkgSub_out_path, fits_block_out_path, scale_factor, save_png_path):
    scale_factor *= 2
    with fits.open(fits_BkgSub_out_path) as hdul:
        original_data = hdul[0].data
        original_header = hdul[0].header

    new_pixel_scale = abs(original_header['CDELT1']) * scale_factor
    new_header = original_header.copy()
    new_header['CDELT1'] = new_pixel_scale
    new_header['CDELT2'] = new_pixel_scale

    # Apply scale factor and adjust for 0.5-pixel offset
    new_header['CRPIX1'] = (original_header['CRPIX1'] / scale_factor) + .4687 #shift for some reason
    new_header['CRPIX2'] = (original_header['CRPIX2'] / scale_factor) + .4867 #shift for some reason

    # Define the new shape for the reprojected data
    new_shape = (int(original_data.shape[0] / scale_factor), int(original_data.shape[1] / scale_factor))

    # Reproject data
    reprojected_data, _ = reproject_exact((original_data, original_header), new_header, shape_out=new_shape)

    expected_shape = int(original_data.shape[0] / scale_factor), int(original_data.shape[1] / scale_factor)
    # Crop NaN border (if necessary)
    reprojected_data = crop_nan_border(reprojected_data, expected_shape)

    # Write the modified data and header to a new FITS file
    hdu = fits.PrimaryHDU(reprojected_data, header=new_header)  
    hdu.writeto(fits_block_out_path, overwrite=True)

    # Save a PNG for visualization
    pngData = reprojected_data.astype(np.uint16)
    cv2.imwrite(save_png_path, pngData)

    return reprojected_data, new_header

def determineParams(fits_file):

    if '128pc' in fits_file:
        print('image is 128 parcec scale')
        return 8, "128pc"
    
    elif '8pc' in fits_file: 
        print('image is 8 parcec scale')
        return 0 , "8pc"
    
    elif '16pc' in fits_file: 
        print('image is 16 parcec scale')
        return 0, "16pc"
    
    elif '32pc' in fits_file: 
        print('image is 32 parcec scale')
        return 2, "32pc"
    
    elif '64pc' in fits_file:
        print('image is 64 parcec scale')
        return 4, "64pc"

    elif '256pc' in fits_file:
        print('image is 256 parcec scale')
        return 16, "256pc"
    
    elif '512pc' in fits_file:
        print('image is 512 parcec scale')
        return 32, "512pc"
    
    elif '1024pc' in fits_file:
        print('image is 1024 parcec scale')
        return 64, "1024pc"

    else:
        print('invalid fits file')

def getDistance(fits_file):
    if('0628' in fits_file):
        return 9.84
    elif('4254' in fits_file):
        return 13.1
    elif('4303' in fits_file):
        return 16.99
    else:
        print("improper file")

def txtToFilaments(result_file, csv_file_path, blocked_data, fits_out_path, scale_factor, original_header):
    expected_header = "s p x y z fg_int bg_int"

    print(f' working on {result_file}')
    assert( os.path.isfile(result_file))
    
    # Read the content of the input file
    
    with open(result_file, 'r') as file:
        lines = file.readlines()

    # Find the index where the header starts
    header_start_index = None
    stopper_index = -1 # sometimes "[" isnt present
    for i, line in enumerate(lines):
        # Normalize line by stripping extra spaces
        normalized_line = ' '.join(line.split())

        if normalized_line == expected_header:
            header_start_index = i
        if "[" in line:
            stopper_index = i
            break

    if header_start_index is None:
        print('HEADER NOT FOUND!')
        return np.nan, np.nan
    
    # Extract header and data
    header_and_data = lines[header_start_index:stopper_index]

    # Write to CSV
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for line in header_and_data:
            # Split columns by whitespace (assuming whitespace is the delimiter)
            row = line.split()
            csv_writer.writerow(row)

    # print(f"{result_file} has been saved to {output_file_path}")

    df = pd.read_csv(csv_file_path)
    # print(df.head(10))
    # Convert coordinates to float
    x_coords = pd.to_numeric(df["x"], errors="coerce")
    # Remove NaN values

    y_coords = pd.to_numeric(df["y"], errors="coerce")
    # Remove NaN values

    coords = pd.DataFrame({"x": x_coords, "y": y_coords}).dropna()

    # Now you can safely round and convert both x and y together
    x_coords_cleaned = [int(round(x)) for x in coords["x"]]
    y_coords_cleaned = [int(round(y)) for y in coords["y"]]

    assert(len(x_coords_cleaned)==len(y_coords_cleaned))
    coordinates = list(zip(x_coords_cleaned,y_coords_cleaned))


    gray_image = np.zeros_like(blocked_data)
    # Ensure coordinates are within the image bounds
    for (y,x) in coordinates:
        if 0 <= x < gray_image.shape[0] and 0 <= y < gray_image.shape[1]:
            gray_image[x, y] = 1

    #get dictionary for filaments
    filament_dict = {}

    # Grouping the coordinates
    for index, row in df.iterrows():
        s_value = row['s']
        if "#" not in s_value:
            s_value = int(s_value)   
            if s_value not in filament_dict:
                filament_dict[s_value] = []  # Initialize the list for this s value
            # Append the (x, y) coordinates as a tuple
            filament_dict[s_value].append((row['x'], row['y']))

    hdu = fits.PrimaryHDU(gray_image, header=original_header)
    time.sleep(1)  # Pause for 1 second before writing
    hdu.writeto(fits_out_path, overwrite=True)
    print('Success, fits file created from SOAX')

    return filament_dict, gray_image

def restore_size(fits_path, data, header, scale_factor):
    scale_factor = 2*scale_factor
    data = upscale_image(data, scale_factor)
    hdu = fits.PrimaryHDU(data, header=header)
    time.sleep(1)  # Pause for 1 second before writing
    hdu.writeto(fits_path, overwrite=True)

    

# functions below are For F function only:

# def computeF(filament_dict, noise, gray_image,t,c, s=1000, v=500):
#     #Computer average length
#     total_filaments = len(filament_dict)
#     total_pixels = sum(len(filaments) for filaments in filament_dict.values())
#     average_tuples_per_key = total_pixels / total_filaments if total_filaments > 0 else 0
#     #get number intersections
#     intersections = len(getSkeletonIntersection(np.array(gray_image)))
#     #get total number of pixels
#     pix_total = np.sum(gray_image)
#     # Create a mask where noise map values are below the threshold
#     mask = noise < t
#     # Use the mask to select values from the binary image and sum them
#     noise_sum = np.sum(gray_image[mask])

#     return -pix_total + c*noise_sum, -pix_total + c*noise_sum - s*average_tuples_per_key + v*intersections


# def neighbours(x, y, image):
#     """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
#     img = image
#     x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
#     return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]

# def getSkeletonIntersection(skeleton):
#     """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.
    
#     Keyword arguments:
#     skeleton -- the skeletonised image to detect the intersections of
    
#     Returns: 
#     List of 2-tuples (x,y) containing the intersection coordinates
#     """
#     validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
#                          [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
#                          [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
#                          [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
#                          [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
#                          [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
#                          [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
#                          [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
#                          [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
#                          [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
#                          [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
#                          [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
#                          [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
#                          [1,0,1,1,0,1,0,0]]
#     image = skeleton.copy() / 255
#     intersections = []
#     for x in range(1, len(image) - 1):
#         for y in range(1, len(image[x]) - 1):
#             # If we have a white pixel
#             if image[x][y] == 1:
#                 neighbors = neighbours(x, y, image)
#                 if neighbors in validIntersection:
#                     intersections.append((y, x))
    
#     # Filter intersections to make sure we don't count them twice or ones that are very close together
#     filtered_intersections = []
#     for point1 in intersections:
#         add_point = True
#         for point2 in filtered_intersections:
#             if ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2:
#                 add_point = False
#                 break
#         if add_point:
#             filtered_intersections.append(point1)
    
#     return filtered_intersections


