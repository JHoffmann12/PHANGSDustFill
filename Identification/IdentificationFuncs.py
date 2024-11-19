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
from astropy.wcs import WCS

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
        return 16, "128pc", 128
    
    elif '8pc' in fits_file: 
        print('image is 8 parcec scale')
        return 0 , "8pc", 8
    
    elif '16pc' in fits_file: 
        print('image is 16 parcec scale')
        return 2, "16pc", 16
    
    elif '32pc' in fits_file: 
        print('image is 32 parcec scale')
        return 4, "32pc", 32
    
    elif '64pc' in fits_file:
        print('image is 64 parcec scale')
        return 8, "64pc", 64

    elif '256pc' in fits_file:
        print('image is 256 parcec scale')
        return 32, "256pc", 256
    
    elif '512pc' in fits_file:
        print('image is 512 parcec scale')
        return 64, "512pc", 512
    
    elif '1024pc' in fits_file:
        print('image is 1024 parcec scale')
        return 128, "1024pc", 1024

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

def txtToFilaments(result_file, csv_file_path, blocked_data, fits_out_path, scale_factor, new_header, original_data, original_header, interpolate_path, img_scale_int, scalepix):
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

    intensity = pd.to_numeric(df["fg_int"], errors="coerce")
    # Remove NaN values

    coords = pd.DataFrame({"x": x_coords, "y": y_coords, "intensity": intensity}).dropna()

    # Now you can safely round and convert both x and y together
    x_coords_cleaned = [int(round(x)) for x in coords["x"]]
    y_coords_cleaned = [int(round(y)) for y in coords["y"]]

    assert(len(x_coords_cleaned)== len(y_coords_cleaned) and len(y_coords_cleaned) == len(coords["intensity"]))
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
            filament_dict[s_value].append((int(round(row['x'])), int(round(row['y'])), row['fg_int']))

    interpolate(filament_dict, new_header, original_header, original_data, interpolate_path, img_scale_int, scalepix)

    hdu = fits.PrimaryHDU(gray_image, header=new_header)
    time.sleep(1)  # Pause for 1 second before writing
    hdu.writeto(fits_out_path, overwrite=True)
    print('Success, fits file created from SOAX')

    return filament_dict, gray_image

def restore_size(fits_path, data, header, scale_factor):
    data = upscale_image(data, scale_factor)
    hdu = fits.PrimaryHDU(data, header=header)
    time.sleep(1)  # Pause for 1 second before writing
    hdu.writeto(fits_path, overwrite=True)

    

def interpolate(dict, new_header, original_header, original_data, path, img_scale_int, scalepix):

    wcs_orig = WCS(original_header)
    wcs_blocked = WCS(new_header)
    final_image = np.zeros_like(original_data)

    filaments = dict.values()
    for filament in filaments:
        x_coords = []
        y_coords = []
        intensity = []
        gray_image = np.zeros_like(original_data)
        for pixel in filament:
            x_coords.append(pixel[0])
            y_coords.append(pixel[1])
            intensity.append(pixel[2])
        world_coords = wcs_blocked.pixel_to_world(x_coords, y_coords)
        x_original, y_original = wcs_orig.world_to_pixel(world_coords)
        # Filter out NaN coordinates and ensure coordinates are valid
        coordinates = [(x, y) for x, y in zip(x_original, y_original) if not (np.isnan(x) or np.isnan(y))]

        # # Ensure coordinates are within the image bounds
        # for (x, y, i) in coordinates:
        #     x = int(round(x))  # Round to nearest integer
        #     y = int(round(y))  # Round to nearest integer
        #     if 0 <= x < gray_image.shape[1] and 0 <= y < gray_image.shape[0]:
        #         gray_image[y, x] = 1  # Mark the pixel as i for intensity

        # new_image = normalize_image(gray_image)
        # new_image = connect_points_bw(new_image)
        new_image = connect_points_sequential(coordinates, np.shape(original_data), img_scale_int, scalepix, original_data)
        final_image+=new_image

    # Save the output image as a FITS file
    hdu = fits.PrimaryHDU(final_image, header=original_header)
    hdu.writeto(path, overwrite=True)

def normalize_image(image):
    """Normalize the image so that all pixels are between 0 and 255."""
    # Find the minimum and maximum pixel values in the image
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Normalize the image: scale it to be between 0 and 255
    normalized_image = (image - min_val) / (max_val - min_val) * 255
    
    # Convert the normalized image to integers
    normalized_image = np.round(normalized_image).astype(np.uint8)
    
    return normalized_image


def connect_points_sequential(points, image_shape, img_scale_int, scalepix, original_data):
    """
    Connects points in sequential order with straight lines.
    
    Parameters:
        points (list of tuple): List of points (x, y) to connect in sequential order.
        image_shape (tuple): Shape of the output image (height, width).

    Returns:
        numpy.ndarray: Black-and-white image with points connected by lines.
    """
    #get width
    radius =  int(img_scale_int/scalepix)
    # Create a blank black image

    points = [(int(x), int(y)) for x, y in points]

    output_array = np.zeros(image_shape, dtype=np.uint8)

      # Draw lines between consecutive points
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        if(original_data[y1,x1] > 3000):
            x2, y2 = points[i + 1]
            cv2.line(output_array, (x1, y1), (x2, y2), 255, thickness=radius)
            cv2.circle(output_array, (x1, y1), radius, 255, thickness=-1)  # Filled circle

    return output_array


def threshSkel(fits_file, out_path, prob = 15):

    with fits.open(fits_file, mode="update") as hdul:
        original_header = hdul[0].header
        image_data = np.array(hdul[0].data) 

        # Thresholding
        _, thresh1 = cv2.threshold(image_data, prob, 255, cv2.THRESH_BINARY)

        # Skeletonize the thresholded image
        skeleton_image = skeletonize(thresh1 > 0).astype(np.uint8)  # Convert to binary for skeletonize
        # Plotting the images side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_data, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(thresh1, cmap='gray')
        axes[1].set_title('Thresholded Image')
        axes[1].axis('off')

        axes[2].imshow(skeleton_image, cmap='gray')
        axes[2].set_title('Skeletonized Image')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        #save image
        skeleton_image = skeleton_image.astype(np.uint16)
            # Update the FITS file with the cleaned image
        hdul[0].data = skeleton_image
        hdul.flush()

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter

def cleanImage(fits_path, original_data, radius, threshold = 15000):
    """
    Cleans the skeletonized image by filtering pixels based on the sum of values
    within a specified radius in the original image.

    Parameters:
        fits_path (str): Path to the FITS file containing the skeletonized image.
        original_data (numpy.ndarray): The original image array for reference.
        radius (int): Radius for the region to search around each pixel.
        threshold (float): Threshold for the sum of pixel values in the radius.

    Returns:
        None: The cleaned skeletonized image is saved to the same FITS file.
    """
    # Open the FITS file and load the skeletonized image data
    with fits.open(fits_path, mode="update") as hdul:
        original_header = hdul[0].header
        skeletonized_image = np.array(hdul[0].data)
        
        # Normalize the original image if needed (same scale as skeletonized_image)
        original_data = original_data / np.max(original_data)
        
        # Calculate the sum of pixel values within the specified radius
        kernel_size = 2 * radius + 1
        local_sum = uniform_filter(original_data, size=kernel_size, mode='constant')
        
        # Apply the threshold condition to the skeletonized image
        clean_image = np.copy(skeletonized_image)
        mask = local_sum < threshold
        clean_image[mask] = 0
        
        # Update the FITS file with the cleaned image
        hdul[0].data = clean_image
        hdul.flush()

if __name__ == "__main__":
    print("hi jake")
    skel_path = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\ngc0628\Composite\ngc0628_F770W_starsub_anchored_CDDss0064pc_arcsinh0p1.fits_Composite.fits"
    threshSkel(skel_path, skel_path)