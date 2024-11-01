#import necessary libraries and Modules
from skimage import data, color, io
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import cv2
import numpy as np
import pandas as pd
import os
import copy
import subprocess
import csv
import math
import importlib
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
import IdentificationFuncs as identify


# #Folder with JWST images
# galaxy_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking"

# #iterate through JWST files
# for galaxy_folder in os.listdir(galaxy_dir):
#     galaxy_folder = os.path.join(galaxy_dir, galaxy_folder)
#     for fits_file in os.listdir(galaxy_folder):

#         fits_path = os.path.join(galaxy_folder, fits_file)
#         if(fits_file.endswith(".fits")):
#             #Extract image and header
#             with fits.open(fits_path) as hdul:
#                 image_data = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU
#                 image_data = np.nan_to_num(image_data, nan=0.0)  # Replace NaNs with 0
#                 original_header = hdul[0].header
#                 cdelt1_deg = original_header['CDELT1']  # Pixel scale in degrees

#             #Determine Params
#             scale_factor = identify.determineParams(fits_file)

#             #set thresholding for masking blank regions
#             threshold = .001 # for real image
#             iters = 6 #for real image
#             kernel_size = 11

#             #mask blank regions
#             dilated_data, mask = identify.dilateBlankRegions(image_data, threshold, iters, kernel_size) #increase iters or kernel for more aggressive dilation of masked region

#             #Determine pixel pc size from header
#             distance_pc = 9.6e6  # Distance to the object in parsecs 
#             # Convert pixel scale from degrees to radians
#             cdelt1_rad = cdelt1_deg * (np.pi / 180)
#             # Calculate pixel size in parsecs
#             scalepix = cdelt1_rad * distance_pc
#             print(f"Pixel size: {scalepix:.2f} parsecs")

#             #subtract bkg and divide by RMS
#             fits_BkgSub_out_path =  fr'{galaxy_folder}\BkgSubDivRMS\{fits_file}_divRMS.fits'
#             divRMS = identify.bkgSub(image_data, mask, scalepix, original_header, fits_BkgSub_out_path)

#             #Block the image
#             fits_block_out_path =  fr'{galaxy_folder}\BlockedFits\{fits_file}_Blocked.fits'
#             save_png_path = fr"{galaxy_folder}\BlockedPng\{fits_file}_Blocked.png"

#             if(scale_factor != 0):
#                 blocked_data = identify.resample_fits(fits_BkgSub_out_path, fits_block_out_path, scale_factor, save_png_path )
#             else: 
#                 hdu = fits.PrimaryHDU(divRMS, original_header) #save bkg subtracted data
#                 hdu.writeto(fits_block_out_path, overwrite=True)
#                 divRMS = divRMS.astype(np.uint16)
#                 cv2.imwrite(save_png_path, divRMS)
#                 blocked_data = image_data

#             #Get thresholded image for comparison
#             thresh_out_path = fr"{galaxy_folder}\Thresholds\{fits_file}_to_Thresh.fits"
#             identify.ThresholdSkel(image_data, original_header, thresh_out_path)

#             #Run SOAX
#             input_image = save_png_path
#             print(input_image)
#             batch = r"C:\Users\HP\Downloads\batch_soax_v3.7.0.exe"
#             output_dir = fr"{galaxy_folder}\SOAXOutput"
#             print(output_dir)
#             parameter_file = fr"C:\Users\HP\Documents\JHU_Academics\Research\Params\best_param1_16pc.txt"

#             assert(os.path.isdir(output_dir))
#             assert( os.path.isfile(parameter_file))
#             assert(os.path.isfile(input_image))

#             print("starting Soax")
#             cmdString = f'"{batch}" soax -i "{input_image}" -p "{parameter_file}" -s "{output_dir}" --ridge 0.005 0.01 0.045 --stretch 3.0 0.5 4'
#             subprocess.run(cmdString, shell=True)
#             print(f"Complete Soax on {fits_file}")

#             #input file path is txt file, output file path is fits from SOAX
#             print("txt to FITS")
#             for result_file in os.listdir(f'{galaxy_folder}\SOAXOutput'):
#                 if(result_file.endswith('.txt')):
#                     csv_file_path = fr"{galaxy_folder}\SOAXOutput\{result_file}_to_CSV.csv"
#                     fits_out_path = fr"{galaxy_folder}\SOAXOutput\{result_file}_to_FITS.fits"
#                     result_file = fr"{galaxy_folder}\SOAXOutput\{result_file}"
#                     filament_dict = identify.txtToFilaments(result_file, csv_file_path, blocked_data, fits_out_path)
        

galaxy_folder = r'C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking\ngc0628'
fits_file = "ngc0628_F770W_starsub_anchored_CDDss0008pc_arcsinh0p1.fits"
#Blur and skeletonize all Soax images, check for improvement
output_directory = fr'{galaxy_folder}\BlurSkel'
for result_file in os.listdir(fr'{galaxy_folder}\SOAXOutput'):
    if(result_file.endswith('.fits')):
        output_name = fr"{result_file}_SkelBlur"
        result_file = fr"{galaxy_folder}\SOAXOutput\{result_file}"
        identify.BlurSkel(result_file, output_name, output_directory)

#generate inferno color map composite/probability image
directory = fr"{galaxy_folder}\SOAXOutput"
output_name = fr"{fits_file}_Composite"
output_directory = fr"{galaxy_folder}\Composite"
common_string = fits_file
identify.create_composite_image(directory, output_name, output_directory, common_string, 33) #33 means a pixel is high in 20% of images to be present in composite

    