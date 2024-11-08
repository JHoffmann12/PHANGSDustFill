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


#Folder with JWST images
galaxy_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking"

#iterate through JWST files
for galaxy_folder in os.listdir(galaxy_dir):
    if(galaxy_folder == "GalaxySim"):
        galaxy_folder = os.path.join(galaxy_dir, galaxy_folder)
        for fits_file in os.listdir(galaxy_folder):
            fits_path = os.path.join(galaxy_folder, fits_file)
            if(fits_file.endswith(".fits")):
                #Extract image and header
                with fits.open(fits_path) as hdul:
                    image_data = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU
                    image_data = np.nan_to_num(image_data, nan=0.0)  # Replace NaNs with 0
                    original_header = hdul[0].header

                #Get thresholded image for comparison
                thresh_out_path = fr"{galaxy_folder}\Thresholds\{fits_file}_to_Thresh.fits"
                identify.ThresholdSkel(image_data, original_header, thresh_out_path) #funky...need to fix later 

                #Determine Params
                scale_factor, img_scale = identify.determineParams(fits_file)

                #set thresholding for masking blank regions
                # threshold = .001 # for real image
                # iters = 6 #for real image
                # kernel_size = 11

                threshold = .05 # for simulated image
                iters = 6 # for simulated image
                kernel_size = 9 #for simulated image

                #mask blank regions
                dilated_data, mask = identify.dilateBlankRegions(image_data, threshold, iters, kernel_size) #increase iters or kernel for more aggressive dilation of masked region

                #Determine pixel pc size from header
                distance_pc = identify.getDistance(fits_file) # Distance to the object in parsecs 
             #   scalepix = .53328 * distance_pc #.53328 based on MIRI specs, should multiply by 2* scale factor but boxes are too big and have too many nans
                scalepix = 6.4 #for simulated image
                print(f"Pixel size: {scalepix:.2f} parsecs")

                #subtract bkg and divide by RMS
                fits_BkgSub_out_path =  fr'{galaxy_folder}\BkgSubDivRMS\{fits_file}_divRMS.fits'
                divRMS = identify.bkgSub(image_data, mask, scalepix, original_header, fits_BkgSub_out_path)

                #Block the image
                fits_block_out_path =  fr'{galaxy_folder}\BlockedFits\{fits_file}_Blocked.fits'
                save_png_path = fr"{galaxy_folder}\BlockedPng\{fits_file}_Blocked.png"

                if(scale_factor != 0):
                    blocked_data, new_header = identify.resample_fits(fits_BkgSub_out_path, fits_block_out_path, scale_factor, save_png_path )
                else: 
                    hdu = fits.PrimaryHDU(divRMS, original_header) #save bkg subtracted data
                    hdu.writeto(fits_block_out_path, overwrite=True)
                    divRMS = divRMS.astype(np.uint16)
                    cv2.imwrite(save_png_path, divRMS)
                    blocked_data = image_data
                    new_header  = original_header

                #Run SOAX
                input_image = save_png_path
                print(input_image)
                batch = r"C:\Users\HP\Downloads\batch_soax_v3.7.0.exe"
                output_dir = fr"{galaxy_folder}\SOAXOutput\{img_scale}"
                print(output_dir)
                parameter_file = fr"C:\Users\HP\Documents\JHU_Academics\Research\Params\best_param1_16pc.txt"

                assert(os.path.isdir(output_dir))
                assert( os.path.isfile(parameter_file))
                assert(os.path.isfile(input_image))

                print("starting Soax")
                cmdString = f'"{batch}" soax -i "{input_image}" -p "{parameter_file}" -s "{output_dir}" --ridge 0.005 0.01 0.045 --stretch 3.0 0.5 4'
                subprocess.run(cmdString, shell=True)
                print(f"Complete Soax on {fits_file}")

                #input file path is txt file, output file path is fits from SOAX
                print("txt to FITS")
                for result_file in os.listdir(f'{galaxy_folder}\SOAXOutput\{img_scale}'):
                    if(result_file.endswith('.txt')):
                        csv_file_path = fr"{galaxy_folder}\SOAXOutput\{img_scale}\{result_file}_to_CSV.csv"
                        fits_out_path = fr"{galaxy_folder}\SOAXOutput\{img_scale}\{result_file}_to_FITS.fits"
                        base_result_file = os.path.splitext(result_file)[0]  # removes the .txt extension
                        skel_path = fr"{galaxy_folder}\SoaxSkel\{base_result_file}_to_Skel.fits"
                        result_file = fr"{galaxy_folder}\SOAXOutput\{img_scale}\{result_file}"
                        filament_dict, soax_data = identify.txtToFilaments(result_file, csv_file_path, blocked_data, fits_out_path, scale_factor, new_header) #use new_header for WCS and blocked_data dimesnions
                        #skeletonize Soax? 
                        if(isinstance(soax_data, np.ndarray) and soax_data.size > 0) and (scale_factor != 0): 
                            print("making skeleton")
                            skel_data = skeletonize(soax_data)
                            skel_data = skel_data.astype(np.uint16)
                            hdu = fits.PrimaryHDU(skel_data, header=new_header)
                            hdu.writeto(skel_path, overwrite=True) 

                #generate composite/probability image
                directory = fr"{galaxy_folder}\SOAXOutput\{img_scale}"
                output_name = fr"{fits_file}_Composite"
                output_directory = fr"{galaxy_folder}\Composite"
                common_string = fits_file
                identify.create_composite_image(directory, output_name, output_directory, common_string)


    #coordinate issue, file with no upscaling, probability including BKg subtracted intensity, summary of process, Filament alignment, 
    # conitnuity after removing minorty pixels, run for simulation

