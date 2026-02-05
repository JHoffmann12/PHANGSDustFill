#Filament map class to produce skeletonized filament maps of astrophysical images

#imports 
from astropy.nddata.utils import Cutout2D
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table, QTable
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from pathlib import Path
import os
from os import path
from glob import glob
import sys
import re
import csv
import math
import random
import copy
import subprocess
import threading
import time
import timeit
import importlib
from matplotlib import scale
from requests import get
from scipy import ndimage

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2
import imageio

from scipy.ndimage import gaussian_filter, zoom, uniform_filter, convolve
from scipy.stats import kde, lognorm, scoreatpercentile

from skimage import measure, color, data, exposure
from skimage.morphology import skeletonize, disk, ball, binary_dilation
from skimage.filters import meijering, sato, frangi, hessian, rank
from skimage.util import img_as_ubyte
from skimage.util.dtype import dtype_range
from skimage.restoration import inpaint

from reproject import reproject_exact, reproject_adaptive, reproject_interp

from photutils.background import Background2D, MedianBackground, LocalBackground, MMMBackground
from photutils.psf import CircularGaussianPRF, make_psf_model_image, PSFPhotometry, SourceGrouper
from photutils.segmentation import detect_sources, deblend_sources

import AnalysisFuncs as AF
import sys


matplotlib.use('Agg')


class FilamentMap:

    def __init__(self,  scalepix, base_dir, label_folder_path, fits_file, label, param_file_path, flatten_perc, min_intensity):

        '''
        Filament Map Constructor.

        Parameters:
        - scalepix (float): parcecs per pixel
        - base_dir (str): path to FilPHANGS base directory
        - label_folder_path (str): path to image subfolder
        - fits_file (str): path to fits file in CDD folder
        - label (str): Image label
        - param_file_path (str): path to the soax parameter file
        - flatten_perc (float): percentage for arctan transform

        '''

        CDD_path = os.path.join(label_folder_path, "CDD")
        fits_path = os.path.join(CDD_path, fits_file)

        self.BaseDir = base_dir
        self.Label = label
        Scale = self._getScale(fits_file)
        self.Scale = Scale
        self.FitsFile = fits_file


        with fits.open(fits_path, ignore_missing=True) as hdul:
            OrigData = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU

            OrigData = self.preprocessImage(OrigData, fits_file, flatten_percent = flatten_perc)

            OrigData = np.nan_to_num(OrigData, nan=0.0)  # Replace NaNs with 0
            OrigHeader = hdul[0].header


        #get original image
        input_dir = Path(f"{base_dir}/OriginalImages")
        for file_name in os.listdir(input_dir):
            file_name_without_extension = os.path.splitext(file_name)[0]
            if file_name_without_extension in fits_file:
                input_path = os.path.join(input_dir, file_name)

        with fits.open(input_path,ignore_missing=True) as hdu:
            hdu.info()
        try:
            img=hdu[0].data
            OrigData[img < min_intensity] = 0

        except:
            OrigData = OrigData
            OrigData[OrigData < min_intensity] = 0
    
        self.ProbabilityMap = np.zeros_like(OrigData)
        self.BkgSubDivRMSMap = np.zeros_like(OrigData)
        self.OrigHeader = OrigHeader
        self.OrigData = OrigData
        self.BlockHeader = OrigHeader.copy() #updated later to relfect blocking
        self.Composite = np.zeros_like(OrigData)
        self.Scalepix = scalepix
        fits_file = os.path.splitext(fits_file)[0]  # removes the .fits extension
        self.BlockFactor = self._getBlockFactor()
        self.BlankRegionMask = self._getBlankRegionMask()

        if(self.BlockFactor != 0):
            self.BlockData = np.zeros((int(self.OrigData.shape[0] / self.BlockFactor), int(self.OrigData.shape[1] / self.BlockFactor))) 
        else: 
            self.BlockData = np.zeros_like(self.OrigData)

        self.IntensityMap = np.zeros_like(self.BlockData)
        self.NoiseMap = np.zeros_like(self.BlockData)
        self.ParamFile = param_file_path



    def preprocessImage(self, image, fits_file, skip_flatten=False, flatten_percent=None):
        
        '''
        Preprocess and flatten the image before running the masking routine.

        Parameters:
        - skip_flatten (bool): optional. Skip the flattening step and use the original image to construct the mask. Default is False.
        - flatten_percent (int) : optional. The percentile of the data (0-100) to set the normalization of the arctan transform. By default, 
            a log-normal distribution is fit and the threshold is set to :math:`\mu + 2\sigma`. If the data contains regions of a much higher 
            intensity than the mean, it is recommended this be set >95 percentile.

        '''

        if 'Extinction' in fits_file: 
            print('Equalizing image')
            Scale = self._getScale(fits_file)
            Scale = Scale.replace("pc", '')
            Scale = int(Scale)
            maskfoot=(image!=image[0,0])

            range = 65535/2

            imagemax=np.max(image)
            imagemin=np.min(image)
            image=range-((image-imagemin)*range/(imagemax-imagemin))
            image[image>range]= range
            image[image<0.]=0.
            image=image.astype('uint16')

            # Equalization
            radius = int(Scale * 2 + 1) #Make box 2% of smaller image dimension...appears to work well
            #make the radius 2 * extracted_scale
            footprint = disk(radius)  # disk of radius for local hist.eq
            processed_img = rank.equalize(image, footprint, mask=maskfoot)
            outhdu = fits.PrimaryHDU(data=processed_img)
            out_path = Path(f"{self.BaseDir}/{self.Label}/BkgSubDivRMS/{self.FitsFile}_BkgSubDivRMS_Eq.fits") 
            outhdu.writeto(out_path ,overwrite=True)
        else:
            if skip_flatten:
                flatten_threshold = None
                flat_img = image
            else:
                # Make flattened image
                if flatten_percent is None:
                    # Fit to a log-normal distribution
                    fit_vals = lognorm.fit(image[~np.isnan(image)])  # Fit only non-NaN values
                    median = lognorm.median(*fit_vals)
                    std = lognorm.std(*fit_vals)
                    thresh_val = median + 2 * std
                else:
                    # Use the specified percentile to calculate threshold
                    thresh_val = np.percentile(image[~np.isnan(image)], flatten_percent)

                # Apply the arctan transform, ensuring that we are dividing by the threshold value
                processed_img = thresh_val * np.arctan(image / thresh_val)
        
        return processed_img
            
    def setBlockFactor(self, bf):
        self.BlockFactor = bf
    

    def _getScale(self, fits_file):
        
        """
        get the scale associated with an image. Scales may need to be added. 

        """

        scales = ["1024pc", "512pc", "256pc", "128pc", "64pc", "32pc", "16pc", "8pc", "4pc", "2pc", "1pc", ".5pc", ".25pc", ".125pc", ".0625pc", ".03125pc"]
        
        for scale in scales:
            if scale in fits_file:
                return scale

        print("Invalid FITS file: Scale not recognized. Please add scale to _getScale method.")
        return None


    def _getBlockFactor(self):

        """
        get the block factor associated with an image

        """

        file_name = self.FitsFile
        folder_path = os.path.join(self.BaseDir, self.Label)
        folder_path = os.path.join(folder_path, "CDD")
        
        # Pattern to match valid powers of two
        power_of_two_pattern = re.compile(r'(?<!\d)0*(.03125|.0625|.125|.25|.5|1|2|4|8|16|32|64|128|256|512|1024)(?!\d)') #2^-5 to 2^10
        powers_of_two = []

        # Extract powers of two from filenames in the folder
        for f in os.listdir(folder_path):
            match = power_of_two_pattern.search(f)
            if match:
                powers_of_two.append(float(match.group(0)))

        if not powers_of_two:
            raise ValueError("No valid power of two found in file names.")
        
        # Sort powers of two in ascending order
        sorted_powers = sorted(set(powers_of_two))

        # Extract the power of two from the current file name
        current_match = power_of_two_pattern.search(file_name)
        if not current_match:
            raise ValueError(f"File name '{file_name}' does not contain a valid power of two.")
        
        current_power = float(current_match.group(0))

        # Ensure the current power exists in the list
        if current_power not in sorted_powers:
            raise ValueError(f"Power of two {current_power} from '{file_name}' not found in folder.")

        # Determine the rank (index) of the current power of two in the sorted list
        rank = sorted_powers.index(current_power)
        block_factor = 2 ** rank
        if(block_factor <= 1):
            block_factor = 0
        return block_factor
    

    def _getBlankRegionMask(self):
        """
        Generate a mask to hide large vacant regions as well as the image border.
        Returns: 
        - a mask, where true indicates a region to be masked.
        """
        threshold = 10**-20  # pixels below this value will be seeds for mask
        data_to_mask = self.OrigData

        # Step 1: Create initial mask (True = masked, False = keep)
        mask_copy = copy.deepcopy(data_to_mask)
        mask_copy[mask_copy > threshold] = 255
        mask_copy = mask_copy.astype(np.uint8)
        
        # Step 2: First dilation (aggressive)
        kernel_size_1 = 10 # Smaller kernel = more expansion of masked region
        kernel_1 = np.ones((kernel_size_1, kernel_size_1), np.uint8)
        dilated_image = cv2.dilate(mask_copy, kernel_1, iterations=5)  # More iterations = stronger expansion

        # Step 3: Convert to boolean mask (True = masked)
        copy_image = copy.deepcopy(data_to_mask).astype(np.float32)
        copy_image[dilated_image < threshold] = np.nan
        mask = np.isnan(copy_image)

        # Step 4: Further expand the True (masked) regions
        binary_mask = np.zeros_like(copy_image, dtype=np.uint8)
        binary_mask[mask] = 255  # White = masked (True) regions to expand

        kernel_size_2 = 25  # Even larger kernel for final expansion
        kernel_2 = np.ones((kernel_size_2, kernel_size_2), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel_2, iterations=3)  # Strong final expansion

        # Step 5: Final mask (True = masked, False = keep)
        final_mask = dilated_mask == 255  # Convert back to boolean

        # Visualization (optional)
        plt.imshow(np.uint8(final_mask) * 255, cmap='gray')
        plt.title(f"Mask of {self.Label} at {self.Scale}")
        plt.savefig(Path(f"{self.BaseDir}/Figures/Mask_{self.Label}_{self.Scale}.png"))
        plt.close()

        return final_mask


    def setBkgSubDivRMS(self, noise_min):

        """
        Subtract the estimated background and divide by RMS noise to improve SOAX performance

        Parameters: 
        - noise_min (float): minimum acceptable noise
        - write_fits (bool): whether or not image should be saved. 

        """
                
        try:
            mask = self.BlankRegionMask
            data = copy.copy(self.BlockData.astype(np.float64)) #photutils should take float64

            #subtract bkg
            bkg_estimator = MedianBackground()

            if(self.BlockFactor == 0):
                blockFactor = 1
            else: 
                blockFactor = self.BlockFactor 

            #set box size
            box_size = round(10.*self.Scalepix/(2.*blockFactor))*2+1
            if(box_size < .02* np.min((np.shape(data)[0], np.shape(data)[1]))):
                print("Correcting box size")
                box_size = int(.02*np.min((np.shape(data)[0], np.shape(data)[1]))) * 2 + 1 #Make box 2% of smaller image dimension.Appears to work well

            bkg = Background2D(data, box_size=box_size, coverage_mask = mask, exclude_percentile = 10, filter_size=(3,3), bkg_estimator=bkg_estimator) #Very different RMS with mask. Minimum noise is MUCH larger. 

            data -= bkg.background #subtract bkg

            data[data < 0] = 0 #Elimate neg values sinc this is over estimating the background. 

            noise = bkg.background_rms #bkg sub/RMS map

            noise[noise < noise_min] = noise_min #replace unphysical and absent noise with 10^-3 just to avoid division by zero

            divRMS = data/noise

            divRMS[(mask == 1)] = 0 #masked regions are zero
            self.BkgSubDivRMSMap = divRMS
            self.NoiseMap = noise

        except ValueError:
            print("Error: Majrity Black pixels, cannot use photutils to enhance image")
            self.BkgSubDivRMSMap = self.BlockData


    def scaleBkgSubDivRMSMap(self,  write_fits):

        """
        Scale the background subtracted and RMS divided image to be 16 bits

        Parameters: 
        - write_fits (bool): whether or not image should be saved. 

        """

        # Normalize the image data
        topval = 20
        image_data = self.BkgSubDivRMSMap

        if topval == 0:
            raise ValueError("Top value for scaling is zero, cannot divide by zero.")
        
        # Scale the image data
        self.BkgSubDivRMSMap = np.array(image_data) * 65535 / topval
        self.BkgSubDivRMSMap[self.BkgSubDivRMSMap > 65535] = 65535

        # Save a PNG for SOAX
        save_png_path = Path(f"{self.BaseDir}/{self.Label}/BlockedPng/{self.FitsFile}_Blocked.png")
        pngData = self.BkgSubDivRMSMap.astype(np.uint16)

        cv2.imwrite(save_png_path, pngData)

        if(write_fits):
            print('saving bkg sub as fits')
            out_path = Path(f"{self.BaseDir}/{self.Label}/BkgSubDivRMS/{self.FitsFile}_BkgSubDivRMS.fits")
            hdu = fits.PrimaryHDU(self.BkgSubDivRMSMap, header=self.BlockHeader)
            hdu.writeto(out_path, overwrite=True)


    def setBlockData(self):

        """
        Fix the blocked header and block the image
        """
            
        if(self.BlockFactor !=0):

            #fix the blocked header
            print(f"Block factor: {self.BlockFactor} and Scale: {self.Scale}")
            try: 
                self.BlockHeader['CDELT1'] = (self.OrigHeader['CDELT1']) * self.BlockFactor
                self.BlockHeader['CDELT2'] = (self.OrigHeader['CDELT2']) * self.BlockFactor
            except KeyError:
                    # Extract the CD values
                CD1_1 = self.BlockHeader['CD1_1']
                CD1_2 = self.BlockHeader['CD1_2']
                CD2_1 = self.BlockHeader['CD2_1']
                CD2_2 = self.BlockHeader['CD2_2']
                
                
                # Add the CDELT values to the header
                self.BlockHeader['CDELT1'] = CD1_1 * self.BlockFactor
                self.BlockHeader['CDELT2'] = CD2_2 * self.BlockFactor
                self.BlockHeader['CD1_1'] = CD1_1 * self.BlockFactor
                self.BlockHeader['CD2_2'] = CD2_2 * self.BlockFactor

            self.BlockHeader['CRPIX1'] = (self.OrigHeader['CRPIX1']) / self.BlockFactor 
            self.BlockHeader['CRPIX2'] = (self.OrigHeader['CRPIX2']) / self.BlockFactor 


            self.BlockData  = self.reprojectWrapper(self.OrigData, self.OrigHeader, self.BlockHeader, self.BlockData)
            self.BlankRegionMask = self.reprojectWrapper(self.BlankRegionMask, self.OrigHeader,self.BlockHeader, self.BlockData)

            assert(np.shape(self.BlankRegionMask)==np.shape(self.BlockData))
            self.BlankRegionMask = self.BlankRegionMask == 1 #convert back to boolean mask, True indicates pixels to mask
        else:
            self.BlockData = self.OrigData

        assert(np.sum(self.BlockData) !=0)




    def _cropNanBorder(self, image, expected_shape):
    
        """
        Crop the nan border away such that the image fits the expected shape

        Parameters: 
        -image (float): The data to crop. 2D array. 
        -expected_shape (int): the expected shape to crop the data into. 2 element tuple. 

        Returns: 
        - returns the cropped image
        """

        # Create a mask of non-NaN values
        image = np.array(image, dtype=float)  # converts numbers; non-convertible become nan
        mask = ~np.isnan(image)

        # Find the rows and columns with at least one non-NaN value
        non_nan_rows = np.where(mask.any(axis=1))[0]
        non_nan_cols = np.where(mask.any(axis=0))[0]

        # Use the min and max of these indices to slice the image
        cropped_image = image[non_nan_rows.min():non_nan_rows.max() + 1, non_nan_cols.min():non_nan_cols.max() + 1]
        
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
    

    def runSoaxThreads(self, min_snake_length_ss, min_fg_int, batch_path):

        """
        Create 5 threads to run soax in 5 pairs of 2

        Parameters:
        - min_snake_length_ss (float): minimum length for a filament in the shortest scale (ss) in pixels such that soax keeps it
        - min_fg_int (float): minim foreground intensity on a scale of 65535 for soax to intialize a snake
        - batch_path (str): Path to the soax batch file
        """

        #Soax params found through trial and error
        stretch_start = 1.75
        stretch_stop = 2.5


        #adjust soax length param to account for blocking
        new_length = round(min_snake_length_ss - ((math.sqrt(self.BlockFactor))*4))
        self.updateMinimumSnakeLength(new_length, min_fg_int)
        print('Starting Threads')


        #begin 5 threads to speed up soax
        threads = []

        for i in range(5):
            t = threading.Thread(target=self.runSoax, kwargs={
                "ridge_start": .02375 + i*0.0075, 
                "ridge_stop": .03 + i*0.0075, 
                "stretch_start": stretch_start, 
                "stretch_stop": stretch_stop, 
                "batch_path": batch_path})
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        print("Threads done")

        

    def runSoax(self, ridge_start, ridge_stop, stretch_start, stretch_stop, batch_path):

        """
        Runs soax batch file and then turns the soax .txt file into a Fits file in the dimensions of the original image. 
        Lines are interpolated between points in the blocked image to transfrom back into the original image size. 

        Parameters:
        - ridge_start (float): starting ridge thresholf value for soax run
        - ridge_stop(float): stopping ridge threshold value for soax run
        - stretch_start (float): starting stretch factor value for soax run
        - stretch_stop (float): stopping stretch factor value for soax run
        - batch_path (str): path to the soax batch file 

        """


        input_image = Path(f"{self.BaseDir}/{self.Label}/BlockedPng/{self.FitsFile}_Blocked.png")
        batch = batch_path
        output_dir = Path(f"{self.BaseDir}/{self.Label}/SOAXOutput/{self.Scale}")

        try: 
            assert(os.path.isdir(output_dir))
            assert( os.path.isfile(self.ParamFile))
            assert(os.path.isfile(input_image))
        except AssertionError as e: 
            print('Error: Assertions failed, a faulty path exists!')
            
        print("starting Soax")
        cmdString = f'"{batch}" soax -i "{input_image}" -p "{self.ParamFile}" -s "{output_dir}" --ridge {ridge_start} 0.0075 {ridge_stop} --stretch {stretch_start} 0.5 {stretch_stop}' 
        with open(os.devnull, 'w') as devnull:
            subprocess.run(cmdString, shell=True, stdout=devnull, stderr=devnull) #supress output from threads because its garbage

        self._convertSoaxToFits(output_dir, ridge_start) #ridge_start can be used to specify the two SOAX files created by one process

        print("Soax converted to Fits, Success!")


    def updateMinimumSnakeLength(self, new_length, min_fg_int):

        """
        Updates the 'minimum-snake-length' field in the given file with a new value.

        Parameters:
        - file_path (str): Path to the .txt file to be updated.
        - new_length (int or float): New value for the 'minimum-snake-length' field.
        """

        file_path = self.ParamFile

        # Read the file contents
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify the relevant line and store updated lines
        updated_lines = []
        for line in lines:
            if line.startswith('minimum-snake-length'):
                updated_lines.append(f'minimum-snake-length\t{new_length}\n')
            elif line.startswith('minimum-foreground'):
                updated_lines.append(f'minimum-foreground\t{min_fg_int}\n')
            else:
                updated_lines.append(line)

        # Write the modified content back to the file
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)



    def _convertSoaxToFits(self, output_dir, ridge_start): 

        """
        After the soax .txt file is produced, convert them to a .Fits file. This function manages a loop, calling the _txtToFilaments function to do the heavy lifting. 

        Parameters:
        - output_dir (str): Directory to output the .Fits files to
        - ridge_start (float): starting ridge threshold for the soax run. Present in the .txt file name and allows threads to avoid one another.
        """

        for result_file in os.listdir(output_dir):
            if(result_file.endswith('.txt') and str(ridge_start) in result_file):
                result_file_base = os.path.splitext(result_file)[0]  # removes the .txt extension

                interpolate_path = Path(f"{self.BaseDir}/{self.Label}/SOAXOutput/{self.Scale}/{result_file_base}.fits")
                result_file = Path(f"{self.BaseDir}/{self.Label}/SOAXOutput/{self.Scale}/{result_file}")
                
                self._txtToFilaments(result_file, interpolate_path) #use new_header for WCS and blocked_data dimesnions


    def _txtToFilaments(self, result_file, interpolate_path):

        """
        Parse through the Soax .txt file and create the .Fits file

        Parameters:
        - result_file (str): soax text file path
        - interpolate_path (str): path to save the Fits file at
        """
                
        expected_header = "s p x y z fg_int bg_int"

        print(f' Begenning Soax txt to Fits File conversion on {result_file}')
        assert(os.path.isfile(result_file))
        
        # Read the content of the input file
        with open(result_file, 'r') as file:
            lines = file.readlines()

        # Find the index where the header starts
        header_start_index = None
        stopper_index = -1  # sometimes "[" isn't present
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
        
        # Extract header and data, after identifying the header start index
        header_and_data = lines[header_start_index + 1:stopper_index]  # Skip the actual header line

        # Assuming header_and_data is a list of strings, where each string represents a row
        # Split columns by whitespace
        data = [line.split() for line in header_and_data]

        # Convert data into a DataFrame, explicitly specifying the correct columns
        df = pd.DataFrame(data, columns=expected_header.split())

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
                filament_dict[s_value].append((round(float(row['x'])), round(float(row['y']))))

        self._interpolate(filament_dict, interpolate_path)


    def _interpolate(self, dict, path):

        """
        Interpolate between the blocked data and the reprojected data to create a Fits file of the original image dimensions. 

        Parameters:
        - dict (int, list): Contains a filament ID and list of coordinates corresponding to each pixel on the filament
        - path (str): path to save the final Fits file to
        """

        wcs_orig = WCS(self.OrigHeader)
        wcs_blocked = WCS(self.BlockHeader)
        final_image = np.zeros_like(self.OrigData)

        filaments = dict.values()
        for filament in filaments:
            x_coords = []
            y_coords = []

            for pixel in filament:
                x_coords.append(pixel[0])
                y_coords.append(pixel[1])

            world_coords = wcs_blocked.pixel_to_world(x_coords, y_coords)
            x_original, y_original = wcs_orig.world_to_pixel(world_coords)
            coordinates = [(x, y) for x, y in zip(x_original, y_original) if (not (np.isnan(x)) and (not np.isnan(y)))]

            new_image = self._connectPointsSequential(coordinates, np.shape(self.OrigData))
            final_image+=new_image

        hdu = fits.PrimaryHDU(final_image, header = self.OrigHeader)
        hdu.writeto(path, overwrite=True)


    def _connectPointsSequential(self, points, image_shape):

        """
        Connect lines between reprojected points

        Parameters:
        - points (list of float tuples): list of coordinates to connect
        - image_shape : Dimensions of final image

        Returns: 
            An all black image except for the input points and lines between them, which are white. 
        """

        points = [(int(x), int(y)) for x, y in points]
        output_array = np.zeros(image_shape, dtype=np.uint8)

        # Draw lines between consecutive points
        for i in range(len(points) - 1):
            x1, y1 = (points[i][0], points[i][1])
            x2, y2 = (points[i+1][0], points[i+1][1])
            cv2.line(output_array, (x1, y1), (x2, y2), 1, thickness= 1)

        return output_array
    

    
    def createComposite(self, write_fits = False):
        """
        Stack all 10 of the interpolated Fits files from soax into a composite

        Parameters:
        - write_fits (bool): Whether or not the stacked composite image should be saved
        """

        output_directory = Path(f"{self.BaseDir}/{self.Label}/Composites")
        directory = Path(f"{self.BaseDir}/{self.Label}/SOAXOutput/{self.Scale}")
        output_name = Path(f"{self.FitsFile}_Composites")
        common_string = self.FitsFile
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Find FITS files based on the common_string
        file_pattern = os.path.join(directory, f"*{common_string}*.fits")
        fits_files = glob(file_pattern)

        # Read each FITS file and store the data
        images = []
        header = None

        for file in fits_files:
            with fits.open(file, ignore_missing=True) as hdul:
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

        max_presence = len(images)
        print(f"Total number of images: {max_presence}")

        # Normalize composite data to the range [0, 255] for grayscale representation
        composite_data = (composite_data / max_presence * 255).astype(np.uint8)

        self.Composite = composite_data
        self.ProbabilityMap = composite_data

        # Save the grayscale composite as FITS? 
        if write_fits: 
            output_fits_path = os.path.join(output_directory, str(output_name) + ".fits")
            hdu = fits.PrimaryHDU(data=composite_data, header=header)
            hdu.writeto(output_fits_path, overwrite=True)
            print(f"Composite image saved as FITS")




    def setIntensityMap(self, orig = True):
        if(orig):
            self.IntensityMap[self.Composite !=0] = 255*self.OrigData[self.Composite!=0]
        else: 
            temp = self.reprojectWrapper(self.ProbabilityMap, self.OrigHeader, self.BlockHeader, self.BlockData)
            self.IntensityMap[temp !=0] = self.BkgSubDivRMSMap[temp!=0]



    def getProbIntensityPlot(self, use_orig_img =True, write_fig =False):
        """
        Create a series of box plots to compare trends in filament probability vs intensity

        Parameters:
        - use_orig_img (bool): Whether or not the background subtracted and RMS divided intensities should be used or the original decomposed image. 
        - write_fig (bool): Whether or not the plot should be saved. 
        """
                
        if use_orig_img:
            probability_flat = self.ProbabilityMap.flatten()
            self.setIntensityMap(orig = True)
        else: 
            self.setIntensityMap(orig = False)
            temp = self.reprojectWrapper(self.ProbabilityMap, self.OrigHeader, self.BlockHeader, self.BlockData)
            probability_flat = temp.flatten()

        # Flatten the images
        probability_flat = np.round((100 / np.max(probability_flat)) * probability_flat).astype(int)
        intensity_flat = self.IntensityMap.flatten()

        # Remove entries where the probability equals zero
        valid_indices = probability_flat != 0
        probability_flat = probability_flat[valid_indices]
        intensity_flat = intensity_flat[valid_indices]

        # Group probabilities into 5% bins
        binned_probabilities = (probability_flat // 5) * 5  # Group into bins of 5%
        unique_bins = np.unique(binned_probabilities)

        # Group intensity values by probability bin
        grouped_intensities = [intensity_flat[binned_probabilities == prob] for prob in unique_bins]

        # Count values in each group
        counts = [len(group) for group in grouped_intensities]

        # Compute the total number of elements across all box plots
        total_elements = sum(counts)

        # Create box plots
        plt.figure(figsize=(10, 6))
        plt.boxplot(grouped_intensities, labels=unique_bins, vert=True, patch_artist=True)

        # Add count annotations below x-axis labels
        ylim = plt.ylim()
        y_min = ylim[0]
        annotation_y = y_min - 0.1 * (ylim[1] - ylim[0])  # Place annotations slightly below the x-axis labels
        for i, count in enumerate(counts, start=1):
            plt.text(i, annotation_y, f'{count}', ha='center', va='top', fontsize=9, color='blue')

        # Adjust y-axis to leave space for text annotations
        plt.ylim(y_min - 0.2 * (ylim[1] - ylim[0]), ylim[1])

        # Customize plot
        plt.xlabel('Probability (%) grouped by 5% bins')
        if use_orig_img:
            plt.ylabel('Intensity value in original data')
        else:
            plt.ylabel('Intensity value in bkg subtracted data')
        plt.title(f'Box Plot of Intensity Values at Each Probability Bin for {self.Label} at {self.Scale} for {total_elements} pixels')
        plt.xticks(rotation=45)
        # plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save or display the plot
        plt.tight_layout()
        if write_fig:
            plt.savefig(f"{self.BaseDir}/Figures/ProbIntensityPlot_{self.Label}_{self.Scale}_{use_orig_img}.png")



    def blurComposite(self, set_blur_as_prob = True, write_fits = True):

        """
        blur the originally dimesnioned composite image so it is not merely a stack of lines. To blur the reproected/dpwnsampled composite, adjust scalepix. 

        Parameters:
        - set_blur_as_prob (bool): set the blurred composite to be the probability map. 
        - write_fits (bool): Save the blurred composite as a fits file
        """
            
        struct_width = self.Scale.replace('pc',"")
        struct_width = float(struct_width)

        # Convert structure width from parsecs to pixels
        structure_width_pixels = struct_width / self.Scalepix
        
        # Calculate sigma for Gaussian convolution
        sigma = structure_width_pixels / 2.355  # FWHM = 2.355 * sigma -> sigma = FWHM / 2.355

        # Apply Gaussian blur
        blurred_image = gaussian_filter(self.Composite, sigma=sigma)

        self.Composite = blurred_image
        if(set_blur_as_prob):
            self.ProbabilityMap = blurred_image

        if(write_fits):
            output_directory = Path(f"{self.BaseDir}/{self.Label}/Composites")
            output_name = Path(f"{self.FitsFile}_CompositeBlur")
            output_fits_path = os.path.join(output_directory, str(output_name) + '.fits')
            hdu = fits.PrimaryHDU(data=blurred_image, header=self.OrigHeader)
            hdu.writeto(output_fits_path, overwrite=True)



    def applyProbabilityThresholdAndSkeletonize(self, probability_threshold, min_len_pix, write_fits = True):

        """
        Given the blurred composite, threshold the image. 

        Parameters:
        - probability_threshold (float): Minimum percentage between 0 and 1 that a pixel must have in order for it to be considered "real". 
        - min_len_pix (int): minimum area in pixels a filament must occupy. 
        - write_fits (bool): Save the cleaned composite as a fits file

        Returns: 
        - skelComposite (float): Thresholded data
        """
                
        #threshold
        ProbabilityThresh = np.max(self.Composite)*probability_threshold
        ret, threshComposite = cv2.threshold(self.Composite, ProbabilityThresh, 255, cv2.THRESH_BINARY)

        #threshold
        ProbabilityThresh = np.max(self.Composite)*probability_threshold
        ret, threshComposite = cv2.threshold(self.Composite, ProbabilityThresh, 255, cv2.THRESH_BINARY)

        #skeltonize
        skelComposite = skeletonize(threshComposite)
        skelComposite = skelComposite.astype(np.uint8)

        # #remove small filaments
        # img = np.array(skelComposite)
        # labels, stats, num_labels = AF.identify_connected_components(np.array(skelComposite))
        # small_areas = AF.sort_label_id(num_labels, stats, min_len_pix)
        # for label_id in small_areas:

        #     # Extract the bounding box coordinates
        #     left = stats[label_id, cv2.CC_STAT_LEFT]
        #     top = stats[label_id, cv2.CC_STAT_TOP]
        #     width = stats[label_id, cv2.CC_STAT_WIDTH]
        #     height = stats[label_id, cv2.CC_STAT_HEIGHT]

        #     for x in range(width):
        #         for y in range(height):
        #             img[top:top+height, left:left+width] = 0

        # skelComposite = img.astype(np.uint8)


        # Identify connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.uint8(skelComposite), connectivity=8
        )

        img = np.copy(skelComposite)

        for label_id in range(1, num_labels):  # skip background (0)
            # Extract component mask
            component_mask = (labels == label_id)

            # Skeletonize the component to measure its length
            skeleton = skeletonize(component_mask)

            # Compute filament length (number of skeleton pixels)
            filament_length = np.sum(skeleton)

            # Remove component if it's shorter than threshold
            if filament_length < min_len_pix:
                img[component_mask] = 0

        skelComposite = img.astype(np.uint8)

        if write_fits:   
            output_directory = Path(f"{self.BaseDir}/{self.Label}/Composites")
            output_name = Path(f"{self.FitsFile}_Composite_{probability_threshold}")
            output_fits_path = os.path.join(output_directory, str(output_name)  + '.fits')
            hdu = fits.PrimaryHDU(data=skelComposite, header=self.OrigHeader)
            hdu.writeto(output_fits_path, overwrite=True)

        
        return skelComposite

    def removeJunctions(self, skelComposite, probability_threshold, min_len_pix, set_as_composite= False, write_fits = True):

        """
        Given the thresholded composite, remove junctions and remove small areas. 

        Parameters:
        -skelComposite (float): skeletonized data to remove junctions and small filaments from
        - probability_threshold (float): Minimum percentage between 0 and 1 that a pixel must have in order for it to be considered "real". 
        - min_len_pix (int): Minimum area in pixels that a box surrounding a filament must cover. 
        -set_as_composite (bool): set the new image as the composite image. Overrides the blurred image as the composite. Set to false to test many probability thresholds at once. 
        write_fits (bool): Save the cleaned composite as a fits file
        """


        #remove junctions
        junctions = AF.getSkeletonIntersection(np.array(255*skelComposite))
        IntersectsRemoved = AF.removeJunctions(junctions, skelComposite, dot_size = 1)

        # #remove small filaments
        # labels, stats, num_labels = AF.identify_connected_components(np.array(IntersectsRemoved))
        # small_areas = AF.sort_label_id(num_labels, stats, min_len_pix)
        # img = np.array(IntersectsRemoved)
        # for label_id in small_areas:

        #     # Extract the bounding box coordinates
        #     left = stats[label_id, cv2.CC_STAT_LEFT]
        #     top = stats[label_id, cv2.CC_STAT_TOP]
        #     width = stats[label_id, cv2.CC_STAT_WIDTH]
        #     height = stats[label_id, cv2.CC_STAT_HEIGHT]

        #     for x in range(width):
        #         for y in range(height):
        #             img[top:top+height, left:left+width] = 0

        # Identify connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            np.uint8(IntersectsRemoved), connectivity=8
        )

        img = np.copy(IntersectsRemoved)

        for label_id in range(1, num_labels):  # skip background (0)
            # Extract component mask
            component_mask = (labels == label_id)

            # Skeletonize the component to measure its length
            skeleton = skeletonize(component_mask)

            # Compute filament length (number of skeleton pixels)
            filament_length = np.sum(skeleton)

            # Remove component if it's shorter than threshold
            if filament_length < min_len_pix:
                img[component_mask] = 0

        if(set_as_composite):
            self.Composite = img

        if write_fits:   
            output_directory = Path(f"{self.BaseDir}/{self.Label}/Composites")
            output_name = Path(f"{self.FitsFile}_Composite_{probability_threshold}_JR")
            output_fits_path = os.path.join(output_directory, str(output_name)  + '.fits')
            hdu = fits.PrimaryHDU(data=img, header=self.OrigHeader)
            hdu.writeto(output_fits_path, overwrite=True)
    

    def getLabel(self):
        return self.Label  
             
    def getBkgSubDivRMSMap(self):
        return self.BkgSubDivRMSMap  
    
    def getScale(self):
        return self.Scale  
                                

    def getFilamentLengthHistogram(self, probability_threshold, write_fig = True):

        """
        Create a histogram of filament lengths in parsecs. 
        NOT RELIABLE. 

        Parameters:
        - probability_threshold (float): Used here in the plot name. Changing this parameter will change filament lengths. 
        - write_fig (bool): Save the histogram
        """

        # Label connected components
        labeled_image = measure.label(self.Composite, connectivity=2)
        regions = measure.regionprops(labeled_image)

        # Calculate lengths (perimeter) of each curve
        lengths = [region.perimeter for region in regions]

        #convert to parcecs
        lengths = [l * self.Scalepix for l in lengths]

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(lengths, bins=20, color='blue', edgecolor='black')
        plt.title(f'Filament Length Histogram using prob_threshold {probability_threshold} for {self.Label} at {self.Scale}')
        plt.xlabel('Length (parcecs)')
        plt.ylabel('Frequency')
        if write_fig:
            plt.savefig(Path(f"{self.BaseDir}/Figures/FilamentLengthHistogram_{self.Label}_{self.Scale}.png"))



    def getNoiseLevelsHistogram(self, noise_min = 10**-2, write_fig = True):

        """
        Create a histogram of RMS noise in the CDD image

        Parameters:
        - noise_min (float): minimum noise to be considered real
        - write_fig: save the histogram
        """

        noise = self.NoiseMap.flatten()
        noise = noise[noise != 0]

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(noise, bins=20, color='blue', edgecolor='black')
        plt.title(f'noise histogram for {self.Label} at {self.Scale} with artificially set minimum of {noise_min} and zero removed)')
        plt.xlabel('noise')
        plt.ylabel('Frequency')
        if write_fig:
            plt.savefig(Path(f"{self.BaseDir}/Figures/NoiseLevelHistogram_{self.Label}_{self.Scale}.png"))



    def reprojectWrapper(self, inData, inHeader, OutputHeader, OutputData):
        """
        Reproject data from one fits file to another. Take OrigData and OrigHeader and reproject into the frame of OutputHeader and OutputData.

        Parameters:
        - OrigData: Data to reproject
        - OrigHeader: Header of data to reproject
        - OutputHeader: Header of data to reproject into
        - OutputData: Data to reproject into, used only for shape
        """

        reprojected_data, _ = reproject_interp((inData, inHeader), OutputHeader, shape_out=(np.shape(OutputData)))
        reprojected_data = self._cropNanBorder(reprojected_data, np.shape(OutputData))
        reprojected_data = np.nan_to_num(reprojected_data, nan=0.0)

        return reprojected_data
    


    def getRegionData(self, use_Regions):

        """
        Find the appropriate region file and reproject into the size of the original image. 

        Parameters:
        - use_Regions (str): directory where all region files live

        Returns:
        - data_new (float): region file reprojected into original image shape
        """

        # Look through files in region directory
        dir = use_Regions
        for file in os.listdir(dir):
            galaxy = self.Label.split("_")
            galaxy = galaxy[0]
            if galaxy.lower() in file.lower():
                fits_path = os.path.join(dir, file)

                with fits.open(fits_path, ignore_missing=True) as hdul:
                    data = np.array(hdul[0].data)  # region map
                    header = hdul[0].header

                data_new = self.reprojectWrapper(data, header, self.BlockHeader, self.BlockData) #use reprojected down data
                return data_new #return region file reprojected into original file shape
            
        print(f'could not find file in {dir}')
        return np.zeros(self.OrigData.shape)


    def getRegion(self, mask, data_new):
        
        """
        Find the region a prticular file belongs to. This is determined by the majority region value within the mask. 

        Parameters:
        - mask (int): Image with all zeros except for the filament pixels, which are 1
        - data_new (float): region file reprojected into original image shape

        Returns:
        - region (int): region where the filament belongs to 
        """
                 
        #data_new is the region mask simple file
        if np.sum(data_new) == 0:
            return -1
        
        # Region values inside mask
        region_values = data_new[mask > 0]

        if region_values.size == 0:
            print("Warning: mask covers no valid pixels")
            return -1

        # Find most common region
        region_ids, counts = np.unique(region_values.astype(int), return_counts=True)
        dominant_region = region_ids[np.argmax(counts)]

        # print(f"success, dominant region is: {dominant_region}")
        return int(dominant_region)

    def processFilamentCenters(self, coords_data):
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.morphologyEx(coords_data, cv2.MORPH_CLOSE, kernel)
        dilated_image = skeletonize(dilated_image.astype(bool))
        coords_data = dilated_image.astype(np.uint8)

        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_filCentersDilated.fits"
        hdu = fits.PrimaryHDU(coords_data, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)


        #Step 4: Remove junctions
        fil_centers = copy.deepcopy(coords_data)
        junctions = AF.getSkeletonIntersection(np.array(fil_centers*255))
        IntersectsRemoved = AF.removeJunctions(junctions, fil_centers, dot_size = 1) #check intersects removed
        IntersectsRemoved[IntersectsRemoved > 0] = 1
        IntersectsRemoved[IntersectsRemoved < 0] = 0
        fil_centers = IntersectsRemoved

        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_filCentersProcessed.fits"
        hdu = fits.PrimaryHDU(fil_centers, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)
        
        return fil_centers, coords_data        


    def createFilamentDictionary(self, rep_centers, use_Regions, min_scale):
        data_new = self.getRegionData(use_Regions)
        label_val = 3
        Scale = self._getScale( self.FitsFile)
        Scale = Scale.replace("pc", '')
        Scale = float(Scale)
    
        if self.BlockFactor !=0:
            min_area =  int(2*((16*self.BlockFactor)/self.Scalepix)**2) #aspect ratio of 8, all images used on blocked image of 16pc character. KEEP AN EYE ON THIS LINE. 
            imgNew = np.zeros_like(self.BlockData, dtype = float)
        else:
            min_area =  int(2*(16/self.Scalepix)**2) #aspect ratio of 8, all images used on blocked image of 16pc character. KEEP AN EYE ON THIS LINE. 
            imgNew = np.zeros_like(self.OrigData, dtype = float)

        #debugging min_area, use 10 for now? 
        min_area = 10

        segment_map = detect_sources(rep_centers, threshold=.5, npixels=min_area)
        segm_deblend = deblend_sources(rep_centers, segment_map, npixels=min_area, nlevels=32, contrast=0.001,progress_bar=False)

        for label in segm_deblend.labels:
            mask = segm_deblend.data == label # Find pixels that belong to this segment
            imgNew[mask] = label_val
            label_val+=10

        # Create a labeled mask for all filaments
        segment_info_reprojected = {}
        imgNew = np.rint(imgNew).astype(int)

        print(f'max label is: {label_val}, number of filaments: {label_val//10}')

        # Extract coordinates for each label
        for lab in range(3, label_val + 1, 10):
            white_mask = (imgNew >= lab -1) & (imgNew <= lab + 1)
            if not np.any(white_mask):
                continue 
            coords = np.argwhere(white_mask)
            img_skel = skeletonize(white_mask.astype(bool))
            numpix = len(np.argwhere(img_skel)) #determine length from skeletonized filament
            coords_list = [(int(x), int(y)) for y, x in coords]
            region = self.getRegion(white_mask, data_new)
            segment_info_reprojected[lab] = (coords_list, numpix, region, lab)

        print('Dictionary reprojected.')
        print(np.max(segment_info_reprojected.keys()))

        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_imgNewForFilDict.fits"
        hdu = fits.PrimaryHDU(imgNew, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)


        return segment_info_reprojected, Scale


    def runPSF(self, data, coords_data, header, write_fits):

        # fwhmval = int(Scale/self.Scalepix)
        fwhmval = int(16/self.Scalepix) #KEEP AN EYE HERE. 


        psf_model = CircularGaussianPRF(flux=1, fwhm=fwhmval)  # No longer divide fwhmval/2.35
        psf_model.x_0.fixed = True  # allowing this to vary when x_coords, y_coords given no intentional offset shows most would only move by ~0.125 pix, so neglect any shift
        psf_model.y_0.fixed = True
        psf_model.fwhm.fixed = False
        psf_model.flux.min = 0.0  #### -1. * np.std(noise)  # keep all models positive
        psf_model.fwhm.max = fwhmval * 2.0  # do not allow the fwhm to encroach into next larger single scale interval
        psf_model.fixed

        # step 7: Apply PSF to create model
        net_scaling_factor = 3.2728865403756338
        y_coords, x_coords = np.where(coords_data > 0)
        init_params = QTable()
        init_params['x'] = x_coords + 0.0
        init_params['y'] = y_coords + 0.0
        init_params['flux'] = coords_data[y_coords, x_coords] * net_scaling_factor


        # Define PSF fitting region
        psf_shape = (2 * int(np.ceil(fwhmval)) + 1, 2 * int(np.ceil(fwhmval)) + 1)
        fit_shape = psf_shape

        try:
            grouper = SourceGrouper(min_separation=1)
            psfphot = PSFPhotometry(
                psf_model,
                fit_shape,
                grouper=grouper,
                fitter_maxiters=2
            ) 
            phot = psfphot(data, error=self.NoiseMap, init_params=init_params)
            tag = 'Grouped'
        except MemoryError:
            psfphot = PSFPhotometry(
                psf_model,
                fit_shape,
                grouper=None,
                fitter_maxiters=2
            ) 
            phot = psfphot(data, error=self.NoiseMap, init_params=init_params)
            tag = 'NotGrouped'
            
        #model
        resid = psfphot.make_residual_image(data)
        model = (data - resid)  # Model is data minus residuals

        #Scale model again
        ratio=data[model != 0]/model[model != 0]
        ratiouseful=ratio[(ratio>0.05) & (ratio<6.)]
        ratiomean,ratiomedian,ratiostd=sigma_clipped_stats(ratiouseful, sigma=2, maxiters=5)
        globalfactor=ratiomedian
        model = globalfactor * model

        if(write_fits):
            out_path = Path(f"{self.BaseDir}/{self.Label}/SyntheticMap/{self.FitsFile}_SyntheticMap.fits")
            hdu = fits.PrimaryHDU(model, header=header)
            hdu.writeto(out_path, overwrite=True)


        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_PSFSyntheticModel.fits"
        hdu = fits.PrimaryHDU(model, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)


        return model, tag, globalfactor, phot


    def getMolecularMass(self, I_CO__2_1_16pc, alphaCO_tag, use_dynamic_alphaCO):

        # Use PHANGS alphaCO
        if use_dynamic_alphaCO is not None:
            print(f'Using dynamic alphaCO')
            dir_path = use_dynamic_alphaCO
            found_alphaCO = False

            for file in os.listdir(dir_path):
                galaxy = self.Label.split("_")[0]  # Get first part of label
                if galaxy.lower() in file.lower() and alphaCO_tag.lower() in file.lower():
                    print("Matched alphaCO map!")
                    fits_path = os.path.join(dir_path, file)

                    with fits.open(fits_path, ignore_missing=True) as hdul:
                        alphaCO = np.array(hdul[0].data)  # region map
                        header = hdul[0].header

                    alphaCO = self.reprojectWrapper(alphaCO, header, self.BlockHeader, self.BlockData)

                    # assert(np.shape(alphaCO) == np.shape(self.OrigData))

                    Molecular_Mass = alphaCO * I_CO__2_1_16pc  # Units of Solar mass per pix^2
                    # Create masks for NaNs (only NaNs, not inf)
                    nan_mask = np.isnan(Molecular_Mass)
                    n_nans = np.count_nonzero(nan_mask)
                    
                    # debugging step
                    # fits.writeto("Molecular_Mass.fits", Molecular_Mass, overwrite=True)
                    # fits.writeto("alphaCO.fits", alphaCO, overwrite=True)
                    # fits.writeto("I_CO_2_1_16pc.fits", I_CO__2_1_16pc, overwrite=True)
                    # fits.writeto("nan_mask.fits", nan_mask.astype(np.uint8), overwrite=True)  
                    # (save mask as 0/1 integers so it’s more readable)

                    if n_nans == 0:
                        print("No NaNs detected in Molecular_Mass.")
                    else:
                        print(f"Detected {n_nans} NaN pixel(s) in Molecular_Mass. Replacing with 5.5 * I_CO__2_1_16pc where possible...")

                        # candidate replacement array
                        replacement = 5.5 * I_CO__2_1_16pc

                        # mask where replacement is finite (so we don't inject NaNs)
                        replacement_ok = np.isfinite(replacement)

                        # positions we can actually replace: originally NaN AND replacement finite
                        can_replace = nan_mask & replacement_ok
                        cannot_replace = nan_mask & ~replacement_ok

                        # do the replacement (in-place)
                        Molecular_Mass[can_replace] = replacement[can_replace]

                        # report results
                        n_replaced = np.count_nonzero(can_replace)
                        n_remaining = np.count_nonzero(cannot_replace)

                        print(f"Replaced {n_replaced} pixel(s).")
                        if n_remaining:
                            print(f"{n_remaining} pixel(s) remain NaN because replacement values were not finite.")
                        else:
                            print("No remaining NaNs; all NaNs successfully replaced.")
                    found_alphaCO = True
                    break

            if not found_alphaCO:
                print(f'Could not find {alphaCO_tag} alphaCO file for {self.Label}')
                Molecular_Mass = 5.5 * I_CO__2_1_16pc  # Default value
        else:
            Molecular_Mass = 5.5 * I_CO__2_1_16pc  # Default value


        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_MolecularMassMap.fits"
        hdu = fits.PrimaryHDU(Molecular_Mass, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)

        return Molecular_Mass

    def convertToCSV(self, Scale, tag, Molecular_Mass, segment_info_reprojected):
        print('converting to csv data')
        csv_data = {}

        Line_Density = []
        Lengths = []
        Mass = []
        curvatures = []
        regions = []
        label = []
        
        for fil_id, pix_info in segment_info_reprojected.items():
            mass_sum = []
            fil_length = pix_info[1]
            if self.BlockFactor != 0:
                fil_length = fil_length * self.BlockFactor * self.Scalepix  #adjust length for blocking
            else:
                fil_length = fil_length * self.Scalepix  #adjust length for blocking
            img = np.zeros_like(Molecular_Mass)

            for values in pix_info[0]:
                x = values[0]
                y = values[1]
                mass_sum.append(Molecular_Mass[y, x])
                img[y, x] = Molecular_Mass[y, x]

            img = img > 0
            curvature_img = skeletonize(img) #use centerline coordinates for curvature
            white_mask = curvature_img == True   # define mask of skeleton pixels
            curvature_coords = np.argwhere(white_mask)  # get coordinates

            Line_Density.append(np.sum(mass_sum) / (fil_length)) #check density
            Lengths.append(fil_length)
            Mass.append(np.sum(mass_sum))
            orientation, curvature, theta = rht_curvature_from_coords(curvature_coords.tolist(), shape = np.shape(Molecular_Mass), radius= int(round(self.Scalepix)), ntheta=180, background_percentile=25) #doule check radius
            curvatures.append(curvature)
            regions.append(pix_info[2])
            label.append(pix_info[3])

        # Store in dictionary with scale-specific column names
        assert(len(regions) == len(Lengths))
        csv_data[f'Line_Density_{Scale}'] = Line_Density
        csv_data[f'Length_{Scale}'] = Lengths
        csv_data[f'Mass_{Scale}'] = Mass
        csv_data[f'Curvature_{Scale}'] = curvatures
        csv_data[f'Regions_{Scale}'] = regions
        csv_data[f'Label_{Scale}'] = label

        print(len(Line_Density), len(Lengths), len(Mass), len(curvatures), len(regions))
        assert len(set([len(Line_Density), len(Lengths), len(Mass), len(curvatures), len(regions)])) == 1, "List length mismatch!"

        df = pd.DataFrame(csv_data)
        # Save to CSV
        csv_path = Path(self.BaseDir) / self.Label / "SyntheticMap" / f"{self.FitsFile}_DensityData_{tag}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)  # make sure dir exists
        if csv_path.exists():
            csv_path.unlink()  # delete existing file first
        df.to_csv(csv_path, index=False)

        # FITS
        path = Path(self.BaseDir) / self.Label / 'Molecular_Mass'
        os.makedirs(path, exist_ok=True)
        fits_path = Path(self.BaseDir) / self.Label / 'Molecular_Mass'/ f"{self.FitsFile}_MolecularMassMap_{tag}.fits"
        fits_path.parent.mkdir(parents=True, exist_ok=True)
        hdu = fits.PrimaryHDU(Molecular_Mass, header=self.OrigHeader)
        hdu.writeto(fits_path, overwrite=True)


        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        csv_path = base / f"{self.FitsFile}_CSVData_{tag}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)  # make sure dir exists
        if csv_path.exists():
            csv_path.unlink()  # delete existing file first
        df.to_csv(csv_path, index=False)


    def extractProperties(self, phot, tag, segment_info_reprojected, alphaCO_tag, use_dynamic_alphaCO, Scale, globalfactor):
            inclination = 9 * np.pi / 180  # Inclination in radians
            sSFR = 1.74 #get specific SFR*
            # I_F770W_16pc = model

            #construct the mass map based on center line fits 
            x_fit = phot['x_fit']
            y_fit = phot['y_fit']
            flux_fit = phot['flux_fit']
            flux_map = np.zeros_like(self.BlockData, dtype=float) #use blocked data since PSF ran on blocked data
            x_fit_int = np.clip(x_fit.astype(int), 0, flux_map.shape[1]-1)
            y_fit_int = np.clip(y_fit.astype(int), 0, flux_map.shape[0]-1)
            for x, y, f in zip(x_fit_int, y_fit_int, flux_fit):
                flux_map[y, x] = f
            I_F770W_16pc = flux_map*globalfactor

            I_F770W_16pc = I_F770W_16pc * np.cos(np.radians(inclination))
            log_C_F770W = -0.21 * (np.log10(sSFR) + 10.14)  
            valid_mask_1 = I_F770W_16pc > 0
            x = np.zeros_like(I_F770W_16pc)
            x[valid_mask_1] = np.log(I_F770W_16pc[valid_mask_1]) - log_C_F770W
            log_I_CO_2_1_16pc = 0.88 * (x - 1.44) + 1.36
            I_CO__2_1_16pc = 10**log_I_CO_2_1_16pc
            I_CO__2_1_16pc[~valid_mask_1] = 0

            Molecular_Mass = self.getMolecularMass(I_CO__2_1_16pc, alphaCO_tag, use_dynamic_alphaCO)

            self.convertToCSV(Scale, tag, Molecular_Mass, segment_info_reprojected)


    def getSyntheticFilamentMapExact(self, min_scale, alphaCO_tag, use_dynamic_alphaCO = None, use_Regions = None, extract_Properties = True, write_fits = True):

        """
        Use PSF fitting to create a synthetic image of only detected filaments. Then use this synthetic map to extract filament properties such as length, curvature, mass, line mass, surface density. 
        The Process is as follows: 

        1. Load the original CDD image and the composite image
        2. Process the composite image to remove junctions
        3. Apply blocking to speed up PSF fitting   
        4. Create a filament dictionary from the blocked and processed composite image using photutils segmentation to identify individual filaments
        5. Create a labeled mask for all filaments
        6. Reproject the labeled mask back to the original image size such that it lines up with the reprojected output from PSF. Use skeletonized filament to determine length and every pixel to find total mass. 
        7. For each filament, fit PSFs along the filament to create a synthetic image of only filaments
        8. If extract_Properties is True, extract filament properties using the synthetic image and the original CDD image

        Parameters:
        - alphaCO_tag (str): Tag to identify which alphaCO value to use from the config file
        - use_dynamic_alphaCO (str): String to dynamic alphaCO map directory
        - use_Regions (str): directory where all region files live
        - extract_Properties (bool): Whether or not to extract filament properties
        - write_fits (bool): Save the map as a fits file
        """

        # Load CDD Image
        print('Begenning synthetic map production')
        fits_path = os.path.join(self.BaseDir, self.Label)
        fits_path = os.path.join(fits_path, "CDD")
        
        if not self.FitsFile.endswith(".fits"):
            fits_path = os.path.join(fits_path, self.FitsFile + ".fits")
        else: 
            fits_path = os.path.join(fits_path, self.FitsFile)

        with fits.open(fits_path, ignore_missing=True) as hdul:
            data = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU 
            header = hdul[0].header

        data[np.isnan(data)] = 0

        coords_data = self.Composite

        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_OriginalFilCenters.fits"
        hdu = fits.PrimaryHDU(coords_data, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)

        #Apply blocking to speed up PSF fitting
        if self.BlockFactor != 0:
            data = self.reprojectWrapper(data, self.OrigHeader, self.BlockHeader, self.BlockData) 
            coords_data = self.reprojectWrapper(coords_data, self.OrigHeader, self.BlockHeader, self.BlockData)
        
        rep_centers, coords_data = self.processFilamentCenters(coords_data) #rep_centers has intersects removed and is used for dictionary creation. 

        model, tag, globalfactor,phot = self.runPSF(data, coords_data, header, write_fits)

        if extract_Properties: 
            #reproject this processed image back and detect filaments using photutils segmentation
            # rep_centers, _ = reproject_exact((rep_centers , self.BlockHeader), self.OrigHeader, shape_out=self.OrigData.shape) #keep downsampled size
            rep_centers[rep_centers > 0] = 1
            #Create a filament dictionary 
            segment_info_reprojected, Scale = self.createFilamentDictionary(rep_centers, use_Regions, min_scale) 
            self.extractProperties(phot, tag, segment_info_reprojected, alphaCO_tag, use_dynamic_alphaCO, Scale, globalfactor)


    def getSyntheticFilamentMapApprox(self, min_scale, alphaCO_tag, use_dynamic_alphaCO = None, use_Regions = None, extract_Properties = True, write_fits = True):


        # Load CDD Image
        print('Begenning synthetic map production')
        fits_path = os.path.join(self.BaseDir, self.Label)
        fits_path = os.path.join(fits_path, "CDD")
        
        if not self.FitsFile.endswith(".fits"):
            fits_path = os.path.join(fits_path, self.FitsFile + ".fits")
        else: 
            fits_path = os.path.join(fits_path, self.FitsFile)

        with fits.open(fits_path, ignore_missing=True) as hdul:
            data = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU 
            header = hdul[0].header

        data[np.isnan(data)] = 0

        coords_data = self.Composite

        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_OriginalFilCenters.fits"
        hdu = fits.PrimaryHDU(coords_data, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)

        #Apply blocking to speed up PSF fitting
        if self.BlockFactor != 0:
            data = self.reprojectWrapper(data, self.OrigHeader, self.BlockHeader, self.BlockData) 
            coords_data = self.reprojectWrapper(coords_data, self.OrigHeader, self.BlockHeader, self.BlockData)
        
        rep_centers, coords_data = self.processFilamentCenters(coords_data) #rep_centers has intersects removed and is used for dictionary creation. 

        model, tag, globalfactor, phot = self.runImageLSE(data, coords_data, header, write_fits)

        if extract_Properties: 
            #reproject this processed image back and detect filaments using photutils segmentation
            # rep_centers, _ = reproject_exact((rep_centers , self.BlockHeader), self.OrigHeader, shape_out=self.OrigData.shape) #keep downsampled size
            rep_centers[rep_centers > 0] = 1
            #Create a filament dictionary 
            segment_info_reprojected, Scale = self.createFilamentDictionary(rep_centers, use_Regions, min_scale) 
            self.extractProperties(phot, tag, segment_info_reprojected, alphaCO_tag, use_dynamic_alphaCO, Scale, globalfactor)

    def runImageLSE(self, data, coords_data, header, write_fits):

        # step 7: Apply PSF to create model
        net_scaling_factor = 3.2728865403756338
        initialImage = coords_data* net_scaling_factor
        alpha = np.dot(initialImage.ravel(), data.ravel()) / np.dot(initialImage.ravel(), initialImage.ravel())
        model = alpha*initialImage
        # Find non-zero pixels
        y_fit, x_fit = np.nonzero(model)

        # Extract flux values
        flux_fit = model[y_fit, x_fit]

        # Build a phot-style table
        phot = Table()
        phot['x_fit'] = x_fit.astype(float)
        phot['y_fit'] = y_fit.astype(float)
        phot['flux_fit'] = flux_fit.astype(float)

        tag = 'approximateFit'
        globalfactor = 1

        if(write_fits):
            out_path = Path(f"{self.BaseDir}/{self.Label}/SyntheticMap/{self.FitsFile}_SyntheticMapApprox.fits")
            hdu = fits.PrimaryHDU(model, header=header)
            hdu.writeto(out_path, overwrite=True)

        #debugging
        base = Path(r"C:/Users/jhoffm72/Documents/FilPHANGS/PropertyTestData")
        out_path = base / f"{self.FitsFile}_UniformScaleSyntheticModel.fits"
        hdu = fits.PrimaryHDU(model, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)

        return model, tag, globalfactor, phot


#Functions copied directy from FilFinder
def rht_curvature_from_coords(coords, shape, radius=10, ntheta=180, background_percentile=25):
    """
    Run RHT curvature analysis on a set of pixel coordinates.

    Parameters
    ----------
    coords : list of (int, int)
        List of (row, col) pixel positions.
    shape : tuple
        Shape of the full image (ny, nx).
    radius : int
        Radius for RHT.
    ntheta : int
        Number of sampled angles.
    background_percentile : float
        Background level to subtract.

    Returns
    -------
    orientation : Quantity
        Mean orientation of the pixels (astropy Quantity, radians).
    curvature : Quantity
        Angular width = curvature (astropy Quantity, radians).
    (theta, R) : tuple
        Full histogram of RHT output.
    """
    # 1. Create mask of selected pixels
    mask = np.zeros(shape, dtype=bool)
    for y,x  in coords:
        mask[y, x] = True

    # 2. Run RHT
    theta, R, quant = rht(mask, radius, ntheta, background_percentile)

    twofive, mean, sevenfive = quant

    # 3. Orientation
    orientation = mean * u.rad

    # 4. Curvature (spread of angles)
    if sevenfive > twofive:
        curvature = np.abs(sevenfive - twofive) * u.rad
    else:
        curvature = (np.abs(sevenfive - twofive) + np.pi) * u.rad

    return orientation, curvature, (theta, R)

def rht(mask, radius, ntheta=180, background_percentile=25, verbose=False):
    '''

    Parameters
    ----------

    mask : numpy.ndarray
        Boolean or integer array. Transform performed at all
        non-zero points.

    radius : int
            Radius of circle used around each pixel.

    ntheta : int, optional
            Number of angles to use in transform.

    background_percentile : float, optional
                            Percentile of data to subtract off. Background is
                            due to limits on pixel resolution.

    verbose : bool, optional
        Enables plotting.

    Returns
    -------

    theta : numpy.ndarray
            Angles transform was performed at.

    R : numpy.ndarray
        Transform output.

    quantiles : numpy.ndarray.
        Contains the 25%, mean, and 75% percentile angles.

    '''

    pad_mask = np.pad(mask.astype(float), radius, padwithnans)

    # The theta=0 case isn't handled properly
    theta = np.linspace(np.pi/2., 1.5*np.pi, ntheta)

    # Create a cube of all angle positions
    circle, mesh = circular_region(radius)
    circles_cube = np.empty((ntheta, circle.shape[0], circle.shape[1]))
    for posn, ang in enumerate(theta):
        diff = mesh[0]*np.sin(ang) - mesh[1]*np.cos(ang)
        diff[np.where(np.abs(diff) < 1.0)] = 0
        circles_cube[posn, :, :] = diff

    R = np.zeros((ntheta,))
    x, y = np.where(mask != 0.0)
    for i, j in zip(x, y):
        region = np.tile(circle * pad_mask[i:i+2*radius+1,
                                        j:j+2*radius+1], (ntheta, 1, 1))
        line = region * np.isclose(circles_cube, 0.0)

        if not np.isnan(line).all():
            R = R + np.nansum(np.nansum(line, axis=2), axis=1)

    # Check that the ends are close.
    if np.isclose(R[0], R[-1], rtol=1.0):
        R = R[:-1]
        theta = theta[:-1]
    else:
        raise ValueError("R(-pi/2) should equal R(pi/2). Check input.")

    # You're likely to get a somewhat constant background, so subtract it out
    R = R - np.median(R[R <= scoreatpercentile(R, background_percentile)])
    if (R < 0.0).any():
        R[R < 0.0] = 0.0  # Ignore negative values after subtraction

    # Return to [-pi/2, pi/2] interval and position to the correct zero point.
    theta -= np.pi
    R = np.fliplr(R[:, np.newaxis])

    mean_circ = circ_mean(theta, weights=R)
    twofive, sevenfive = circ_CI(theta, weights=R, u_ci=0.67)
    twofive = twofive[0]
    sevenfive = sevenfive[0]
    quantiles = (twofive, mean_circ, sevenfive)


    return theta, R, quantiles


def circular_region(radius):
    '''
    Create a circle of a given radius.
    Values are NaNs outside of the circle.

    Parameters
    ----------
    radius : int
        Circle radius.

    Returns
    -------
    circle : numpy.ndarray
        Array containing the circle.
    [xx, yy] : numpy.ndarray
        Grids used to create the circle.
    '''
    xx, yy = np.mgrid[-radius:radius+1, -radius:radius+1]

    circle = xx**2. + yy**2.
    circle = circle < radius**2.

    circle = circle.astype(float)
    circle[np.where(circle == 0.)] = np.nan

    return circle, [xx, yy]


def padwithnans(vector, pad_width, iaxis, kwargs):
    left, right = (int(pad_width[0]), int(pad_width[1]))
    vector[:left] = np.nan
    vector[-right:] = np.nan
    return vector


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def find_nearest_posn(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


def circ_mean(theta, weights=None):
    """
    Calculates the median of a set of angles on the circle and returns
    a value on the interval from (-pi, pi].  Angles expected in radians.

    Parameters
    ----------
    """

    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]

    if weights is None:
        weights = np.ones(theta.shape)

    medangle = np.arctan2(np.nansum(np.sin(2*theta) * weights),
                        np.nansum(np.cos(2*theta) * weights))

    medangle /= 2.0

    return medangle


def fourier_shifter(x, shift, axis):
    '''
    Shift an array by some value along its axis.
    '''
    ftx = np.fft.fft(x, axis=axis)
    m = np.fft.fftfreq(x.shape[axis])
    # m_shape = [1] * x.ndim
    # m_shape[axis] = m.shape[0]
    # m = m.reshape(m_shape)
    slices = [slice(None) if ii == axis else None for ii in range(x.ndim)]
    m = m[tuple(slices)]
    phase = np.exp(-2 * np.pi * m * 1j * shift)
    x2 = np.real(np.fft.ifft(ftx * phase, axis=axis))
    return x2


def circ_CI(theta, weights=None, u_ci=0.67, axis=0):
    '''

    '''

    if len(theta.shape) == 1:
        theta = theta[:, np.newaxis]

    if weights is None:
        weights = np.ones(theta.shape)
    else:
        if len(weights.shape) == 1:
            weights = weights[:, np.newaxis]

    assert theta.shape == weights.shape

    # Normalize weights
    weights /= np.sum(weights, axis=axis)

    mean_ang = circ_mean(theta, weights=weights)

    # Now center the data around the mean to find the CI intervals
    diff_val = np.diff(theta[:2, 0])[0]

    theta_mid = theta[theta.shape[0] // 2]

    diff_posn = - (theta_mid - mean_ang) / diff_val

    theta_copy = fourier_shifter(theta, diff_posn, axis=0)

    vec_length2 = np.sum(weights * np.cos(theta_copy), axis=axis)**2. + \
        np.sum(weights * np.sin(theta_copy), axis=axis)**2.

    alpha = np.sum(weights * np.cos(2 * theta_copy), axis=axis)

    var_w = (1 - alpha) / (4 * vec_length2)

    # Make sure the CI stays within the interval. Otherwise assign it to
    # pi/2 (largest possible on interval of pi)
    sin_arg = u_ci * np.sqrt(2 * var_w)

    if sin_arg <= 1:
        ci = np.arcsin(sin_arg)
    else:
        ci = np.pi / 2.

    samp_cis = np.vstack([mean_ang - ci, mean_ang + ci])

    return samp_cis

def filter_short_components(binary_image, min_len_pix):
    """
    Removes connected components shorter than a given length threshold.
    Automatically detects whether components are already skeletonized.

    Parameters
    ----------
    binary_image : 2D array (bool or int)
        Input binary mask (True/1 for object pixels).
    min_len_pix : int
        Minimum length threshold (in pixels) for keeping components.

    Returns
    -------
    filtered_image : 2D np.ndarray (uint8)
        Binary image with short components removed.
    """
    if binary_image.dtype != bool:
        binary_image = binary_image.astype(bool)

    # Label connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image.astype(np.uint8), connectivity=8
    )

    filtered_image = np.copy(binary_image)

    for label_id in range(1, num_labels):  # skip background
        component_mask = (labels == label_id)
        if not np.any(component_mask):
            continue

        # --- Detect if already skeletonized ---
        neighbors = ndimage.convolve(component_mask.astype(int),
                                    np.ones((3, 3)),
                                    mode='constant', cval=0)
        avg_neighbors = np.mean(neighbors[component_mask])
        is_skeleton = avg_neighbors < 2.5  # empirically safe threshold

        # Skeletonize if needed
        skeleton = component_mask if is_skeleton else skeletonize(component_mask)

        filament_length = np.sum(skeleton)

        if filament_length < min_len_pix:
            filtered_image[component_mask] = 0

    return filtered_image.astype(np.uint8)