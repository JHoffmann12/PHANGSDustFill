#Filament map class to produce skeletonized filament maps of astrophysical images

#imports 
import AnalysisFuncs as AF
import copy
import csv
import cv2
import importlib
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import subprocess
import sys
import threading
import time
import timeit
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from glob import glob
from matplotlib.colors import Normalize
from photutils.background import Background2D, MedianBackground
from reproject import reproject_exact
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import kde, lognorm
from skimage import measure
from skimage.morphology import skeletonize

matplotlib.use('Agg')


class FilamentMap:

    def __init__(self,  scalepix, base_dir, label_folder_path, fits_file, label, param_file_path, flatten_perc):

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

        with fits.open(fits_path, ignore_missing=True) as hdul:
            OrigData = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU
            plt.imshow(OrigData)
            plt.title("Original")            
            plt.savefig(f"{base_dir}/{label}/BlockedPng/Zero_{self._getScale(fits_file)}.png")
            plt.close()

            OrigData = self.preprocessImage(OrigData, flatten_percent = flatten_perc)

            OrigData = np.nan_to_num(OrigData, nan=0.0)  # Replace NaNs with 0
            OrigHeader = hdul[0].header

        self.Label = label
        self.ProbabilityMap = np.zeros_like(OrigData)
        self.BkgSubDivRMSMap = np.zeros_like(OrigData)
        Scale = self._getScale(fits_file)
        self.Scale = Scale
        self.OrigHeader = OrigHeader
        self.OrigData = OrigData
        self.BlockHeader = OrigHeader.copy() #updated later
        self.Composite = np.zeros_like(OrigData)
        self.Scalepix = scalepix
        self.BaseDir = base_dir
        fits_file = os.path.splitext(fits_file)[0]  # removes the .fits extension
        self.FitsFile = fits_file
        self.BlockFactor = self._getBlockFactor()
        self.BlankRegionMask = self._getBlankRegionMask()

        if(self.BlockFactor != 0):
            self.BlockData = np.zeros((int(self.OrigData.shape[0] / self.BlockFactor), int(self.OrigData.shape[1] / self.BlockFactor))) 
        else: 
            self.BlockData = np.zeros_like(self.OrigData)

        self.IntensityMap = np.zeros_like(self.BlockData)
        self.NoiseMap = np.zeros_like(self.BlockData)
        self.ParamFile = param_file_path



    def preprocessImage(self, image, skip_flatten=False, flatten_percent=None):
        
        '''
        Preprocess and flatten the image before running the masking routine.

        Parameters:
        - skip_flatten (bool): optional. Skip the flattening step and use the original image to construct the mask. Default is False.
        - flatten_percent (int) : optional. The percentile of the data (0-100) to set the normalization of the arctan transform. By default, 
            a log-normal distribution is fit and the threshold is set to :math:`\mu + 2\sigma`. If the data contains regions of a much higher 
            intensity than the mean, it is recommended this be set >95 percentile.

        '''

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
            flat_img = thresh_val * np.arctan(image / thresh_val)
        
        return flat_img
            
    def setBlockFactor(self, bf):
        self.BlockFactor = bf
    

    def _getScale(self, fits_file):
        
        """
        get the scale associated with an image 

        """

        scales = ["1024pc", "512pc", "256pc", "128pc", "64pc", "32pc", "16pc", "8pc", "4pc", "2pc", "1pc", ".5pc", ".25pc", ".125pc", ".0625pc", ".03125pc"]
        
        for scale in scales:
            if scale in fits_file:
                return scale

        print("Invalid FITS file: Scale not recognized")
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
        # if(current_power <=1):
        #     return 0
        # else: 
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
        
        threshold = 10**-20 #pixels below this value will be seeds for mask

        data_to_mask = self.OrigData
        
        mask_copy = copy.deepcopy(data_to_mask)

        #Dilate White Pixels
        mask_copy[mask_copy > threshold] = 255
        mask_copy = mask_copy.astype(np.uint8)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_image = cv2.dilate(mask_copy, kernel, iterations= 5) 

        #threshold other way
        copy_image = copy.deepcopy(data_to_mask).astype(np.float32)
        copy_image[dilated_image < threshold] = np.nan
        mask = np.isnan(copy_image)

        #set up binary mask
        binary_mask = np.zeros_like(copy_image, dtype=np.uint8)
        binary_mask[mask] = 255
        binary_mask[~mask] = 0

        #dilate the mask
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        dilated_image = copy.deepcopy(data_to_mask).astype(np.float32)
        dilated_image[dilated_mask==255] = np.nan

        #make a mask based on dilated image, true value indicates pixel should be masked
        mask = np.isnan(dilated_image)
        assert(not np.isnan(mask).any())

        # plt.imshow(np.uint8(mask) * 255)
        # plt.title(f"Mask of {self.Label} at {self.Scale}")
        # plt.savefig(f"{self.BaseDir}\Figures\Mask_{self.Label}_{self.Scale}.png")
        # plt.close()

        return  mask


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
                box_size = int(.02*np.min((np.shape(data)[0], np.shape(data)[1]))) * 2 + 1 #Make box 2% of smaller image dimension...appears to work well

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


    def scaleBkgSubDivRMSMap(self, write_fits):

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
        save_png_path = fr"{self.BaseDir}\{self.Label}\BlockedPng\{self.FitsFile}_Blocked.png"
        pngData = self.BkgSubDivRMSMap.astype(np.uint16)

        cv2.imwrite(save_png_path, pngData)

        if(write_fits):
            print('saving bkg sub as fits')
            out_path = fr"{self.BaseDir}\{self.Label}\BkgSubDivRMS\{self.FitsFile}_BkgSubDivRMS.fits"
            hdu = fits.PrimaryHDU(self.BkgSubDivRMSMap, header=self.BlockHeader)
            hdu.writeto(out_path, overwrite=True)


    def setBlockData(self):

        """
        Fix the blocked header and block the image
        """
            
        if(self.BlockFactor !=0):

            #fix the blocked header
            print(f"Block factor: {self.BlockFactor} and Scale: {self.Scale}")
            self.BlockHeader['CDELT1'] = (self.OrigHeader['CDELT1']) * self.BlockFactor
            self.BlockHeader['CDELT2'] = (self.OrigHeader['CDELT2']) * self.BlockFactor

            self.BlockHeader['CRPIX1'] = (self.OrigHeader['CRPIX1']) / self.BlockFactor 
            self.BlockHeader['CRPIX2'] = (self.OrigHeader['CRPIX2']) / self.BlockFactor 

            #reproject the data
            reprojected_data = self.reprojectWrapper(self.OrigData, self.OrigHeader, self.BlockHeader, self.BlockData)
            self.BlankRegionMask = self.reprojectWrapper(self.BlankRegionMask, self.OrigHeader,self.BlockHeader, self.BlockData)

            # Crop NaN border 
            self.BlockData = self._cropNanBorder(reprojected_data, (np.shape(self.BlockData)))
            self.BlankRegionMask = self._cropNanBorder(self.BlankRegionMask, (np.shape(self.BlockData)))
            assert(np.shape(self.BlankRegionMask)==np.shape(self.BlockData))
            self.BlankRegionMask = self.BlankRegionMask == 1 #boolean mask, True indicates pixels to mask
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


        input_image = fr"{self.BaseDir}\{self.Label}\BlockedPng\{self.FitsFile}_Blocked.png"
        batch = batch_path
        output_dir = fr"{self.BaseDir}\{self.Label}\SOAXOutput\{self.Scale}"

        try: 
            assert(os.path.isdir(output_dir))
            assert( os.path.isfile(self.ParamFile))
            assert(os.path.isfile(input_image))
        except AssertionError as e: 
            print('Error: Assertions failed, a faulty path exists!')
            
        print("starting Soax")
        cmdString = f'"{batch}" soax -i "{input_image}" -p "{self.ParamFile}" -s "{output_dir}" --ridge {ridge_start} 0.0075 {ridge_stop} --stretch {stretch_start} 0.5 {stretch_stop}' #can update ridge and stretch later
        
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
        After the soax .txt file is produced, convert them to a .Fits file. This function manages a loop, calling the txtToFilaments function to do the heavy lifting. 

        Parameters:
        - output_dir (str): Directory to output the .Fits files to
        - ridge_start (float): starting ridge threshold for the soax run. Present in the .txt file name and allows threads to avoid one another.
        """

        for result_file in os.listdir(output_dir):
            if(result_file.endswith('.txt') and str(ridge_start) in result_file):
                result_file_base = os.path.splitext(result_file)[0]  # removes the .txt extension

                interpolate_path = fr"{self.BaseDir}\{self.Label}\SOAXOutput\{self.Scale}\{result_file_base}.fits"
                result_file = fr"{self.BaseDir}\{self.Label}\SOAXOutput\{self.Scale}\{result_file}"
                
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

        output_directory = fr"{self.BaseDir}\{self.Label}\Composites"
        directory = fr"{self.BaseDir}\{self.Label}\SOAXOutput\{self.Scale}"
        output_name = fr"{self.FitsFile}_Composites"
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
            output_fits_path = os.path.join(output_directory, output_name + ".fits")
            hdu = fits.PrimaryHDU(data=composite_data, header=header)
            hdu.writeto(output_fits_path, overwrite=True)
            print(f"Composite image saved as FITS")




    def setIntensityMap(self, orig = True):
        if(orig):
            self.IntensityMap[self.Composite !=0] = 255*self.OrigData[self.Composite!=0]
        else: 
            temp = self.reprojectWrapper(self.ProbabilityMap, self.OrigHeader, self.BlockHeader, self.BlockData)
            temp = self._cropNanBorder(temp, (np.shape(self.BlockData)))
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
            temp = self._cropNanBorder(temp, (np.shape(self.BlockData)))
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
        blur the composite so it is not merely a stack of lines

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
            output_directory = fr"{self.BaseDir}\{self.Label}\Composites"
            output_name = fr"{self.FitsFile}_CompositeBlur"
            output_fits_path = os.path.join(output_directory, output_name + '.fits')
            hdu = fits.PrimaryHDU(data=blurred_image, header=self.OrigHeader)
            hdu.writeto(output_fits_path, overwrite=True)



    def cleanComposite(self, probability_threshold, min_area_pix, set_as_composite= False, write_fits = True):

        """
        Given the blurred composite, threshold the image, skeletonize, remove junctions, and remove small areas. 

        Parameters:
        - probability_threshold (float): Minimum percentage between 0 and 1 that a pixel must have in order for it to be considered "real". 
        - min_area_pix (int): Minimum area in pixels that a box surrounding a filament must cover. 
        -set_as_composite (bool): set the new image as the composite image. Overrides the blurred image as the composite. Set to false to test many probability thresholds at once. 
        write_fits (bool): Save the cleaned composite as a fits file
        """

        #threshold
        ProbabilityThresh = np.max(self.Composite)*probability_threshold
        ret, threshComposite = cv2.threshold(self.Composite, ProbabilityThresh, 255, cv2.THRESH_BINARY)

        #skeltonize
        skelComposite = skeletonize(threshComposite)
        
        #remove junctions
        junctions = AF.getSkeletonIntersection(np.array(255*skelComposite))
        IntersectsRemoved = AF.removeJunctions(junctions, skelComposite, dot_size = 3)

        #remove small filaments
        labels, stats, num_labels = AF.identify_connected_components(np.array(IntersectsRemoved))
        small_areas = AF.sort_label_id(num_labels, stats, min_area_pix)
        img = np.array(IntersectsRemoved)
        for label_id in small_areas:

            # Extract the bounding box coordinates
            left = stats[label_id, cv2.CC_STAT_LEFT]
            top = stats[label_id, cv2.CC_STAT_TOP]
            width = stats[label_id, cv2.CC_STAT_WIDTH]
            height = stats[label_id, cv2.CC_STAT_HEIGHT]

            for x in range(width):
                for y in range(height):
                    img[top:top+height, left:left+width] = 0

        if(set_as_composite):
            self.Composite = img

        if write_fits:   
            output_directory = fr"{self.BaseDir}\{self.Label}\Composites"
            output_name = fr"{self.FitsFile}_Composite_{probability_threshold}"
            output_fits_path = os.path.join(output_directory, output_name + '.fits')
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
        Create a histogram of filament lengths in parsecs

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
            plt.savefig(fr"{self.BaseDir}\Figures\FilamentLengthHistogram_{self.Label}_{self.Scale}.png")



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
            plt.savefig(fr"{self.BaseDir}\Figures\NoiseLevelHistogram_{self.Label}_{self.Scale}.png")



    def getSyntheticFilamentMap(self,  write_fits = True):

        """
        Create a map of estimated filaments

        Parameters:
        - write_fits (bool): Save the map as a fits file
        """


        fits_path = os.path.join(self.BaseDir, self.Label)
        fits_path = os.path.join(fits_path, "CDD")
        fits_path = os.path.join(fits_path, self.FitsFile + ".fits")
        with fits.open(fits_path, ignore_missing=True) as hdul:
            inData = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU

        intensity_skel = self.Composite * inData
        struct_width = self.Scale.replace('pc',"")
        struct_width = float(struct_width)

        # Convert structure width from parsecs to pixels
        structure_width_pixels = struct_width / self.Scalepix
        # Calculate sigma for Gaussian convolution
        sigma = structure_width_pixels / 2.355  # FWHM = 2.355 * sigma -> sigma = FWHM / 2.355
        blurred_image = gaussian_filter(intensity_skel, sigma=sigma)        

        plt.figure(figsize=(8, 6))
        plt.imshow(blurred_image)
        plt.title(f'Synthetic Filament Map')
        os.makedirs(f"{self.BaseDir}\SyntheticMap", exist_ok=True)
        save_path = os.path.join(f"{self.BaseDir}\{self.Label}\SyntheticMap", f"SyntheticMap_{self.Scale}.fits")

        if write_fits:
            hdu = fits.PrimaryHDU(blurred_image, header=self.OrigHeader)
            hdu.writeto(save_path, overwrite=True)



    def reprojectWrapper(self, OrigData, OrigHeader, BlockHeader, BlockData):
        start = time.time()
        reprojected_data, _ = reproject_exact((OrigData, OrigHeader), BlockHeader, shape_out=(np.shape(BlockData)))
        end = time.time()
        elapsed_time = end - start
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"Reprojection complete in {hours:02d}:{minutes:02d}:{seconds:02d} in total!")
        return reprojected_data
    
    def applyIntensityThreshold(self, thresh):
        self.Composite[self.OrigData < thresh] = 0
