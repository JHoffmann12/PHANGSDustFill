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
import random 
import subprocess
import re
from scipy.stats import kde
from scipy.ndimage import gaussian_filter
import AnalysisFuncs as AF

class FilamentMap:

    def __init__(self,  Scalepix, HomeDir, FitsFile, Galaxy):
        fits_path = os.path.join(HomeDir, FitsFile)
        with fits.open(fits_path) as hdul:
            OrigData = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU
            OrigData = np.nan_to_num(OrigData, nan=0.0)  # Replace NaNs with 0
            OrigHeader = hdul[0].header
        self.Galaxy = Galaxy
        self.IntensityMap = np.zeros_like(OrigData)
        self.ProbabilityMap = np.zeros_like(OrigData)
        self.BkgSubMap = np.zeros_like(OrigData)
        Scale = self._getScale(FitsFile)
        self.Scale = Scale
        self.OrigHeader = OrigHeader
        self.OrigData = OrigData
        self.BlockHeader = OrigHeader.copy() #updated later
        self.Composite = np.zeros_like(OrigData)
        self.Scalepix = Scalepix
        self.HomeDir = HomeDir
        FitsFile = os.path.splitext(FitsFile)[0]  # removes the .fits extension
        self.FitsFile = FitsFile
        self.BlockFactor = self._getBlockFactor()
        if(self.BlockFactor != 0):
            self.BlockData = np.zeros((int(self.OrigData.shape[0] / self.BlockFactor), int(self.OrigData.shape[1] / self.BlockFactor))) 
        else: 
            self.BlockData = np.zeros_like(self.OrigData)



    def _getScale(self, fits_file):
        if '128pc' in fits_file:
            print('image is 128 parcec scale')
            return "128pc"
        
        elif '8pc' in fits_file: 
            print('image is 8 parcec scale')
            return "8pc"
        
        elif '16pc' in fits_file: 
            print('image is 16 parcec scale')
            return "16pc"
        
        elif '32pc' in fits_file: 
            print('image is 32 parcec scale')
            return "32pc"
        
        elif '64pc' in fits_file:
            print('image is 64 parcec scale')
            return "64pc"

        elif '256pc' in fits_file:
            print('image is 256 parcec scale')
            return "256pc"
        
        elif '512pc' in fits_file:
            print('image is 512 parcec scale')
            return "512pc"
        
        elif '1024pc' in fits_file:
            print('image is 1024 parcec scale')
            return "1024pc"

        else:
            print('invalid fits file')


    def _getBlockFactor(self):
        file_name = self.FitsFile
        folder_path = self.HomeDir
        
        # Pattern to match valid powers of two
        power_of_two_pattern = re.compile(r'(?<!\d)0*(8|16|32|64|128|256|512|1024)(?!\d)')
        powers_of_two = []

        # Extract powers of two from filenames in the folder
        for f in os.listdir(folder_path):
            match = power_of_two_pattern.search(f)
            if match:
                powers_of_two.append(int(match.group(0)))

        if not powers_of_two:
            raise ValueError("No valid power of two found in file names.")
        
        # Sort powers of two in ascending order
        sorted_powers = sorted(set(powers_of_two))

        # Extract the power of two from the current file name
        current_match = power_of_two_pattern.search(file_name)
        if not current_match:
            raise ValueError(f"File name '{file_name}' does not contain a valid power of two.")
        
        current_power = int(current_match.group(0))

        # Ensure the current power exists in the list
        if current_power not in sorted_powers:
            raise ValueError(f"Power of two {current_power} from '{file_name}' not found in folder.")

        # Determine the rank (index) of the current power of two in the sorted list
        rank = sorted_powers.index(current_power)


        # Return the block factor as 2 raised to the rank
        block_factor = 2 ** rank
        if(block_factor == 1):
            block_factor = 0

        print(f"Block Factor: {block_factor}")

        return block_factor


    def _GenerateBlankRegionMask(self, Sim, data_to_mask): # "_" indicates a protected method
        copy_image = copy.deepcopy(data_to_mask)
        #set thresholding for masking blank regions
        if(Sim):
            threshold = .05 # for simulated image
            iters = 6 # for simulated image
            kernel_size = 9 #for simulated image
        else: 
            threshold = .001 # for real image
            iters = 6 #for real image
            kernel_size = 11

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

        return  mask



    def SetBkgSub(self, Sim = False):
        try:
            mask = self._GenerateBlankRegionMask(Sim, self.BlockData)
            #copy data
            data = copy.deepcopy(self.BlockData.astype(np.float64)) #photutils should take float64
            #subtract bkg
            bkg_estimator = MedianBackground()
            bkg = Background2D(data, box_size=round(10.*self.Scalepix/2.)*2+1, coverage_mask = mask,filter_size=(3,3), bkg_estimator=bkg_estimator) #Very different RMS with mask. Minimum is MUCH larger
            data -= bkg.background #subtract bkg
            data[data < 0] = 0 #Elimate neg values. This is over estimating the background and messes up fits files


            print(f"bkg sub max: {np.max(data)}")
            #bkg sub/RMS map
            noise = bkg.background_rms
            print(f"noise max: {np.max(noise)}")
            noise[noise == 0] = 10**-3 #replace with small number...if noise = 0, we are in background, and output will be set to 0 anyway in two lines

            divRMS = data/noise
            divRMS[mask] = 0 #masked regions are zero...can change to some reasonable value but doesn't really matter for SOAX
            self.BkgSubMap = divRMS

        except ValueError:
            print("Error: Image is majoprity Black pixels. Invalid Data to reduce noise. Bkg Sub Map is set to Blocked Data. ")
            self.BkgSubMap = self.BlockData

        #save as fits if Write is true
        out_path = fr"{self.HomeDir}\BkgSubDivRMS\{self.FitsFile}_divRMS.fits"
        hdu = fits.PrimaryHDU(self.BkgSubMap, header=self.BlockHeader)
        hdu.writeto(out_path, overwrite=True)


    def ScaleBkgSub(self):
        file = fr"{self.HomeDir}\BkgSubDivRMS\{self.FitsFile}_divRMS.fits"
        with fits.open(file, mode='update') as hdul:
            image_data = hdul[0].data  # Access the image data from the PrimaryHDU

            # Ensure the data is a NumPy array
            if image_data is None:
                raise ValueError(f"No image data found in FITS file: {file}")
            
            # Normalize the image data
            topval = self._getTopVal()  # Assuming this method returns the top value for scaling
            print(f"Top Val: {topval}")
            if topval == 0:
                raise ValueError("Top value for scaling is zero, cannot divide by zero.")
            
            # Scale the image data
            hdul[0].data = np.array(image_data) * 65535 / topval
            self.BkgSubMap = hdul[0].data
            
            # Optionally, save the modified FITS file (if you are not using 'update' mode)
            hdul.flush()  # Writes changes to the file


    def _getTopVal(self):
        return np.max(self.BkgSubMap)

    def SetBlockData(self, Write = False):
        if(self.BlockFactor !=0):
            new_pixel_scale = abs(self.OrigHeader['CDELT1']) * self.BlockFactor
            self.BlockHeader['CDELT1'] = new_pixel_scale
            self.BlockHeader['CDELT2'] = new_pixel_scale

            # Apply scale factor and adjust for 0.5-pixel offset
            self.BlockHeader['CRPIX1'] = (self.OrigHeader['CRPIX1'] / self.BlockFactor) + .4687 #shift for some reason?
            self.BlockHeader['CRPIX2'] = (self.OrigHeader['CRPIX2'] / self.BlockFactor) + .4867 #shift for some reason?

            # Reproject data
            reprojected_data, _ = reproject_exact((self.OrigData, self.OrigHeader), self.BlockHeader, shape_out=(np.shape(self.BlockData)))

            # Crop NaN border (if necessary)
            self.BlockData = self._crop_nan_border(reprojected_data, (np.shape(self.BlockData)))
        else:
            self.BlockData = self.OrigData

        assert(np.sum(self.BlockData) !=0)
        print(f"Block Max: {np.max(self.BlockData)}")
        # Save a PNG for SOAX
        save_png_path = fr"{self.HomeDir}\BlockedPng\{self.FitsFile}_Blocked.png"
        pngData = self.BlockData.astype(np.uint16)
        cv2.imwrite(save_png_path, pngData)

        # Write the modified data and header to a new FITS file
        if(Write):
            out_path = fr"{self.HomeDir}\BlockedFits\{self.FitsFile}_Blocked.fits"
            hdu = fits.PrimaryHDU(self.BlockData, header=self.BlockHeader)  
            hdu.writeto(out_path, overwrite=True)



    def _crop_nan_border(self, image, expected_shape):
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
    

    def RunSoax(self):
        input_image = fr"{self.HomeDir}\BlockedPng\{self.FitsFile}_Blocked.png"
        batch = r"C:\Users\HP\Downloads\batch_soax_v3.7.0.exe"
        parameter_folder = fr"C:\Users\HP\Documents\JHU_Academics\Research\Params" 
        for parameter_file in os.listdir(parameter_folder):
            param_text = os.path.join(parameter_folder, parameter_file)
            base_param_file = os.path.splitext(parameter_file)[0]  # removes the .txt extension
            output_dir = fr"{self.HomeDir}\SOAXOutput\{self.Scale}\{base_param_file}"
            print(f"out {output_dir}")
            assert(os.path.isdir(output_dir))
            assert( os.path.isfile(param_text))
            assert(os.path.isfile(input_image))
            print("starting Soax")
            cmdString = f'"{batch}" soax -i "{input_image}" -p "{param_text}" -s "{output_dir}" --ridge 0.02 0.0075 0.06 --stretch 1.5 0.5 3' #can update ridge and stretch later
            subprocess.run(cmdString, shell=True)
            print(f"Complete Soax on {self.FitsFile}, converting set to FITS")
            self._ConvertSoaxToFits(output_dir, base_param_file)
            self.CreateComposite(base_param_file)

    def _ConvertSoaxToFits(self, outputDir, base_param_file):
        for result_file in os.listdir(outputDir):
            if(result_file.endswith('.txt')):
                result_file_base = os.path.splitext(result_file)[0]  # removes the .txt extension
                interpolate_path = fr"{self.HomeDir}\SOAXOutput\{self.Scale}\{base_param_file}\Interpolate\{result_file_base}.fits"
                result_file = fr"{self.HomeDir}\SOAXOutput\{self.Scale}\{base_param_file}\{result_file}"
                self._txtToFilaments(result_file, interpolate_path) #use new_header for WCS and blocked_data dimesnions


    def _txtToFilaments(self, result_file, interpolate_path):
        expected_header = "s p x y z fg_int bg_int"

        print(f' working on {result_file}')
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
                filament_dict[s_value].append((round(float(row['x'])), round(float(row['y'])), float(row['fg_int'])))

        self._Interpolate(filament_dict, interpolate_path)


    def _Interpolate(self, dict, path):
        wcs_orig = WCS(self.OrigHeader)
        wcs_blocked = WCS(self.BlockHeader)
        final_image = np.zeros_like(self.OrigData)

        filaments = dict.values()
        for filament in filaments:
            x_coords = []
            y_coords = []
            intensity = []
            for pixel in filament:
                x_coords.append(pixel[0])
                y_coords.append(pixel[1])
                intensity.append(pixel[2])
            world_coords = wcs_blocked.pixel_to_world(x_coords, y_coords)
            x_original, y_original = wcs_orig.world_to_pixel(world_coords)
            coordinates = [(x, y, i) for x, y, i in zip(x_original, y_original, intensity) if not (np.isnan(x) or np.isnan(y))]
            new_image = self._connect_points_sequential(coordinates, np.shape(self.OrigData))
            final_image+=new_image

        # Save the output image as a FITS file
        hdu = fits.PrimaryHDU(final_image, header = self.OrigHeader)
        hdu.writeto(path, overwrite=True)


    def _connect_points_sequential(self, points, image_shape):
        points = [(int(x), int(y), i) for x, y,i in points]
        output_array = np.zeros(image_shape, dtype=np.uint8)

        # Draw lines between consecutive points
        for i in range(len(points) - 1):
            x1, y1 = (points[i][0], points[i][1])
            x2, y2 = (points[i+1][0], points[i+1][1])
            cv2.line(output_array, (x1, y1), (x2, y2), 1, thickness= 1)
            cv2.circle(output_array, (x1, y1), 1, 1, thickness= 1)  # Filled circle
        return output_array
    
    def CreateComposite(self, base_param_file):
        output_directory = fr"{self.HomeDir}\Composite"
        directory = fr"{self.HomeDir}\SOAXOutput\{self.Scale}\{base_param_file}\Interpolate"
        output_name = fr"{self.FitsFile}_Composite"
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
        print(f"Total number of images: {max_presence}")

        # Normalize composite data to the range [0, 255] for grayscale representation
        composite_data = (composite_data / max_presence * 255).astype(np.uint8)

        # Save the grayscale composite as FITS
        output_fits_path = os.path.join(output_directory, output_name + ".fits")
        hdu = fits.PrimaryHDU(data=composite_data, header=header)
        hdu.writeto(output_fits_path, overwrite=True)
        self.Composite = composite_data
        self.ProbabilityMap = composite_data
        print(f"Composite image saved as FITS: {output_fits_path}")

    def SetIntensityMap(self, Orig = True):
        if(Orig):
            self.IntensityMap[self.Composite !=0] = 255*self.OrigData[self.Composite!=0]
        else: 
            self.IntensityMap[self.Composite !=0] = self.BkgSubMap[self.Composite!=0]

    def DisplayProbIntensityPlot(self, galaxy_dir, Orig=True, Write=False, verbose=False):

        # Flatten the images
        probability_flat = self.ProbabilityMap.flatten()
        probability_flat = np.round(100 / 255 * probability_flat).astype(int)
        intensity_flat = self.IntensityMap.flatten()

        # Remove entries where the probability equals zero
        valid_indices = probability_flat != 0
        probability_flat = probability_flat[valid_indices]
        intensity_flat = intensity_flat[valid_indices]

        # Identify unique probabilities
        unique_probabilities = np.unique(probability_flat)

        # Group intensity values by probability
        grouped_intensities = [intensity_flat[probability_flat == prob] for prob in unique_probabilities]

        # Count values in each group
        counts = [len(group) for group in grouped_intensities]

        # Compute the total number of elements across all box plots
        total_elements = sum(counts)
        print(f"Total number of elements across all box plots: {total_elements}")

        # Create box plots
        plt.figure(figsize=(10, 6))
        plt.boxplot(grouped_intensities, labels=unique_probabilities, vert=True, patch_artist=True)

        # Add count annotations below x-axis labels
        ylim = plt.ylim()
        y_min = ylim[0]
        annotation_y = y_min - 0.1 * (ylim[1] - ylim[0])  # Place annotations slightly below the x-axis labels
        for i, count in enumerate(counts, start=1):
            plt.text(i, annotation_y, f'{count}', ha='center', va='top', fontsize=9, color='blue')

        # Adjust y-axis to leave space for text annotations
        plt.ylim(y_min - 0.2 * (ylim[1] - ylim[0]), ylim[1])

        # Customize plot
        plt.xlabel('Probability (%) from Soax')
        if Orig:
            plt.ylabel('Intensity value in original data')
        else:
            plt.ylabel('Intensity value in bkg subtracted data')
        plt.title(f'Box Plot of Intensity Values at Each Probability for {self.Galaxy} at {self.Scale} for {total_elements} pixels')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save or display the plot
        plt.tight_layout()
        if Write:
            plt.savefig(f"{galaxy_dir}/Figures/ProbIntensityPlot_{self.Galaxy}_{self.Scale}_{Orig}.png")
        if verbose:
            plt.show()




    def BlurComposite(self, set_blur_as_prob = True):
        struct_width = self.Scale.replace('pc',"")
        struct_width = int(struct_width)
        # Convert structure width from parsecs to pixels
        structure_width_pixels = struct_width / self.Scalepix
        
        # Calculate sigma for Gaussian convolution
        sigma = structure_width_pixels / 2.355  # FWHM = 2.355 * sigma -> sigma = FWHM / 2.355
        print(f"Blurring with sigma value: {sigma}")
        # Apply Gaussian blur
        blurred_image = gaussian_filter(self.Composite, sigma=sigma)

        self.Composite = blurred_image
        if(set_blur_as_prob):
            self.ProbabilityMap = blurred_image

    def ReHashComposite(self, ProbabilityThreshPercentile, minPixBoxSize):
        print(f"Blur Max: {np.max(self.Composite)}")
        ProbabilityThresh = np.max(self.Composite)*ProbabilityThreshPercentile
        ret, threshComposite = cv2.threshold(self.Composite,ProbabilityThresh,255, cv2.THRESH_BINARY)
        skelComposite = skeletonize(threshComposite)
        junctions = AF.getSkeletonIntersection(np.array(255*skelComposite))
        print(f"Number of junctions: {len(junctions)}")
        IntersectsRemoved = AF.removeJunctions(junctions, skelComposite, dot_size = 3)
        labels, stats, num_labels = AF.identify_connected_components(np.array(IntersectsRemoved))
        small_areas = AF.sort_label_id(num_labels, stats, minPixBoxSize)
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

        self.Composite = img

        output_directory = fr"{self.HomeDir}\Composite"
        output_name = fr"{self.FitsFile}_Composite_{ProbabilityThreshPercentile}"
        output_fits_path = os.path.join(output_directory, output_name + '.fits')
        hdu = fits.PrimaryHDU(data=img, header=self.OrigHeader)
        hdu.writeto(output_fits_path, overwrite=True)
    
    def getGalaxy(self):
        return self.Galaxy   
             
    def getBkgSubMap(self):
        return self.BkgSubMap  
    
    def getScale(self):
        return self.Scale  
                                
