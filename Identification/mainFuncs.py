#Functions not associated with a filament map object but used within the main block of FilPHANGS

import FilamentMap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.table import Table
import astropy.units as u  # Add this line to import the units

matplotlib.use('Agg')


def getInfo(label, csv_path):

    """
    Read the csv file for information about an image and return the distance, res, pixscale, and powers of 2

    Parameter
    - label (str): The label for the celestial object that the image is of, commonly a galaxy name
    - csv_path (str): The path to the csv file containing relevant information

    Returns:
    - distance (float): distance to the image
    - res (float): angular resolution
    - pixscale (float): pixel resolution
    - min_power (float): minimum power of 2 for scale decomposition
    - max_power (float): maximum power of 2 for scale decomposition

    """

    table = pd.read_excel(csv_path)

    try: 
        label_info = table[table['label'].str.lower() == label.lower()]
    except KeyError as e:
        print("Error: Cannot find 'label' in csv file")
        exit(1)

    if not label_info.empty:
        distance = label_info.iloc[0]['current_dist']
        res = label_info.iloc[0]['res']
        pixscale = label_info.iloc[0]['pixscale']
        min_power = label_info.iloc[0]['Power of 2 min']
        max_power = label_info.iloc[0]['Power of 2 max']
        return distance, res, pixscale, min_power, max_power
    else: 
        print("Galaxy not found in csv!")


 
def setUpGalaxy(base_dir, label_folder_path,  label, distance_Mpc, res, pixscale, param_file_path, noise_min, flatten_perc): 

    """
    Constructs a filament map object for each scale decomposed image of a label/celestial object. 
    Sets the blocked data and signal to noise image used in the SOAX algorithm. 

    Parameters:
    - base_dir (str): Base directory for all FilPHANGS files
    - label_folder_path (str): path to the folder associated with the specified label/celestial object
    - label (str): label of the desired celestial object
    - distance_Mpc (float): Distance in Mega Parsecs to the celestial object
    - res (float): angula resolution associated with the image
    - pixscale (str): The pixel level resolution associated with the image
    - param_file_path (float): Path to the file containing the soax parameters
    - noise_min (float): minimum noise to be considered realistic in the image
    - flatten_perc (str): Percentage to use in the arctan transform
    
    Returns:
    - FilamentMapList (Filament Map): returns a list of the filament map objects for each scale of an image. 
   
    """
        
    FilamentMapList = []

    CDD_folder = os.path.join(label_folder_path, "CDD")

    for fits_file in os.listdir(CDD_folder): #iterate through CDD folder to create filament map objects for each scale decomposed image

        if(fits_file.endswith(".fits")): 
            ScalePix = pixscale * 4.848 * distance_Mpc  #convert to parcecs per pixel
            filMap = FilamentMap.FilamentMap(ScalePix, base_dir, label_folder_path, fits_file, label, param_file_path, flatten_perc) #create object
            filMap.setBlockData() #set the blocked data
            filMap.setBkgSubDivRMS(noise_min) #set the background subtracted and noise divided data
            FilamentMapList.append(filMap) 

    return FilamentMapList



def CreateSNRPlot(FilamentMapList, base_dir, percentile, write = False):

    """
    Create a plot of the Signal to noise ratio in an image before scaling the background subtracted and nosie divided image. 

    Parameters:
    - FilamentMapList (Filament Map): List of Scale ecomposed Filament Maps associated with a single label
    - base_dir (str): path to the base directory
    - percentile (float): percentile to create the SNR plot from
    - write (bool): Boolean to indicate whether or not the plot should be saved 
    """

    label_dict = {}

    for filMap in FilamentMapList:  # Iterate over each object, extract the needed data, and append to label_dict
        label = filMap.getLabel()

        if label not in label_dict:
            label_dict[label] = []

        SNRMap = filMap.getBkgSubDivRMSMap() 
        scale = filMap.getScale()
        scale = scale.replace('pc', "")
        scale = float(scale)
        label_dict[label].append((scale, np.percentile(SNRMap, percentile))) 

    # Create scatter plot with points from each scale decomposed image
    for label, data in label_dict.items():
        scales, percentiles = zip(*data)  # Unpack scales and percentiles
        plt.figure()
        plt.scatter(scales, percentiles, label= f"Celestial Object: {label}")
        plt.xlabel("Scale (pc)")
        plt.ylabel(f"SNR {percentile} percentile")
        plt.title(f"SNR Plot for Galaxy: {label} without normalization and using unique masks")
        plt.legend()
        plt.grid(True)

    if write:
        plt.savefig(f"{base_dir}\Figures\SNRPlot_{label}.png")
    plt.close()


def createDirectoryStructure(base_directory, csv_path):

    """
    Creates the directiry structure as described in the ReadME. Subfolders are created on images present in the "OriginalImages" fodler. 

    Parameters:
    - base_directory (str): path to the base directory for which all subfolders and files will be held
    - csv_path (str): Path to the csv file containing image information
    """

    folder_path = os.path.join(base_directory, "OriginalImages")
    os.makedirs(base_directory, exist_ok=True)  # Ensure the root directory exists
    figures_folder = os.path.join(base_directory, "Figures") #make the figures folder
    os.makedirs(figures_folder, exist_ok=True)

    # Iterate through all FITS files in the folder and create the needed subfolders
    for filename in os.listdir(folder_path):

        if filename.endswith('.fits'):
            # Extract the galaxy name using regex
            match = re.match(r"(.*?)_.*?\.fits", filename)

            if match:
                label = match.group(1)
                # Create the galaxy folder
                label_folder = os.path.join(base_directory, label)
                os.makedirs(label_folder, exist_ok=True)

                # Create subfolders
                subfolders = [
                    "CDD", "Composites", "BlockedPng", "SyntheticMap", "SoaxOutput", "BkgSubDivRMS"
                ]
                _, _, _, min_power, max_power = getInfo(label, csv_path) #get relevant information for image

                soax_subfolders = []
                for i in range(min_power, max_power + 1):
                    soax_subfolders.append(str(2**i).lstrip("0") + "pc")

                for subfolder in subfolders:
                    subfolder_path = os.path.join(label_folder, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)

                    # Create SOAXOutput subfolders
                    if subfolder == "SoaxOutput":
                        for soax_subfolder in soax_subfolders:
                            os.makedirs(os.path.join(subfolder_path, soax_subfolder), exist_ok=True)

                print(f"Directory structure created for galaxy: {label}")



def clearAllFiles(base_directory, csv_path, param_file_path):

    """
    Clears all files in subfolders under the specified base directory,
    but keeps files directly in the base directory and files in the "originalImages" folder untouched.

    Parameters:
    - base_directory (str): Path to the base directory to clear.
    - csv_path (str): Path to the CSV file to exclude from deletion.
    - param_file_path (str): Path to the parameter file to exclude from deletion.
    """

    # Walk through all directories and files
    for foldername, subfolders, filenames in os.walk(base_directory):
        # Skip the root directory itself (no files will be deleted here)
        if foldername == base_directory:
            continue

        # Skip the "originalImages" folder and its contents
        if "originalimages" in foldername.lower():  # Ensures case-insensitive check
            continue
        
        # Delete files in subdirectories
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            
            # Check if the file is not the CSV or parameter file, and ensure it's not in the "originalImages" folder
            if file_path != csv_path and file_path != param_file_path:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

    print("All files cleared from subdirectories of the directory structure.")



def renameFitsFiles(base_dir, csv_path):

    """
    Renames FITS files based on information from an Excel file. Forces naming convention discussed in the ReadMe. 

    Parameters:
    - base_dir (str): Path to the base directory containing the FITS files.
    - csv_path (str): Path to the Excel file containing image information.
    """

    # Load the Excel file into a DataFrame
    table = pd.read_excel(csv_path)
    
    fits_file_folder_path = os.path.join(base_dir, "OriginalImages")

    for fits_file in os.listdir(fits_file_folder_path):
        # Construct the full path to the current FITS file
        full_file_path = os.path.join(fits_file_folder_path, fits_file)

        # Extract the galaxy name from the FITS filename (assumes galaxy is the first part before the underscore)
        filename = os.path.basename(fits_file)
        galaxy_match = re.match(r'([^_]+)', filename)
        
        if galaxy_match:
            galaxy_name = galaxy_match.group(1)
            
            # Find the row in the Excel file corresponding to the galaxy

            try: 
                galaxy_info = table[table['label'].str.lower() == galaxy_name.lower()]
            except KeyError as e:
                print("Error: Cannot find 'label' in csv file")
                exit(1)
    
            if not galaxy_info.empty:
                telescope = galaxy_info.iloc[0]['Telescope']
                band = galaxy_info.iloc[0]['Band']
                
                # Check if 'starsub' is in the original filename
                if "starsub" in filename.lower():
                    new_filename = f"{galaxy_name}_{telescope}_{band}_starsub.fits"
                else:
                    new_filename = f"{galaxy_name}_{telescope}_{band}.fits"

                new_filepath = os.path.join(fits_file_folder_path, new_filename)

                # Rename the file (ensure the full paths are used)
                os.rename(full_file_path, new_filepath)
                print(f"Renamed {filename} to {new_filename}")
            else:
                print(f"Galaxy '{galaxy_name}' not found in Excel file.")
        else:
            print(f"Could not extract galaxy name from {filename}")

    print("Renaming process completed.")



