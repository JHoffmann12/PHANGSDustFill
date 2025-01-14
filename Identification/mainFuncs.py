import os
import FilamentMap
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import re
from astropy.io import fits
from astropy.coordinates import EarthLocation
import astropy.units as u  # Add this line to import the units
import pandas as pd
matplotlib.use('Agg')

def getInfo(Galaxy,  csv_path):
    table = pd.read_excel(csv_path)
    galaxy_info = table[table['galaxy'].str.lower() == Galaxy.lower()]

    if not galaxy_info.empty:
        distance = galaxy_info.iloc[0]['current_dist']
        res = galaxy_info.iloc[0]['res']
        pixscale = galaxy_info.iloc[0]['pixscale']
        min_power = galaxy_info.iloc[0]['Power of 2 min']
        max_power = galaxy_info.iloc[0]['Power of 2 max']
        return distance, res, pixscale, min_power, max_power
    else: 
        print("Galaxy not found in csv!")

    # Check if galaxy exists in the distance table
    if Galaxy in table['galaxy']:
        # Extract the full row for the galaxy from the distance table
        galaxy_row = table[table['galaxy'] == Galaxy]
        print(f"Galaxy {Galaxy} found in distance table:")
        return galaxy_row
    else:
        print(f"Galaxy name '{Galaxy}' not found in the distance table.")
        exit()
 
def setUpGalaxy(base_dir, galaxy_folder_path,  Galaxy, distance_Mpc, res, pixscale, param_file_path): #Assumes no data is a simulation for now
    FilamentMapList = []
    CDD_folder = os.path.join(galaxy_folder_path, "CDD")
    for fits_file in os.listdir(CDD_folder):
        if(fits_file.endswith(".fits")): 
            ScalePix = pixscale * (1/206265) * distance_Mpc * 10**6
            MyFilMap = FilamentMap.FilamentMap(ScalePix, base_dir, galaxy_folder_path, fits_file, Galaxy, param_file_path) #change name later to be specific to the image
            # MyFilMap.setBlockFactor(4)
            MyFilMap.SetBlockData()
            #get BkgSub Image
            MyFilMap.SetBkgSub()
            #Block the image
            FilamentMapList.append(MyFilMap)
    return FilamentMapList

def CreateSNRPlot(FilamentMapList, galaxy_dir, Write = False, verbose = False):

    galaxy_dict = {}
    # Iterate over each object and group them
    for myFilMap in FilamentMapList:
        galaxy = myFilMap.getGalaxy()
        if galaxy not in galaxy_dict:
            galaxy_dict[galaxy] = []
        SNRMap = myFilMap.getBkgSubMap()
        #to add to plot:
        scale = myFilMap.getScale()
        scale = scale.replace('pc', "")
        scale = int(scale)
        galaxy_dict[galaxy].append((scale, np.percentile(SNRMap, 99.99))) 

    # Plotting scatter plots for each galaxy
    for galaxy, data in galaxy_dict.items():
        scales, percentiles = zip(*data)  # Unpack scales and percentiles
        plt.figure()
        plt.scatter(scales, percentiles, label=f"Galaxy: {galaxy}")
        plt.xlabel("Scale (pc)")
        plt.ylabel("SNR 99.99 percentile")
        plt.title(f"SNR Plot for Galaxy: {galaxy} without normalization and using unique masks")
        plt.legend()
        plt.grid(True)

        if Write:
            plt.savefig(f"{galaxy_dir}\Figures\SNRPlot_{galaxy}.png")
        if(verbose):
            plt.show()


def create_directory_structure(root_directory, csv_path):
    """
    Creates a directory structure based on FITS file names.

    Parameters:
    - folder_path (str): Path to the folder containing FITS files.
    - root_directory (str): Path to the root directory where folders will be created.
    """
    folder_path = os.path.join(root_directory, "OriginalImages")
    # Ensure the root directory exists
    os.makedirs(root_directory, exist_ok=True)
    figures_folder = os.path.join(root_directory, "Figures")
    os.makedirs(figures_folder, exist_ok=True)

    # Iterate through all FITS files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.fits'):
            # Extract the galaxy name using regex
            match = re.match(r"(.*?)_.*?\.fits", filename)
            if match:
                galaxy_name = match.group(1)
                # Create the galaxy folder
                galaxy_folder = os.path.join(root_directory, galaxy_name)
                os.makedirs(galaxy_folder, exist_ok=True)

                # Create subfolders
                subfolders = [
                    "CDD", "Composites", "BlockedPng", "SyntheticMap", "SoaxOutput", "BkgDivRMS"
                ]
                _, _, _, min_power, max_power = getInfo(galaxy_name, csv_path) #get relevant information for image

                soax_subfolders = []
                for i in range(min_power, max_power + 1):
                    soax_subfolders.append(str(2**i).lstrip("0") + "pc")

                for subfolder in subfolders:
                    subfolder_path = os.path.join(galaxy_folder, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)

                    # Create SOAXOutput subfolders
                    if subfolder == "SoaxOutput":
                        for soax_subfolder in soax_subfolders:
                            os.makedirs(os.path.join(subfolder_path, soax_subfolder), exist_ok=True)

                print(f"Directory structure created for galaxy: {galaxy_name}")

def clear_all_files(root_directory, csv_path, param_file_path):
    """
    Clears all files in subfolders under the specified root directory,
    but keeps files directly in the root directory and files in the "originalImages" folder untouched.

    Parameters:
    - root_directory (str): Path to the root directory to clear.
    - csv_path (str): Path to the CSV file to exclude from deletion.
    - param_file_path (str): Path to the parameter file to exclude from deletion.
    """
    # Walk through all directories and files
    for foldername, subfolders, filenames in os.walk(root_directory):
        # Skip the root directory itself (no files will be deleted here)
        if foldername == root_directory:
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

def rename_fits_files(base_dir, xlsx_path):
    """
    Renames FITS files based on information from an Excel file.

    Parameters:
    - base_dir (str): Path to the base directory containing the FITS files.
    - xlsx_path (str): Path to the Excel file containing galaxy information.
    """
    import os
    import pandas as pd
    import re

    # Load the Excel file into a DataFrame
    table = pd.read_excel(xlsx_path)
    
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
            galaxy_info = table[table['galaxy'].str.lower() == galaxy_name.lower()]

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



def decomposition_exists(root_path):
    """
    Checks if the "CDD" folder in the specified path is empty or not.

    Parameters:
    - root_path (str): The root directory to check.

    Returns:
    - bool: True if the folder is not empty, False if it is empty or doesn't exist.
    """
    # Construct the CDD folder path
    cdd_path = os.path.join(root_path, "CDD")

    # Check if the folder exists
    if not os.path.exists(cdd_path):
        print(f"The folder 'CDD' does not exist in {root_path}.")
        return False

    # Check if the folder is empty
    if not os.listdir(cdd_path):
        print(f"The folder 'CDD' is empty in {root_path}.")
        return False
    else:
        print(f"The folder 'CDD' is not empty in {root_path}.")
        return True