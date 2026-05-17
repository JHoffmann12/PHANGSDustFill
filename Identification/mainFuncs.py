import logging
from pathlib import Path
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
import astropy.units as u

matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def getMJysr(bandstr, inststr):
    if bandstr=='F770W': sigma_MJysr=0.11 # JWST Cycle 1 imaging 1-sigma surface brightness sensitivity from Lee+23 (JWST survey paper) with units of MJy/sr
    if bandstr=='F1000W': sigma_MJysr=0.12
    if bandstr=='F1130W': sigma_MJysr=0.15
    if bandstr=='F2100W': sigma_MJysr=0.25
    if bandstr=='F360M': sigma_MJysr=0.58
    if bandstr=='F335M': sigma_MJysr=0.42
    if bandstr=='F300M': sigma_MJysr=0.45
    if bandstr=='F200W': sigma_MJysr=0.68
    if bandstr=='F187N': sigma_MJysr=1.00 #TBD using Cycle 2, also do the others above change for Cycle 2??
    if bandstr=='F150W': sigma_MJysr=1.00 #TBD using Cycle 2
    else:
        sigma_MJysr = 0
    return sigma_MJysr 


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

    logger.info('Processing label: %s', label)
    
    # Check for any hidden characters or extra whitespace
    label_clean = label.strip()
    if label != label_clean:
        logger.warning("Label had whitespace. Original: '%s', Cleaned: '%s'", label, label_clean)
        label = label_clean
    
    table = pd.read_excel(csv_path)
    
    # Split on underscore
    parts = label.split("_")
    if len(parts) < 2:
        print(f"Error: Label '{label}' doesn't contain an underscore separator")
        return None
    
    # Take everything except the last part as the label, last part as band
    band = parts[-1]
    label_name = "_".join(parts[:-1])
    
    logger.debug("Searching for: label='%s', band='%s'", label_name, band)
    
    logger.debug("Available labels in Excel: %s", table['label'].unique())
    logger.debug("Available bands in Excel: %s", table['Band'].unique())

    try: 
        label_info = table[
            (table['label'].str.strip().str.lower() == label_name.lower()) & 
            (table['Band'].str.strip().str.lower() == band.lower())
        ]
    except KeyError as e:
        logger.error("Cannot find required columns in Excel file: %s. Available columns: %s", e, table.columns.tolist())
        exit(1)

    if not label_info.empty:
        logger.info("Found match: %s_%s", label_name, band)
        distance = label_info.iloc[0]['current_dist']
        res = label_info.iloc[0]['res']
        pixscale = label_info.iloc[0]['pixscale']
        min_power = label_info.iloc[0]['Power of 2 min']
        max_power = label_info.iloc[0]['Power of 2 max']
        Rem_sources = label_info.iloc[0]['Rem_sources']
        Band = label_info.iloc[0]['Band']
        Instr = label_info.iloc[0]['INSTR']

        try: 
            sSFR = label_info.iloc[0]['SSFR']
            inclination = label_info.iloc[0]['Inclination Angle']
        except: 
            sSFR = np.nan
            inclination = np.nan

        if bool(Rem_sources):
            MJysr = getMJysr(Band, Instr)
        else:
            MJysr = np.nan #not needed

        return distance, res, pixscale, MJysr, Band, min_power, max_power, bool(Rem_sources), sSFR, inclination
    
    else: 
        logger.error("Image '%s' with band '%s' not found in Excel file. Check label and Band columns.", label_name, band)
        return None


def getMinSnakeLengthFromAspectRatio(min_aspect_ratio, ref_scale_pc=16.0, ref_scalepix=5.25):
    """
    Convert a minimum aspect ratio into a SOAX minimum snake length.

    Filament width in pixels = Scale / ScalePix (16 pc / ScalePix at the minimum CDD scale).
    min_snake_length = round(min_aspect_ratio * width).

    The reference ScalePix of 5.25 pc/px is representative of the PHANGS F770W dataset at 16 pc.
    With the default ratio of 8.2, this returns 25 px — matching the previous hardcoded value.

    Parameters:
    - min_aspect_ratio (float): desired minimum length-to-width ratio for a valid filament
    - ref_scale_pc (float): minimum CDD scale in parsecs (default 16)
    - ref_scalepix (float): reference parsecs-per-pixel at that scale (default 5.25)

    Returns:
    - min_snake_length_ss (int): minimum snake length in pixels for SOAX at the shortest scale
    """
    ref_width_pix = ref_scale_pc / ref_scalepix
    min_snake_length_ss = round(min_aspect_ratio * ref_width_pix)
    logger.info(
        "min_aspect_ratio=%.2f → ref_width=%.2f px → min_snake_length_ss=%d",
        min_aspect_ratio, ref_width_pix, min_snake_length_ss,
    )
    return min_snake_length_ss


def setUpGalaxy(base_dir, label_folder_path,  label, distance_Mpc, res, pixscale, param_file_path, noise_min, flatten_perc, min_intensity, sSFR, Inclination): 

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
    - min_intensity (float): Minimum intensity in original image for valid pixel
    
    Returns:
    - FilamentMapList (Filament Map): returns a list of the filament map objects for each scale of an image. 
   
    """
        
    FilamentMapList = []

    CDD_folder = os.path.join(label_folder_path, "CDD")

    for fits_file in os.listdir(CDD_folder): #iterate through CDD folder to create filament map objects for each scale decomposed image

        if(fits_file.endswith(".fits")): 
            ScalePix = pixscale * 4.848 * distance_Mpc  #convert to parcecs per pixel
            filMap = FilamentMap.FilamentMap(ScalePix, base_dir, label_folder_path, fits_file, label, param_file_path, flatten_perc, min_intensity, sSFR, Inclination) #create object
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
        plt.savefig(Path(f"{base_dir}/Figures/SNRPlot_{label}.png"))
    plt.close()



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
                logger.debug("Deleted file: %s", file_path)

    logger.info("All files cleared from subdirectories of the directory structure.")


def createDirectoryStructure(base_directory, csv_path, ID_set=False):
    """
    Creates the directory structure as described in the ReadME. Subfolders are created based on images present in the "OriginalImages" folder.

    Parameters:
    - base_directory (str): Path to the base directory for which all subfolders and files will be held.
    - csv_path (str): Path to the CSV file containing image information.
    - ID_set (bool): If True, an image ID is extracted from the filename as the string after the final
                     underscore (before the extension) and appended to the galaxy label folder name.
                     If False, behaviour is identical to the original.
    """

    folder_path = os.path.join(base_directory, "OriginalImages")
    os.makedirs(base_directory, exist_ok=True)
    figures_folder = os.path.join(base_directory, "Figures")
    os.makedirs(figures_folder, exist_ok=True)

    for filename in os.listdir(folder_path):

        if filename.endswith('.fits'):
            match = re.match(r"(.+?)_(F\d+[A-Z])[_.]", filename)

            if match:
                label_name = match.group(1)
                band       = match.group(2)
                label      = f"{label_name}_{band}"

                # Extract image ID if requested
                if ID_set:
                    stem     = os.path.splitext(filename)[0]   # strip .fits
                    image_id = stem.rsplit('_', 1)[-1]          # last token
                    folder_label = f"{label}_{image_id}"
                else:
                    folder_label = label

                logger.info("Processing file: %s (label=%s%s)", filename, label, f"  ID: {image_id}" if ID_set else "")

                # Create the galaxy folder (with or without ID suffix)
                label_folder = os.path.join(base_directory, folder_label)
                os.makedirs(label_folder, exist_ok=True)

                info_result = getInfo(label, csv_path)

                if info_result is None:
                    logger.warning("Skipping %s — not found in Excel file", label)
                    continue

                _, _, _, _, _, min_power, max_power, _, _, _ = info_result

                subfolders = [
                    "CDD", "Composites", "BlockedPng", "SyntheticMap",
                    "SoaxOutput", "BkgSubDivRMS", "Source_Removal"
                ]

                soax_subfolders = [
                    str(2**i) + "pc"
                    for i in range(int(min_power), int(max_power) + 1)
                ]

                for subfolder in subfolders:
                    subfolder_path = os.path.join(label_folder, subfolder)
                    os.makedirs(subfolder_path, exist_ok=True)

                    if subfolder == "SoaxOutput":
                        for soax_subfolder in soax_subfolders:
                            os.makedirs(os.path.join(subfolder_path, soax_subfolder), exist_ok=True)

                    if subfolder == "Source_Removal":
                        os.makedirs(os.path.join(subfolder_path, "CDD_Pix"),        exist_ok=True)
                        os.makedirs(os.path.join(subfolder_path, "Source_Tables"),  exist_ok=True)

                logger.info("Directory structure created for: %s", folder_label)
            else:
                logger.warning("Could not parse filename: %s", filename)


def renameFitsFiles(base_dir, csv_path, ID_set=False):
    """
    Renames FITS files based on information from an Excel file. Forces naming
    convention discussed in the ReadMe.

    Parameters:
    - base_dir (str): Path to the base directory containing the FITS files.
    - csv_path (str): Path to the Excel file containing image information.
    - ID_set (bool): If True, the image ID (string after the final underscore,
                     before the extension) is preserved and appended to the new
                     filename.  If False, behaviour is identical to the original.
    """

    table = pd.read_excel(csv_path)
    fits_file_folder_path = os.path.join(base_dir, "OriginalImages")

    for fits_file in os.listdir(fits_file_folder_path):
        full_file_path = os.path.join(fits_file_folder_path, fits_file)
        filename       = os.path.basename(fits_file)

        match = re.match(r"([^_]+)_([^_]+)", filename)

        if not match:
            print(f"Could not extract galaxy name from {filename}")
            continue

        label = match.group(1)
        band  = match.group(2)

        # Extract image ID if requested
        if ID_set:
            stem     = os.path.splitext(filename)[0]
            image_id = stem.rsplit('_', 1)[-1]
        else:
            image_id = None

        try:
            label_info = table[
                (table['label'].str.lower() == label.lower()) &
                (table['Band'].str.lower()  == band.lower())
            ]
        except KeyError:
            logger.error("Cannot find 'label' column in Excel file")
            exit(1)

        if not label_info.empty:
            telescope = label_info.iloc[0]['Telescope']
            band      = label_info.iloc[0]['Band']
            img_type  = label_info.iloc[0]['Image_Type']

            # Build base new name
            if "starsub" in filename.lower():
                base_name = f"{label}_{band}_{telescope}_{img_type}_starsub"
            else:
                base_name = f"{label}_{band}_{telescope}_{img_type}"

            # Append ID if present
            if image_id:
                new_filename = f"{base_name}_{image_id}.fits"
            else:
                new_filename = f"{base_name}.fits"

            new_filepath = os.path.join(fits_file_folder_path, new_filename)
            os.rename(full_file_path, new_filepath)
            logger.info("Renamed %s → %s", filename, new_filename)
        else:
            logger.warning("Not found in Excel file: %s / %s", label, band)

    logger.info("Renaming process completed.")
