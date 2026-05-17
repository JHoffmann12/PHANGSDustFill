import logging
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import FilamentMap

matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def getMJysr(bandstr, inststr):
    """Return the JWST Cycle 1 1-sigma surface brightness sensitivity in MJy/sr for a given band.

    Values are from Lee et al. 2023 (JWST survey paper). Cycle 2 values for F187N and F150W
    are placeholders pending updated measurements.

    Parameters
    ----------
    - bandstr (str): Filter name, e.g. 'F770W'.
    - inststr (str): Instrument name (unused; reserved for future multi-instrument support).

    Returns
    -------
    - sigma_MJysr (float): 1-sigma sensitivity in MJy/sr, or 0 if the band is not recognised.
    """
    band_sensitivity = {
        'F770W':  0.11,
        'F1000W': 0.12,
        'F1130W': 0.15,
        'F2100W': 0.25,
        'F360M':  0.58,
        'F335M':  0.42,
        'F300M':  0.45,
        'F200W':  0.68,
        'F187N':  1.00,  # TBD from Cycle 2 data
        'F150W':  1.00,  # TBD from Cycle 2 data
    }
    return band_sensitivity.get(bandstr, 0)


def getInfo(label, csv_path):
    """Read per-image metadata from the Excel catalogue and return processing parameters.

    Parameters
    ----------
    - label (str): Image label in the form 'galaxyname_BAND', e.g. 'NGC0628_F770W'.
    - csv_path (str or Path): Path to the ImageData.xlsx catalogue.

    Returns
    -------
    - distance_Mpc (float): Distance to the galaxy in megaparsecs.
    - res (float): Angular resolution in arcseconds.
    - pixscale (float): Pixel scale in arcseconds per pixel.
    - MJysr (float): 1-sigma surface brightness sensitivity in MJy/sr (NaN if source removal disabled).
    - Band (str): Filter name.
    - min_power (int): log2 of the minimum CDD decomposition scale in parsecs.
    - max_power (int): log2 of the maximum CDD decomposition scale in parsecs.
    - Rem_sources (bool): Whether compact source removal should be applied.
    - sSFR (float): Specific star formation rate (NaN if not in catalogue).
    - inclination (float): Galaxy inclination angle in degrees (NaN if not in catalogue).
    """
    logger.info('Processing label: %s', label)

    label = label.strip()
    table = pd.read_excel(csv_path)

    parts = label.split("_")
    if len(parts) < 2:
        logger.error("Label '%s' does not contain an underscore separator", label)
        return None

    band       = parts[-1]
    label_name = "_".join(parts[:-1])

    logger.debug("Searching for: label='%s', band='%s'", label_name, band)
    logger.debug("Available labels in Excel: %s", table['label'].unique())
    logger.debug("Available bands in Excel: %s", table['Band'].unique())

    try:
        label_info = table[
            (table['label'].str.strip().str.lower() == label_name.lower()) &
            (table['Band'].str.strip().str.lower()  == band.lower())
        ]
    except KeyError as e:
        logger.error("Cannot find required columns in Excel file: %s. Available: %s", e, table.columns.tolist())
        exit(1)

    if label_info.empty:
        logger.error("Image '%s' with band '%s' not found in Excel file.", label_name, band)
        return None

    logger.info("Found match: %s_%s", label_name, band)
    distance  = label_info.iloc[0]['current_dist']
    res       = label_info.iloc[0]['res']
    pixscale  = label_info.iloc[0]['pixscale']
    min_power = label_info.iloc[0]['Power of 2 min']
    max_power = label_info.iloc[0]['Power of 2 max']
    Rem_sources = label_info.iloc[0]['Rem_sources']
    Band      = label_info.iloc[0]['Band']
    Instr     = label_info.iloc[0]['INSTR']

    try:
        sSFR        = label_info.iloc[0]['SSFR']
        inclination = label_info.iloc[0]['Inclination Angle']
    except Exception:
        sSFR        = np.nan
        inclination = np.nan

    MJysr = getMJysr(Band, Instr) if bool(Rem_sources) else np.nan

    return distance, res, pixscale, MJysr, Band, min_power, max_power, bool(Rem_sources), sSFR, inclination


def getMinSnakeLengthFromAspectRatio(min_aspect_ratio, ref_scale_pc=16.0, ref_scalepix=5.25):
    """Convert a minimum aspect ratio into the SOAX minimum snake length at the shortest CDD scale.

    Filament width in pixels equals the scale divided by the pixel scale (16 pc / ScalePix).
    The reference pixel scale of 5.25 pc/px is representative of PHANGS F770W at 16 pc; with
    the default ratio of 8.2 this returns 25 pixels, matching the previously hardcoded value.

    Parameters
    ----------
    - min_aspect_ratio (float): Minimum length-to-width ratio for a valid filament.
    - ref_scale_pc (float): Minimum CDD scale in parsecs (default 16).
    - ref_scalepix (float): Reference parsecs per pixel at that scale (default 5.25).

    Returns
    -------
    - min_snake_length_ss (int): Minimum snake length in pixels for SOAX at the shortest scale.
    """
    ref_width_pix      = ref_scale_pc / ref_scalepix
    min_snake_length_ss = round(min_aspect_ratio * ref_width_pix)
    logger.info("min_aspect_ratio=%.2f → ref_width=%.2f px → min_snake_length_ss=%d", min_aspect_ratio, ref_width_pix, min_snake_length_ss)
    return min_snake_length_ss


def setUpGalaxy(base_dir, label_folder_path, label, distance_Mpc, pixscale,
                param_file_path, noise_min, flatten_perc, min_intensity, sSFR, Inclination):
    """Construct a FilamentMap object for each scale-decomposed image of a galaxy.

    Parameters
    ----------
    - base_dir (str or Path): Base directory for all FilPHANGS output files.
    - label_folder_path (str or Path): Path to the output folder for this galaxy.
    - label (str): Galaxy label, e.g. 'NGC0628_F770W'.
    - distance_Mpc (float): Distance to the galaxy in megaparsecs.
    - pixscale (float): Pixel scale in arcseconds per pixel.
    - param_file_path (str or Path): Path to the SOAX parameter text file.
    - noise_min (float): Floor on background RMS noise to prevent division by near-zero.
    - flatten_perc (float): Percentile for the arctan intensity rescaling applied before SOAX.
    - min_intensity (float): Pixels in the original image below this value are zeroed out.
    - sSFR (float): Specific star formation rate used in CO conversion.
    - Inclination (float): Galaxy inclination in degrees used in CO conversion.

    Returns
    -------
    - FilamentMapList (list): One FilamentMap per scale-decomposed FITS file found in CDD/.
    """
    FilamentMapList = []
    CDD_folder = os.path.join(label_folder_path, "CDD")

    for fits_file in os.listdir(CDD_folder):
        if fits_file.endswith(".fits"):
            ScalePix = pixscale * 4.848 * distance_Mpc
            filMap = FilamentMap.FilamentMap(ScalePix, base_dir, label_folder_path, fits_file, label, param_file_path, flatten_perc, min_intensity, sSFR, Inclination)
            filMap.setBlockData()
            filMap.setBkgSubDivRMS(noise_min)
            FilamentMapList.append(filMap)

    return FilamentMapList


def CreateSNRPlot(FilamentMapList, base_dir, percentile, write=False):
    """Plot the SNR at a given percentile versus physical scale for each galaxy.

    Parameters
    ----------
    - FilamentMapList (list): FilamentMap objects for a single galaxy across all scales.
    - base_dir (str or Path): Base directory; figure is saved to base_dir/Figures/.
    - percentile (float): Percentile (0–100) of the SNR map to extract per scale.
    - write (bool): Whether to save the figure to disk.
    """
    label_dict = {}
    for filMap in FilamentMapList:
        label = filMap.getLabel()
        if label not in label_dict:
            label_dict[label] = []
        SNRMap = filMap.getBkgSubDivRMSMap()
        scale  = float(filMap.getScale().replace('pc', ''))
        label_dict[label].append((scale, np.percentile(SNRMap, percentile)))

    for label, data in label_dict.items():
        scales, percentiles = zip(*data)
        plt.figure()
        plt.scatter(scales, percentiles, label=f"Celestial Object: {label}")
        plt.xlabel("Scale (pc)")
        plt.ylabel(f"SNR {percentile}th percentile")
        plt.title(f"SNR vs Scale — {label}")
        plt.legend()
        plt.grid(True)

    if write:
        plt.savefig(Path(f"{base_dir}/Figures/SNRPlot_{label}.png"))
    plt.close()


def clearAllFiles(base_directory, csv_path, param_file_path):
    """Delete all pipeline output files while preserving OriginalImages and the two metadata files.

    Parameters
    ----------
    - base_directory (str or Path): Root output directory to clear.
    - csv_path (str or Path): Path to ImageData.xlsx — excluded from deletion.
    - param_file_path (str or Path): Path to SoaxParams.txt — excluded from deletion.
    """
    for foldername, subfolders, filenames in os.walk(base_directory):
        if foldername == base_directory:
            continue
        if "originalimages" in foldername.lower():
            continue
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            if file_path != csv_path and file_path != param_file_path:
                os.remove(file_path)
                logger.debug("Deleted file: %s", file_path)

    logger.info("All files cleared from subdirectories of the directory structure.")


def createDirectoryStructure(base_directory, csv_path, ID_set=False):
    """Create the per-galaxy output folder tree under base_directory for each image in OriginalImages/.

    Existing directories are left untouched, so this is safe to re-run. Each galaxy receives
    subfolders for CDD, Composites, BlockedPng, SyntheticMap, SoaxOutput, BkgSubDivRMS, and
    Source_Removal, with per-scale subdirectories inside SoaxOutput.

    Parameters
    ----------
    - base_directory (str or Path): Root output directory.
    - csv_path (str or Path): Path to ImageData.xlsx, used to look up scale ranges.
    - ID_set (bool): If True, a unique image ID extracted from the filename is appended to
      the galaxy folder name to support multiple images of the same galaxy.
    """
    folder_path    = os.path.join(base_directory, "OriginalImages")
    os.makedirs(base_directory, exist_ok=True)
    os.makedirs(os.path.join(base_directory, "Figures"), exist_ok=True)

    for filename in os.listdir(folder_path):
        if not filename.endswith('.fits'):
            continue

        match = re.match(r"(.+?)_(F\d+[A-Z])[_.]", filename)
        if not match:
            logger.warning("Could not parse filename: %s", filename)
            continue

        label_name = match.group(1)
        band       = match.group(2)
        label      = f"{label_name}_{band}"

        if ID_set:
            stem         = os.path.splitext(filename)[0]
            image_id     = stem.rsplit('_', 1)[-1]
            folder_label = f"{label}_{image_id}"
        else:
            image_id     = None
            folder_label = label

        logger.info(
            "Processing file: %s (label=%s%s)",
            filename, label, f"  ID: {image_id}" if ID_set else "",
        )

        label_folder = os.path.join(base_directory, folder_label)
        os.makedirs(label_folder, exist_ok=True)

        info_result = getInfo(label, csv_path)
        if info_result is None:
            logger.warning("Skipping %s — not found in Excel file", label)
            continue

        _, _, _, _, _, min_power, max_power, _, _, _ = info_result

        subfolders = [
            "CDD", "Composites", "BlockedPng", "SyntheticMap",
            "SoaxOutput", "BkgSubDivRMS", "Source_Removal",
        ]
        soax_subfolders = [
            f"{2**i}pc" for i in range(int(min_power), int(max_power) + 1)
        ]

        for subfolder in subfolders:
            subfolder_path = os.path.join(label_folder, subfolder)
            os.makedirs(subfolder_path, exist_ok=True)
            if subfolder == "SoaxOutput":
                for s in soax_subfolders:
                    os.makedirs(os.path.join(subfolder_path, s), exist_ok=True)
            if subfolder == "Source_Removal":
                os.makedirs(os.path.join(subfolder_path, "CDD_Pix"),       exist_ok=True)
                os.makedirs(os.path.join(subfolder_path, "Source_Tables"), exist_ok=True)

        logger.info("Directory structure created for: %s", folder_label)


def renameFitsFiles(base_dir, csv_path, ID_set=False):
    """Standardise FITS filenames in OriginalImages/ using metadata from the Excel catalogue.

    The new name format is: label_Band_Telescope_ImageType[_ID].fits. Files not found in the
    catalogue are skipped with a warning. Re-running is safe; files already at the target name
    are renamed to themselves.

    Parameters
    ----------
    - base_dir (str or Path): Base directory containing the OriginalImages/ subfolder.
    - csv_path (str or Path): Path to ImageData.xlsx.
    - ID_set (bool): If True, the trailing token of the original filename is preserved as an
      image ID appended to the new name.
    """
    table                = pd.read_excel(csv_path)
    fits_file_folder_path = os.path.join(base_dir, "OriginalImages")

    for fits_file in os.listdir(fits_file_folder_path):
        full_file_path = os.path.join(fits_file_folder_path, fits_file)
        filename       = os.path.basename(fits_file)

        match = re.match(r"([^_]+)_([^_]+)", filename)
        if not match:
            logger.warning("Could not extract galaxy name from %s", filename)
            continue

        label = match.group(1)
        band  = match.group(2)

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

        if label_info.empty:
            logger.warning("Not found in Excel file: %s / %s", label, band)
            continue

        telescope = label_info.iloc[0]['Telescope']
        band      = label_info.iloc[0]['Band']
        img_type  = label_info.iloc[0]['Image_Type']

        suffix    = "_starsub" if "starsub" in filename.lower() else ""
        base_name = f"{label}_{band}_{telescope}_{img_type}{suffix}"
        new_filename = f"{base_name}_{image_id}.fits" if image_id else f"{base_name}.fits"

        new_filepath = os.path.join(fits_file_folder_path, new_filename)
        os.rename(full_file_path, new_filepath)
        logger.info("Renamed %s → %s", filename, new_filename)

    logger.info("Renaming process completed.")
