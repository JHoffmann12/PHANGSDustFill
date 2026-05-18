#FilPHANGS Main script

#imports
import logging
from pathlib import Path
import FilamentMap
import Modified_Constrained_Diffusion
from Modified_Constrained_Diffusion import get_fits_file_path
import mainFuncs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import MySourceFinder
from astropy.io import fits
import cdd_pix
import CloudClean
import warnings
from astropy.wcs import FITSFixedWarning
matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=FITSFixedWarning)
logging.getLogger('reproject').setLevel(logging.WARNING)  # suppress non-dask mode INFO spam

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('filphangs.log', mode='a'),
    ]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Paths — update these for your environment
    # -------------------------------------------------------------------------
    base_dir        = Path(r"C:\Users\jhoffm72\Documents\FilPHANGS\Data")
    csv_path        = Path(r"C:\Users\jhoffm72\Documents\FilPHANGS\Data\ImageData.xlsx")
    param_file_path = Path(r"C:\Users\jhoffm72\Documents\FilPHANGS\Data\SoaxParams.txt")
    batch_path      = Path(r"C:\Users\jhoffm72\Downloads\batch_soax_v3.7.0.exe")

    # Source removal requires Julia and CloudClean (see README)
    julia_path     = Path(r"C:\Users\jhoffm72\Documents\FilPHANGS\PHANGSDustFill\Identification\JuliaCloudClean_Output1.ipynb")
    julia_out_path = julia_path

    # Optional: set to None to disable region assignment or dynamic alphaCO
    region_dir_path      = Path(r"C:\Users\jhoffm72\Documents\FilPHANGS\Data\masks_v5_simple")
    dynamic_alphaCO_path = Path(r"C:\Users\jhoffm72\Documents\FilPHANGS\Data\PHANGS_alphaCO_conversion_factor_maps")

    # -------------------------------------------------------------------------
    # Detection parameters
    # -------------------------------------------------------------------------

    # Minimum filament length-to-width ratio. Width = 16 pc / ScalePix.
    # At 16pc (BlockFactor=0, Scalepix~5.25): min_skel ~25 px -> ~131 pc minimum length.
    # The effective ratio increases at larger (blocked) scales due to the sqrt-BF reduction.
    min_aspect_ratio    = 8.2
    min_snake_length_ss = mainFuncs.getMinSnakeLengthFromAspectRatio(min_aspect_ratio)

    # SOAX minimum foreground intensity (0–65535). Pixels below this are ignored
    # during snake initialization.
    min_fg_int = 1638

    # Floor on background RMS noise. Prevents division by near-zero in faint regions.
    # Use ~0.55 for extinction maps (IC5146), ~0.01 for JWST F770W.
    noise_min = 1e-2

    # Percentile used to set the knee of the arctan intensity rescaling applied
    # before SOAX. Use ~99 for high-dynamic-range images (IC5146), ~90 for F770W.
    flatten_perc = 90

    # Pixels in the original image below this intensity are zeroed out before
    # processing. Use 0 for JWST, ~4 for Herschel.
    min_intensity = 0

    # -------------------------------------------------------------------------

    start = time.time()

    # Uncomment to restrict processing to specific galaxies:
    # todo = ['0628', '1566', '4535', '7496']

    # Uncomment to wipe all outputs and start fresh:
    # mainFuncs.clearAllFiles(base_dir, csv_path, param_file_path)

    # Standardize filenames and create the per-image output directory structure.
    # Safe to re-run; existing directories are left untouched.
    mainFuncs.renameFitsFiles(base_dir, csv_path, ID_set=True)
    mainFuncs.createDirectoryStructure(base_dir, csv_path)

    for label in os.listdir(base_dir):

        label_folder_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_folder_path):
            continue

        # Skip non-galaxy folders
        if label in ('OriginalMiriImages', 'Figures') or 'IC5146' in label or 'masks_v5' in label: # or not '0628_F770W' in label: 
            continue

        # Uncomment to process only a subset:
        # if not any(t in label for t in todo):
        #     continue

        info = mainFuncs.getInfo(label, csv_path)
        if info is None:
            continue
        distance_Mpc, res, pixscale, MJysr, Band, min_power, max_power, Rem_sources, sSFR, Inclination = info

        orig_image = get_fits_file_path(os.path.join(base_dir, "OriginalImages"), label)

        if Rem_sources:
            if orig_image is None:
                logger.warning('Skipping %s: no FITS file found in OriginalImages', label)
                continue
            cdd_pix.decompose(label_folder_path, base_dir, label, numscales=3)
            mask_save_path = MySourceFinder.CreateSourceMask(label_folder_path, orig_image, res, pixscale, MJysr, Band, pixscale * 4.848 * distance_Mpc)
            image_path = CloudClean.Remove(julia_path, julia_out_path, mask_save_path, orig_image, label_folder_path)
            image_path = MySourceFinder.CloudCleanCheck(image_path, mask_save_path, orig_image, label_folder_path)
        else:
            image_path = get_fits_file_path(os.path.join(base_dir, "OriginalImages"), label)

        # Decompose into physical scales via constrained diffusion
        Modified_Constrained_Diffusion.decompose(image_path, label_folder_path, base_dir, label, distance_Mpc, res, pixscale, min_power, max_power, Rem_sources)

        # Build a FilamentMap object for each decomposed scale
        FilamentMapList = mainFuncs.setUpGalaxy(base_dir, label_folder_path, label, distance_Mpc, pixscale, param_file_path, noise_min, flatten_perc, min_intensity, sSFR, Inclination)

        for filMap in FilamentMapList:

            filMap.scaleBkgSubDivRMSMap(write_fits=False)
            filMap.runSoaxThreads(min_snake_length_ss, min_fg_int, batch_path)
            filMap.createComposite(write_fits=False)
            rep_centers = filMap.processComposite(min_confidence=0.1, min_overlap_fraction=0.1)

            # PSF-based synthetic map + property extraction (primary pipeline)
            filMap.getSyntheticFilamentMapExact(min_scale=2**min_power, rep_centers=rep_centers, alphaCO_tag='SL24', use_dynamic_alphaCO=dynamic_alphaCO_path, use_Regions=region_dir_path, extract_Properties=True, write_fits=True, min_aspect_ratio=min_aspect_ratio)

            # LSE approximate synthetic map (faster, less accurate — uncomment to use instead)
            # filMap.getSyntheticFilamentMapApprox(min_scale=2**min_power, rep_centers=rep_centers, alphaCO_tag='SL24', use_dynamic_alphaCO=dynamic_alphaCO_path, use_Regions=region_dir_path, extract_Properties=False, write_fits=True, min_aspect_ratio=min_aspect_ratio)

            # Legacy SOAX-composite processing (uncomment to use instead of PSF pipeline)

            # Diagnostic plots (uncomment as needed)
            # mainFuncs.CreateSNRPlot(FilamentMapList, base_dir, percentile=99, write=True)
            # filMap.getProbIntensityPlot(use_orig_img=False, write_fig=True)
            # filMap.getNoiseLevelsHistogram(noise_min=noise_min, write_fig=True)

    # Display Time information
    end = time.time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    logger.info('FilPHANGS took: %02d:%02d:%02d total', hours, minutes, seconds)
