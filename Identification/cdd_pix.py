# Scale decomposition used by the source-removal branch of the pipeline.

import logging
import math
import os
from math import log
from pathlib import Path

import constrained_diffusion_decomposition_specificscales as cddss
import numpy as np
from astropy.io import fits
from scipy import ndimage

logger = logging.getLogger(__name__)


def get_fits_file_path(folder_path, galaxy_name):
    """Return the path of the first FITS file whose name contains galaxy_name.

    Parameters
    ----------
    folder_path : str
        Directory to search.
    galaxy_name : str
        Substring to match against filenames.

    Returns
    -------
    str or None
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.fits') and galaxy_name in file_name:
            return str(Path(folder_path) / file_name)
    logger.warning("No FITS file found for galaxy: %s", galaxy_name)
    return None


def decompose(label_folder_path, base_dir, label, numscales=3):
    """Decompose an image into pixel scales and write outputs to Source_Removal/CDD_Pix/.

    Parameters
    ----------
    label_folder_path : str or Path
        Per-galaxy output directory.
    base_dir : str or Path
        Root FilPHANGS data directory.
    label : str
        Galaxy label string.
    numscales : int
        Number of CDD scales to compute.
    """
    if decompositionExists(label_folder_path):
        return

    source_rem_dir  = Path(label_folder_path) / "Source_Removal" / "CDD_Pix"
    orig_image_path = get_fits_file_path(Path(base_dir) / "OriginalImages", label)

    with fits.open(orig_image_path) as hdu:
        try:
            image  = hdu[0].data
            header = hdu[0].header
            if image is None:
                image  = hdu[1].data
                header = hdu[1].header
        except IndexError:
            image  = hdu[1].data
            header = hdu[1].header
        try:
            min_dim_img = np.min([header['NAXIS1'], header['NAXIS2']])
        except KeyError:
            try:
                header['NAXIS1'] = image.shape[1]
                header['NAXIS2'] = image.shape[0]
            except AttributeError:
                image  = hdu[1].data
                header = hdu[1].header
                try:
                    min_dim_img = np.min([header['NAXIS1'], header['NAXIS2']])
                except KeyError:
                    header['NAXIS1'] = image.shape[1]
                    header['NAXIS2'] = image.shape[0]

    result, residual = constrained_diffusion_decomposition(
        image, e_rel=3e-2, max_n=numscales, sm_mode='reflect'
    )

    for idx, channel in enumerate(result):
        save_path = source_rem_dir / f"_CDDfs{str(2**idx).rjust(4, '0')}pix.fits"
        logger.debug("writing CDD channel %d to %s", idx, save_path)
        fits.PrimaryHDU(data=channel, header=header).writeto(save_path, overwrite=True)

    source_rem_bkgd = Path(label_folder_path) / "Source_Removal"
    summed = np.zeros_like(result[0])
    for idx, channel in enumerate(result):
        summed += channel
        bkgd_path  = source_rem_bkgd / f"_CDDfs{str(2**idx).rjust(4, '0')}BKGD.fits"
        ratio_path = source_rem_bkgd / f"_CDDfs{str(2**idx).rjust(4, '0')}BKGDRATIO.fits"
        fits.PrimaryHDU(data=image - summed, header=header).writeto(bkgd_path, overwrite=True)
        fits.PrimaryHDU(data=summed / (image - summed), header=header).writeto(ratio_path, overwrite=True)


def constrained_diffusion_decomposition(data, e_rel=3e-2, max_n=None, sm_mode='reflect'):
    """Constrained diffusion decomposition using power-of-2 scales.

    Parameters
    ----------
    data : ndarray
        Input image.
    e_rel : float
        Relative error tolerance.
    max_n : int or None
        Maximum number of channels.
    sm_mode : str
        Boundary mode for gaussian_filter.

    Returns
    -------
    result : list of ndarray
    residual : ndarray
    """
    ntot = int(log(min(data.shape)) / log(2) - 1)
    if max_n is not None:
        ntot = min(ntot, max_n)
    logger.debug("constrained diffusion: %d channels", ntot)

    result      = []
    diff_image  = data.copy() * 0

    for i in range(ntot):
        channel_image   = data.copy() * 0
        scale_end       = float(pow(2, i + 1))
        scale_beginning = float(pow(2, i))
        t_end           = scale_end**2 / 2
        t_beginning     = scale_beginning**2 / 2

        delta_t_max = t_beginning * (0.1 if i == 0 else e_rel)
        niter       = int((t_end - t_beginning) / delta_t_max + 0.5)
        delta_t     = (t_end - t_beginning) / niter
        kernel_size = np.sqrt(2 * delta_t)

        for _ in range(niter):
            smooth_image = ndimage.gaussian_filter(data, kernel_size, mode=sm_mode)
            sm_image_1   = np.minimum(data, smooth_image)
            sm_image_2   = np.maximum(data, smooth_image)

            diff_image_1 = data - sm_image_1
            diff_image_2 = data - sm_image_2
            diff_image   = diff_image * 0

            pos1 = np.where(np.logical_and(diff_image_1 > 0, data > 0))
            pos2 = np.where(np.logical_and(diff_image_2 < 0, data < 0))
            diff_image[pos1] = diff_image_1[pos1]
            diff_image[pos2] = diff_image_2[pos2]

            channel_image = channel_image + diff_image
            data          = data - diff_image

        result.append(channel_image)

    residual = data
    return result, residual


def roundToNearestPowerOf2(n):
    """Return the power of 2 nearest to n."""
    if n <= 0:
        raise ValueError("Input must be a positive number.")
    lower = 2 ** math.floor(math.log2(n))
    upper = 2 ** math.ceil(math.log2(n))
    return lower if (n - lower) < (upper - n) else upper


def decompositionExists(base_path):
    """Return True if the Source_Removal/CDD_Pix folder exists and is non-empty."""
    cdd_path = Path(base_path) / "Source_Removal" / "CDD_Pix"
    if not os.path.exists(cdd_path):
        logger.debug("CDD_Pix folder not found in %s", base_path)
        return False
    if not os.listdir(cdd_path):
        logger.debug("CDD_Pix folder is empty in %s", base_path)
        return False
    return True
