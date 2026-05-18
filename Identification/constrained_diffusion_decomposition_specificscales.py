import logging
from math import log

import numpy as np
from astropy.io import fits
from scipy import ndimage

logger = logging.getLogger(__name__)


def constrained_diffusion_decomposition_specificscales(
    data, scales_pix, scales_pix_lo, scales_pix_hi,
    e_rel=3e-2, max_n=None, sm_mode='reflect',
):
    """Constrained diffusion decomposition at explicit pixel scales.

    Parameters
    ----------
    data : ndarray
        Input image (n-dimensional array).
    scales_pix : array-like
        Target pixel scales for each decomposition channel.
    scales_pix_lo, scales_pix_hi : array-like
        Lower and upper pixel-scale bounds for each channel.
    e_rel : float
        Relative error tolerance; smaller values increase accuracy and cost.
    max_n : int or None
        Maximum number of channels. None uses all scales provided.
    sm_mode : str
        Boundary mode passed to scipy gaussian_filter.

    Returns
    -------
    result : list of ndarray
        Per-channel structure maps. result[i] contains structures whose sizes
        fall between scales_pix_lo[i] and scales_pix_hi[i].
    residual : ndarray
        Structures too large to fit in any channel.
    kernel_sizes : list of float
        Gaussian kernel size used for each channel.
    """
    ntot = int(len(scales_pix))
    if max_n is not None:
        ntot = min(ntot, max_n)
    logger.debug("constrained diffusion: %d channels", ntot)

    result = []
    kernel_sizes = []
    diff_image = data.copy() * 0

    for i in range(ntot):
        channel_image = data.copy() * 0

        scale_end       = scales_pix_hi[i]
        scale_beginning = scales_pix_lo[i]
        t_end           = scale_end**2 / 2
        t_beginning     = scale_beginning**2 / 2

        delta_t_max = t_beginning * (0.1 if i == 0 else e_rel)
        niter       = int((t_end - t_beginning) / delta_t_max + 0.5)
        delta_t     = (t_end - t_beginning) / niter
        kernel_size = np.sqrt(2 * delta_t)
        logger.debug("channel %d: scale %.2f-%.2f px, kernel=%.3f, niter=%d",
                     i, scale_beginning, scale_end, kernel_size, niter)

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
        kernel_sizes.append(kernel_size)

    residual = data
    return result, residual, kernel_sizes


if __name__ == "__main__":
    # Standalone test — update the path below to point to a local FITS file.
    pix_pc = 5.24
    scales     = (2**np.array(range(3, 9))) / pix_pc
    scales_lo  = (2**(np.array(range(3, 9)) - 0.5)) / pix_pc
    scales_hi  = (2**(np.array(range(3, 9)) + 0.5)) / pix_pc

    fname = r"path/to/your/image.fits"
    with fits.open(fname) as hdulist:
        data = hdulist[0].data.astype(float)
        hdr  = hdulist[0].header

    data[np.isnan(data)] = 0
    result, residual, kernel_sizes = constrained_diffusion_decomposition_specificscales(
        data, scales, scales_lo, scales_hi
    )

    out = fits.PrimaryHDU(np.array(result), header=hdr)
    out.header['DMIN'] = np.nanmin(result)
    out.header['DMAX'] = np.nanmax(result)
    out.writeto(fname + '_scale.fits', overwrite=True)
