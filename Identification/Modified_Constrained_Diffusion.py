import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from astropy.io import fits
from pylab import *
import os
import constrained_diffusion_decomposition_specificscales as cddss
from astropy.convolution import convolve,Gaussian2DKernel
from astropy.visualization import make_lupton_rgb
from astropy.table import Table
from skimage import data
from skimage import color
from skimage.filters import meijering, sato, frangi, hessian
import re
from scipy.ndimage import gaussian_filter
from astropy.io import fits
import sys
import numpy as np
from math import log
from scipy import ndimage

plt.rcParams['figure.figsize'] = [15, 15]


# Define a 2D Gaussian function
def gaussian_2d(xy, x0, y0, sigma, amplitude, offset):
    x, y = xy
    exp_term = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return amplitude * exp_term + offset

def get_fits_file_path(folder_path, galaxy_name):
    # Loop through all files in the given folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a FITS file and contains the galaxy name
        if file_name.endswith('.fits') and galaxy_name in file_name:
            return os.path.join(folder_path, file_name)
    
    # If no matching file is found
    print(f"No FITS file found for galaxy: {galaxy_name}")
    return None

def decompose(base_dir, Galaxy, distance_mpc, res, pixscale, min_power, max_power, perform_arcsin = False):

    fracsmooth=0.333 # to remove divots, CDD outputs will be smoothed with Gaussian kernel having sigma=fracsmooth*pcscale[or matching pixscale]


    imagepath = get_fits_file_path(os.path.join(base_dir, "OriginalImages"), Galaxy)

    # Open FITS file
    with fits.open(imagepath) as hdu:
        try:
            image_in = hdu[0].data
            header_in = hdu[0].header
        except IndexError:
            image_in = hdu[1].data
            header_in = hdu[1].header
        try:
            min_dim_img = np.min([header_in['NAXIS1'], header_in['NAXIS2']])
        except KeyError:
            try: 
                header_in['NAXIS1'] = image_in.shape[1]
                header_in['NAXIS2'] = image_in.shape[0]
            except AttributeError: 
                image_in = hdu[1].data
                header_in = hdu[1].header
                try: 
                    min_dim_img = np.min([header_in['NAXIS1'], header_in['NAXIS2']])
                except KeyError: 
                    header_in['NAXIS1'] = image_in.shape[1]
                    header_in['NAXIS2'] = image_in.shape[0]

        hdu.info()

    pix_pc=4.848*pixscale*distance_mpc
    pixscales=(2.0**np.array(range(min_power,max_power + 1)))/pix_pc
    pcscales = pixscales * pix_pc
    pixscales_lo =(2.0**(np.array(range(min_power,max_power + 1))-0.5))/pix_pc
    pixscales_hi =(2.0**(np.array(range(min_power, max_power + 1))+0.5))/pix_pc
    res_pc=4.848*res*distance_mpc
    idx=(pixscales_lo*pix_pc>=1.33*res_pc/2.35)  & (pixscales_hi*pix_pc/res_pc<=0.5*min_dim_img/2.35)  #*****FIX LATER!**********
    pixscales_hi = pixscales_hi[idx]
    pixscales_lo = pixscales_lo[idx]
    pixscales = pixscales[idx]
    pcscales = pcscales[idx]
    print(f"pc ranges: {pixscales*pix_pc}")

    result_in, residual_in, kernel_sizes = cddss.constrained_diffusion_decomposition_specificscales(
        image_in, pixscales, pixscales_lo, pixscales_hi, e_rel=3.e-2
    )

    # Process and save outputs for each kernel size
    for idx, image_now in enumerate(result_in):
        header = header_in.copy()
        header['KERNPX'] = kernel_sizes[idx]
        header['SCLEPX'] = pixscales[idx]
        header['SCLEPXLO'] = pixscales_lo[idx]
        header['SCLEPXHI'] = pixscales_hi[idx]

        psf_stddev = (pixscales[idx] * 2.35) * fracsmooth / 2.35  # Original stddev

        image_now =convolve(image_now, Gaussian2DKernel(x_stddev=psf_stddev))        
        # Process arcsinh transformed images
        if(perform_arcsin):
            a = .1
            hduout = fits.PrimaryHDU(data=a * np.arcsinh(image_now / a), header=header)
            tag = 'pc_arcsinh0p1.fits'
        else:
            hduout = fits.PrimaryHDU(data = image_now, header=header)
            tag = 'pc.fits'

        base_name = os.path.splitext(os.path.basename(imagepath))[0]
        if pcscales[idx].is_integer():  # Check if the value is an integer
            outputpath = fr"{base_dir}\{Galaxy}\CDD\{base_name}_CDDss{str(int(pcscales[idx])).rjust(4, '0')}{tag}"
        else: 
            outputpath = fr"{base_dir}\{Galaxy}\CDD\{base_name}_CDDss{str(float(pcscales[idx])).rjust(4, '0')}{tag}"
        hduout.writeto(outputpath, overwrite=True)
        print('Image saved')




