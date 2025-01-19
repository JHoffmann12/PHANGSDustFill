#Decomposes image into specified scales

#imports
import constrained_diffusion_decomposition_specificscales as cddss
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import make_lupton_rgb
from math import log
import math
from matplotlib import cm
from pylab import *
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import color
from skimage import data
from skimage.filters import frangi, hessian, meijering, sato


plt.rcParams['figure.figsize'] = [15, 15]


def gaussian_2d(xy, x0, y0, sigma, amplitude, offset):

    """
    Define a 2D Gaussian function
    
    Parameters:
    - xy: A tuple of (x, y) coordinates (arrays or values).
    - x0: The x-coordinate of the center of the Gaussian.
    - y0: The y-coordinate of the center of the Gaussian.
    - sigma: The standard deviation (spread) of the Gaussian.
    - amplitude: The peak value of the Gaussian.
    - offset: A constant value to add to the Gaussian function (baseline).
    """

    x, y = xy
    exp_term = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    return amplitude * exp_term + offset


def get_fits_file_path(folder_path, galaxy_name):

    """
    Function to get the path of a FITS file containing a specific galaxy name

    Parameters:
    - folder_path: The path to the folder where FITS files are stored.
    - galaxy_name: The name (or part of the name) of the galaxy to search for in FITS file names.

    Returns:
    - The full path to the matching FITS file, or None if no file is found.
    """

    for file_name in os.listdir(folder_path):  #Loop through all files in the given folder

        if file_name.endswith('.fits') and galaxy_name in file_name:   # Check if the file is a FITS file and contains the galaxy name

            return os.path.join(folder_path, file_name)
    
    print(f"No FITS file found for galaxy: {galaxy_name}") #If no matching file is found
    
    return None

def decompose(label_folder_path, base_dir, label, distance_mpc, res, pixscale, min_power, max_power):

    """
    Decompose th eimage into specified scales and save the decompositions into CDD subfolder

    Parameters:
    - label_folder_path (str): path to the folder associated with the specified label/celestial object
    - base_dir (str): Base directory for all FilPHANGS files
    - label (str): label of the desired celestial object
    - distance_Mpc (float): Distance in Mega Parsecs to the celestial object
    - res (float): angula resolution associated with the image
    - pixscale (str): The pixel level resolution associated with the image
    - min_power(float): minimum power of 2 for scale decomposition
    - max_power(float): maximum power of 2 for scale decomposition
    """
    
    if(not decompositionExists(label_folder_path)): #check if scale decomposed image exists


        fracsmooth=0.333 # to remove divots, CDD outputs will be smoothed with Gaussian kernel having sigma=fracsmooth*pcscale[or matching pixscale]


        imagepath = get_fits_file_path(os.path.join(base_dir, "OriginalImages"), label)

        # Open FITS file and handle various erros
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

        pix_pc=4.848*pixscale*distance_mpc # convert to parcecs per pixel
    
        #generate scales to decompose image into
        pixscales=(2.0**np.array(range(min_power, max_power + 1)))/pix_pc
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

        #decompose the image
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

            hduout = fits.PrimaryHDU(data = image_now, header=header)
            tag = 'pc.fits'

            base_name = os.path.splitext(os.path.basename(imagepath))[0]

            if pcscales[idx] > 1:  # Check if the value is an integer
                pcscales[idx] = roundToNearestPowerOf2(pcscales[idx])
                outputpath = fr"{base_dir}\{label}\CDD\{base_name}_CDDss{str(int(pcscales[idx])).rjust(4, '0')}{tag}"
            else: 
                outputpath = fr"{base_dir}\{label}\CDD\{base_name}_CDDss{str(float(pcscales[idx])).rjust(4, '0')}{tag}"

            hduout.writeto(outputpath, overwrite=True)
            print('Image saved')

def roundToNearestPowerOf2(n):

    """
    Rounds to nearest power of 2. Corrects small error that may occur in file names. 

    Parameters:
    - n (float): Number to round

    Returns:
    - bool: Nearest power of 2
    """
        
    if n <= 0:
        raise ValueError("Input must be a positive number.")
    
    lower_power = 2 ** math.floor(math.log2(n))
    upper_power = 2 ** math.ceil(math.log2(n))

    return lower_power if (n - lower_power) < (upper_power - n) else upper_power


def decompositionExists(base_path):

    """
    Checks if the "CDD" folder in the specified path is empty or not.

    Parameters:
    - root_path (str): The root directory to check.

    Returns:
    - bool: True if the folder is not empty, False if it is empty or doesn't exist.
    """

    # Construct the CDD folder path
    cdd_path = os.path.join(base_path, "CDD")

    # Check if the folder exists
    if not os.path.exists(cdd_path):
        print(f"The folder 'CDD' does not exist in {base_path}.")
        return False

    # Check if the folder is empty
    if not os.listdir(cdd_path):
        print(f"The folder 'CDD' is empty in {base_path}.")
        return False
    else:
        print(f"The folder 'CDD' is not empty in {base_path}.")
        return True