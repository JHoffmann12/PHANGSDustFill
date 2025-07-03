#Decomposes image into specified scales

#imports
from pathlib import Path
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
import imageio
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from skimage.morphology import disk, binary_dilation
from skimage.restoration import inpaint
from skimage import color
from os import path
from skimage import data
from skimage.filters import meijering, sato, frangi, hessian
import matplotlib
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank
import copy 

plt.rcParams['figure.figsize'] = [15, 15]



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

def decompose(label_folder_path, base_dir, label, numscales=3):

    if(not decompositionExists(label_folder_path)): #check if scale decomposed image exists
        source_rem_dir = os.path.join(label_folder_path, "Source_Removal\CDD_Pix")

        orig_image_path = get_fits_file_path(os.path.join(base_dir, "OriginalImages"), label)

        with fits.open(orig_image_path, ignore_missing=True) as hdul:
            image = np.array(hdul[0].data)  # Assuming the image data is in the primary HDU
            header = hdul[0].header

        result, residual = constrained_diffusion_decomposition(image, e_rel=3e-2,max_n=numscales, sm_mode='reflect')  #for YMC stuff keep max_n=numscales small for execution speed and since full decomp irrelevent

        idx=0

        for i in result:
            save_path = os.path.join(source_rem_dir, '_CDDfs'+str(2**idx).rjust(4, '0')+'pix.fits')
            print(f'saving to {save_path}')
            hduout=fits.PrimaryHDU(data=i,header=header)
            hduout.writeto(save_path, overwrite=True)
            idx=idx+1

        dumpbkgds=True
        source_rem_dir = os.path.join(label_folder_path, "Source_Removal")

        if dumpbkgds==True:
            idx=0
            summed=np.zeros_like(i)
            for i in result:
                summed=summed+i
                save_path = os.path.join(source_rem_dir, '_CDDfs'+str(2**idx).rjust(4, '0')+'BKGD.fits')
                save_path_1 = os.path.join(source_rem_dir, '_CDDfs'+str(2**idx).rjust(4, '0')+'BKGDRATIO.fits')

                hduout=fits.PrimaryHDU(data=image-summed,header=header)
                hduout.writeto(save_path,overwrite=True)
                hduout=fits.PrimaryHDU(data=summed/(image-summed),header=header)
                hduout.writeto(save_path_1,overwrite=True)
                idx=idx+1


def constrained_diffusion_decomposition(data,
                                      e_rel=3e-2,
                                      max_n=None, sm_mode='reflect'):

    """
        perform constrained diffusion decomposition
        inputs:
            data: 
                n-dimensional array
            e_rel:
                relative error, a smaller e_rel means a better
                accuracy yet a larger computational cost
            max_n: 
                maximum number of channels. Channel number
                ranges from 0 to max_n
                if None, the program will calculate it automatically
            sm_mode: 
                {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional The mode
                parameter determines how the input array is extended beyond its
                boundaries in the convolution operation. Default is 'reflect'.
        output:
            results: of constained diffusion decomposition. Assuming that the input
            is a n-dimensional array, then the output would be a n+1 dimensional
            array. The added dimension is the scale. Component maps can be accessed
            via output[n], where n is the channel number.

                output[i] contains structures of sizes larger than 2**i pixels
                yet smaller than 2**(i+1) pixels.
            residual: structures too large to be contained in the results
                
    """
    
    ntot = int(log(min(data.shape))/log(2) - 1)  
    # the total number of scale map
    
    result = []
    # residual = []
    # residual.append(data)
    if  max_n is not None:
        ntot = np.min([ntot, max_n])
    print("ntot", ntot)

    diff_image = data.copy() * 0

    for i in range(ntot):
        print("i =", i)
        channel_image = data.copy() * 0  

        # computing the step size
        scale_end = float(pow(2, i + 1))
        scale_begining = float(pow(2, i))
        t_end = scale_end**2  / 2  # t at the end of this scale
        t_beginning = scale_begining**2  / 2 # t at the beginning of this scale

        if i == 0:
            delta_t_max = t_beginning * 0.1
        else:
            delta_t_max = t_beginning * e_rel



        niter = int((t_end - t_beginning) / delta_t_max + 0.5)
        delta_t = (t_end - t_beginning) / niter
        kernel_size = np.sqrt(2 * delta_t)    # size of gaussian kernel
        print(scale_begining,scale_end)
        print("kernel_size", kernel_size)
        for kk in range(niter):
            smooth_image = ndimage.gaussian_filter(data, kernel_size,
                                                   mode=sm_mode)
            sm_image_1 = np.minimum(data, smooth_image)
            sm_image_2 = np.maximum(data, smooth_image)

            diff_image_1 = data - sm_image_1
            diff_image_2 = data - sm_image_2

            diff_image = diff_image * 0

            positions_1 = np.where(np.logical_and(diff_image_1 > 0, data > 0))
            positions_2 = np.where(np.logical_and(diff_image_2 < 0, data < 0))

            diff_image[positions_1] = diff_image_1[positions_1]
            diff_image[positions_2] = diff_image_2[positions_2]

            channel_image = channel_image + diff_image

            data = data - diff_image
            # data = ndimage.gaussian_filter(data, kernel_size)     # !!!!
        result.append(channel_image)
        # residual.append(data)
    residual = data
    return result, residual



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
    cdd_path = os.path.join(base_path, "Source_Removal\CDD_Pix") #suspicious
    print(cdd_path)
    
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
    
