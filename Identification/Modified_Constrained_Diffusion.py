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



def decompose(dir):
    miriband='f770w'
    if miriband=='f770w':
        res=0.27 #arcsec
        pixscale=0.11 #arcsec
    else:
        print('Only F770W allowed presently.')
        exit()
    #suffix='_miri_lv3_'+miriband+'_i2d_anchor.fits'
    suffix='_F770W_starsub_anchored.fits'
    filedir = os.path.join(dir, "OriginalMiriImages")
    extnum=0 #NOW NEED IT AS 0 FOR STARSUB INPUTS
    fracsmooth=0.333 # to remove divots, CDD outputs will be smoothed with Gaussian kernel having sigma=fracsmooth*pcscale[or matching pixscale]

    # Load the distance table
    dist_table_path = os.path.join(dir, "DistanceTable.txt")
    try:
        dist_table = Table.read(dist_table_path, format='ascii')
        print("Distance table loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Distance table not found at {dist_table_path}. Please verify the path.")
        exit()

    for inputpath in os.listdir(filedir):
        if inputpath.endswith('.fits'):
            imagepath = os.path.join(filedir, inputpath)

            # Dynamic galaxy name matching
            filename = inputpath.split('/')[-1]
            match = re.search(r'(ngc\d+)_', filename.lower())
            if match:
                galaxytab = match.group(1)  # Extracted galaxy name (e.g., 'ngc0628')
                print(f"Extracted galaxy name: {galaxytab}")
            else:
                print(f"Galaxy name could not be extracted from filename: {filename}")
                continue  # Skip this file

            if galaxytab not in dist_table['galaxy']:
                print(f"{galaxy} ({galaxytab}) is not found in the distance table. Skipping.")
                continue

            path_parts = inputpath.split('/')
            filename = path_parts[-1]
            galaxy = filename.replace('_F770W_starsub_anchored.fits', '')
            outfilename_prefix = os.path.join(dir, galaxytab)
            print(f"out path: {outfilename_prefix}")
            outfilename_prefix = outfilename_prefix.replace('.fits', '')
            print(inputpath)
            print(filename)
            print(galaxy)

            # Retrieve distance information
            distance_mpc = dist_table['current_dist'][dist_table['galaxy'] == galaxytab][0]
            distance_err_mpc = dist_table['current_dist_err'][dist_table['galaxy'] == galaxytab][0]
            dm = 5.0 * np.log10(distance_mpc * 1.e6) - 5.0

            print(f'{galaxytab} is {distance_mpc} Mpc away!')
            # Open FITS file
            with fits.open(imagepath) as hdu:
                print(f'path: {imagepath}')
                image_in = hdu[extnum].data
                header_in = hdu[extnum].header
                try:
                    min_dim_img = np.min([header_in['NAXIS1'], header_in['NAXIS2']])
                except KeyError:
                    header_in['NAXIS1'] = image_in.shape[1]
                    header_in['NAXIS2'] = image_in.shape[0]
                    min_dim_img = np.min([header_in['NAXIS1'], header_in['NAXIS2']])

                hdu.info()


            print(image_in.shape)

            pix_pc=4.848*pixscale*distance_mpc
            print(f'parcecs per pixel: {pix_pc}')
            pixscales=(2**np.array(range(3,9)))/pix_pc
            pcscales = pixscales * pix_pc
            pixscales_lo =(2**(np.array(range(3,9))-0.5))/pix_pc
            pixscales_hi =(2**(np.array(range(3,9))+0.5))/pix_pc
            res_pc=4.848*res*distance_mpc
            idx=(pixscales_lo*pix_pc>=1.33*res_pc/2.35) & (pixscales_hi*pix_pc/res_pc<=0.5*min_dim_img/2.35)
            pixscales_hi = pixscales_hi[idx]
            pixscales_lo = pixscales_lo[idx]
            pixscales = pixscales[idx]
            pcscales = pcscales[idx]
            print(f"pc ranges: {pixscales*pix_pc}")

            # Pass the modified image to your function
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
                a = .1
                hduout = fits.PrimaryHDU(data=a * np.arcsinh(image_now / a), header=header)
                out_name = os.path.join(outfilename_prefix, galaxytab + suffix + '_CDDss' + str(int(pcscales[idx])).rjust(4, '0') + 'pc_arcsinh0p1.fits')
                hduout.writeto(out_name, overwrite=True)
                print('Image saved')

