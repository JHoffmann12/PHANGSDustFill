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

plt.rcParams['figure.figsize'] = [15, 15]


def pick_scales_flex(dmpc, fwhm_asec, pix_asec, min_dim_img, force_std=False):

    stdladder_pc=2**np.array(range(8))
    stdladder_pc_lolim=2**(np.array(range(8))-0.5)
    stdladder_pc_hilim=2**(np.array(range(8))+0.5)

    altladder_pc=2**(np.array(range(8))-0.5)
    altladder_pc_lolim=2**(np.array(range(8))-1.0)
    altladder_pc_hilim=2**(np.array(range(8))+0.0)

    res_pc=4.848*fwhm_asec*dmpc
    pix_pc=4.848*pix_asec*dmpc

    print('Evaluating standard ladder rungs:')
    #too aggressive at small end .... idxstd=(stdladder_pc_lolim>=res_pc/2.35) & (stdladder_pc_hilim/pix_pc<=0.5*min_dim_img/2.35)
    idxstd=(stdladder_pc_lolim>=1.33*res_pc/2.35) & (stdladder_pc_hilim/pix_pc<=0.5*min_dim_img/2.35)
    print(res_pc, 'pc res. ; ',res_pc/2.35,'pc approx. dispersion')
    print(stdladder_pc[idxstd])
    print(stdladder_pc_lolim[idxstd])
    print(stdladder_pc_hilim[idxstd])
    print(res_pc/pix_pc, 'pix res. ; ',res_pc/pix_pc/2.35,'pix approx. dispersion')
    print(stdladder_pc[idxstd]/pix_pc)
    print(stdladder_pc_lolim[idxstd]/pix_pc)
    print(stdladder_pc_hilim[idxstd]/pix_pc)

    print('Evaluating alternate ladder rungs:')
    #too aggressive at small end .... idxalt=(altladder_pc_lolim>=res_pc/2.35) & (altladder_pc_hilim/pix_pc<=0.5*min_dim_img/2.35)
    idxalt=(altladder_pc_lolim>=1.33*res_pc/2.35) & (altladder_pc_hilim/pix_pc<=0.5*min_dim_img/2.35)
    print(res_pc, 'pc res. ; ',res_pc/2.35,'pc approx. dispersion')
    print(altladder_pc[idxalt])
    print(altladder_pc_lolim[idxalt])
    print(altladder_pc_hilim[idxalt])
    print(res_pc/pix_pc, 'pix res. ; ',res_pc/pix_pc/2.35,'pix approx. dispersion')
    print(altladder_pc[idxalt]/pix_pc)
    print(altladder_pc_lolim[idxalt]/pix_pc)
    print(altladder_pc_hilim[idxalt]/pix_pc)

    if (np.min(altladder_pc_hilim[idxalt]/pix_pc)<np.min(stdladder_pc_hilim[idxstd]/pix_pc)) & (force_std==False):
      print('Using alternate ladder!')
      ladder_pc=altladder_pc
      ladder_pc_lolim=altladder_pc_lolim
      ladder_pc_hilim=altladder_pc_hilim
      idx=idxalt.copy()
    else:
      print('Using standard ladder!')
      if (force_std==True): print('Standard ladder has been forced in code, not necessarily chosen via evaluation.')
      ladder_pc=stdladder_pc
      ladder_pc_lolim=stdladder_pc_lolim
      ladder_pc_hilim=stdladder_pc_hilim
      idx=idxstd.copy()

    return ladder_pc[idx]/pix_pc, ladder_pc_lolim[idx]/pix_pc, ladder_pc_hilim[idx]/pix_pc, ladder_pc[idx], ladder_pc_lolim[idx], ladder_pc_hilim[idx]




#consider PSF matching first
#or to a common spatial res amongst all galaxies
miriband='f770w'
if miriband=='f770w':
    res=0.27 #arcsec
    pixscale=0.11 #arcsec
else:
    print('Only F770W allowed presently.')
    exit()
#suffix='_miri_lv3_'+miriband+'_i2d_anchor.fits'
suffix='_F770W_starsub_anchored.fits'
filedir = r"c:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\OriginalMiriImages"

extnum=1 #PREVIOUS TO STARSUB INPUTS --- initial input file extension number, changed to 0 below after writing of CDD family products
# extnum=0 #NOW NEED IT AS 0 FOR STARSUB INPUTS
fracsmooth=0.333 # to remove divots, CDD outputs will be smoothed with Gaussian kernel having sigma=fracsmooth*pcscale[or matching pixscale]




# Load the distance table
dist_table_path = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\DistanceTable.txt"

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
        match = re.search(r'_jwst_miri_(ngc\d+)_', filename.lower())
        if match:
            galaxytab = match.group(1)  # Extracted galaxy name (e.g., 'ngc0628')
            print(f"Extracted galaxy name: {galaxytab}")
        else:
            print(f"Galaxy name could not be extracted from filename: {filename}")
            continue  # Skip this file

        if galaxytab not in dist_table['galaxy']:
            print(f"{galaxy} ({galaxytab}) is not found in the distance table. Skipping.")
            continue

        print(imagepath)
        path_parts = inputpath.split('/')
        filename = path_parts[-1]
        galaxy = filename.replace('_F770W_starsub_anchored.fits', '')
        dir = r'C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2'
        outfilename_prefix = os.path.join(dir, galaxytab + suffix)
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

        # Show the image
        plt.imshow(image_in, vmin=np.nanpercentile(image_in, 5.), vmax=np.nanpercentile(image_in, 95.), origin='lower')
        plt.show()

        # Pick scales dynamically for the galaxy
        pixscales, pixscales_lo, pixscales_hi, pcscales, pcscales_lo, pcscales_hi = pick_scales_flex(
            distance_mpc, res, pixscale, min_dim_img, force_std= True
        )
        print(pcscales)
        print(pixscales)

        # Perform constrained diffusion decomposition
        result_in, residual_in, kernel_sizes = cddss.constrained_diffusion_decomposition_specificscales(
            image_in, pixscales, pixscales_lo, pixscales_hi, e_rel=3.e-3
        )
        print(kernel_sizes)

        # Process and save outputs for each kernel size
        for idx, image_now in enumerate(result_in):
            header_in['KERNPX'] = kernel_sizes[idx]
            header_in['SCLEPX'] = pixscales[idx]
            header_in['SCLEPXLO'] = pixscales_lo[idx]
            header_in['SCLEPXHI'] = pixscales_hi[idx]

            psf_stddev = (pixscales[idx] * 2.35) * fracsmooth / 2.35  # Original stddev
            increased_psf_stddev = psf_stddev * 1.3  # Increase PSF by 30%

            hduout = fits.PrimaryHDU(
                data=convolve(image_now, Gaussian2DKernel(x_stddev=increased_psf_stddev)),
                header=header_in
            )
            hduout.writeto(outfilename_prefix + '_CDDss' + str(pcscales[idx]).rjust(4, '0') + 'pc.fits', overwrite=True)

        # Process arcsinh transformed images
        a = .1
        for idx in range(len(kernel_sizes)):
            with fits.open(outfilename_prefix + '_CDDss' + str(pcscales[idx]).rjust(4, '0') + 'pc.fits') as hdu:
                image_now = hdu[0].data
                header_now = hdu[0].header
            hduout = fits.PrimaryHDU(data=a * np.arcsinh(image_now / a), header=header_now)
            hduout.writeto(
                outfilename_prefix + '_CDDss' + str(pcscales[idx]).rjust(4, '0') + 'pc_arcsinh0p1.fits', overwrite=True
            )
            print('Image saved')
            plt.figure()
            plt.imshow(
                a * np.arcsinh(image_now / a),
                vmin=np.percentile(a * np.arcsinh(image_now / a), 10.),
                vmax=np.percentile(a * np.arcsinh(image_now / a), 99.),
                origin='lower'
            )
            plt.colorbar()
        print('Flattening with asinh transform after decomposition')
