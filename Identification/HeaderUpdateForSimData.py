from astropy.io import fits

# Path to your FITS file
fits_file_path = r"c:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\GalaxySim\galaxysim_SigmaHI_CDDss0016pc_arcsinh0p1.fits"

# Open the FITS file
with fits.open(fits_file_path, mode='update') as hdul:
    # Access the primary header
    header = hdul[0].header

    # Add or update header keywords
    header['SIMPLE'] = (True, 'conforms to FITS standard')
    header['BITPIX'] = (-64, 'array data type')
    header['NAXIS'] = (2, 'number of array dimensions')
    header['NAXIS1'] = 2000
    header['NAXIS2'] = 2000
    header['WCSAXES'] = (2, 'number of World Coordinate System axes')
    header['CRPIX1'] = (942.62798074861, 'axis 1 coordinate of the reference pixel')
    header['CRPIX2'] = (1302.079567189, 'axis 2 coordinate of the reference pixel')
    header['CRVAL1'] = (24.171642007329, 'first axis value at the reference pixel')
    header['CRVAL2'] = (15.788103283, 'second axis value at the reference pixel')
    header['CTYPE1'] = ('RA---TAN', 'Axis 1 type')
    header['CTYPE2'] = ('DEC--TAN', 'Axis 2 type')
    header['CUNIT1'] = ('deg', 'Axis 1 units')
    header['CUNIT2'] = ('deg', 'Axis 2 units')
    header['CDELT1'] = (3.0812737269353E-05, 'Axis 1 coordinate increment at reference point')
    header['CDELT2'] = (3.0812737269353E-05, 'Axis 2 coordinate increment at reference point')
    header['PC1_1'] = (-0.99999998836263, 'linear transformation matrix element')
    header['PC1_2'] = (0.00015305834175289, 'linear transformation matrix element')
    header['PC2_1'] = (0.00015305835402243, 'linear transformation matrix element')
    header['PC2_2'] = (0.99999998819069, 'linear transformation matrix element')
    header['S_REGION'] = ('POLYGON ICRS  24.201820737 15.748064929 24.201832658 15.828178022 &', 'spatial extent of the observation')
    header['VELOSYS'] = (-29172.65, '[m/s] Barycentric correction to radial velocity')
    header['EXTVER'] = (1, 'extension value')
    header['RES'] = 'F770"'
    header['KERNPX'] = 0.3582408884456071
    header['SCLEPX'] = 1.602101957768592
    header['SCLEPXLO'] = 1.132857158490416
    header['SCLEPXHI'] = 2.265714316980831

    # Save changes (already in 'update' mode)
    hdul.flush()

print("Header updated successfully!")
