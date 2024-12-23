import os
import numpy as np
from astropy.io import fits

# Specify the folder containing the FITS files
folder_path = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\Stacks"
output_file = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\Stacks\compositew16With8.fits"

# Initialize variables for stacking
composite_data = None

# Iterate through all FITS files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".fits"):
        file_path = os.path.join(folder_path, file_name)
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            if composite_data is None:
                composite_data = np.zeros_like(data, dtype=np.float64)
            composite_data += data  # Add pixel values

# Save the composite data as a new FITS file
if composite_data is not None:
    hdu = fits.PrimaryHDU(data=composite_data)
    hdu.writeto(output_file, overwrite=True)
    print(f"Composite FITS file saved as {output_file}")
else:
    print("No FITS files found in the specified folder.")
