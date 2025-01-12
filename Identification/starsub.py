from matplotlib import rcParams
rcParams['figure.figsize']=(8,8)
rcParams['font.family']='STIXGeneral'
rcParams['font.size']=15
rcParams['mathtext.fontset']='stix'
rcParams['legend.numpoints']=1 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from scipy.ndimage import rotate, zoom
import sys
sys.path.insert(0, '../jwst_scripts/LEGACY/')
from utils_jwst import *


sigma_f770  = 2*0.023254588397886626
sigma_f1130 = 2*0.03565945763914347
sigma_f2100 = 2*0.07287633636356125


f770_f200 = 0.12323 #ratio between F770/F200, using CIGALE stellar model + attenuation 
f770_f300 = 0.22 #ratio between F770_starlight/F300M, using CIGALE stellar model + attenuation 