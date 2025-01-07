#https://archive.stsci.edu/hlsp/phangs.html#hst_image_products_table
#https://stsci.app.box.com/s/mhr1srey2h05ta26grd6hyp839pdlvco/folder/300932267185?page=2
import os
import FilamentMap
import numpy as np 
import mainFuncs
import Modified_Constrained_Diffusion
import threading 
import warnings
from astropy.io import fits
import logging
import time
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for thread safety
import matplotlib.pyplot as plt

# warnings.filterwarnings('ignore', category=UserWarning, module='astropy')
# logging.getLogger('astropy').setLevel(logging.ERROR)


if __name__ == "__main__":
    start = time.time()
    #Folder with JWST image
    galaxy_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2"
    decomposition_exists = True
    if(not decomposition_exists):
        Modified_Constrained_Diffusion.decompose(galaxy_dir)
    FilamentMapList = mainFuncs.setUp(galaxy_dir)
    mainFuncs.CreateSNRPlot(FilamentMapList, galaxy_dir, Write = True, verbose = False)

    # #iterate through JWST files
    for myFilMap in FilamentMapList:
    #     #Run Everything to get Composite
        myFilMap.ScaleBkgSub()
        myFilMap.RunSoaxThreads()
        myFilMap.CreateComposite("best_param1") #either this or run soax
        myFilMap.BlurComposite(set_blur_as_prob = True)
        myFilMap.SetIntensityMap(Orig = False)
        myFilMap.DisplayProbIntensityPlot(galaxy_dir, Orig = False, Write = True, verbose = False)
        myFilMap.ReHashComposite(ProbabilityThreshPercentile = .33, minPixBoxSize = 75) #formerly 75 increased to 100 for sim
    end = time.time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f'FilPHANGS took: {hours:02d}:{minutes:02d}:{seconds:02d} in total')



