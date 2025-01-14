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
import matplotlib.pyplot as plt
matplotlib.use('Agg')

if __name__ == "__main__":
    start = time.time()

    #Params to Set
    base_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\FilPHANGS"
    csv_path = r"C:\Users\HP\Documents\JHU_Academics\Research\FilPHANGS\ImageData.xlsx"
    param_file_path = r"C:\Users\HP\Documents\JHU_Academics\Research\FilPHANGS\SoaxParams.txt"
    probability_threshold = .01
    min_area_pix = 75

    mainFuncs.clear_all_files(base_dir, csv_path, param_file_path)
    mainFuncs.rename_fits_files(base_dir, csv_path)
    #Create Directory
    mainFuncs.create_directory_structure(base_dir, csv_path)
    #Clear Directory
    #****star sub TBD****

    #Loop through Each Galaxy
    for Galaxy in os.listdir(base_dir):
        galaxy_folder_path = os.path.join(base_dir, Galaxy)
        if not os.path.isdir(galaxy_folder_path):  # Skip if it's not a directory
            continue
        if(Galaxy != 'OriginalMiriImages' and Galaxy != "Figures"): #remove later
            print(f"Preparing analysis for {Galaxy}")
            distance_Mpc,res, pixscale, min_power, max_power = mainFuncs.getInfo(Galaxy, csv_path) #get relevant information for image
            if(not mainFuncs.decomposition_exists(galaxy_folder_path)): #decomposie original image into scales
                Modified_Constrained_Diffusion.decompose(base_dir, Galaxy, distance_Mpc, res, pixscale, min_power, max_power)
            FilamentMapList = mainFuncs.setUpGalaxy(base_dir, galaxy_folder_path, Galaxy, distance_Mpc, res, pixscale, param_file_path) #prepare all CDD images for one galaxy at a time
            # mainFuncs.CreateSNRPlot(FilamentMapList, base_dir, Write = False, verbose = False)
            for myFilMap in FilamentMapList: #iterate through each decomposed image
                myFilMap.ScaleBkgSub(WriteFits = True)
                myFilMap.getNoiseLevelsHistogram(min = 10**-2, verbose = False, WriteFig = True)
                myFilMap.RunSoaxThreads()
                myFilMap.CreateComposite(param_file_path) #either this or run soax update to take param path
                myFilMap.BlurComposite(set_blur_as_prob = True, WriteFits = True)
                myFilMap.CreateProbIntensityPlot(base_dir, Orig = False, WriteFig = True, verbose = False) #setIntensityMap in this function??***********
                myFilMap.ReHashComposite(ProbabilityThreshPercentile = probability_threshold, minPixBoxSize = min_area_pix, setAsComposite=True, WriteFits = True) 
                myFilMap.generateSyntheticFilaments(verbose = False, WriteFits = True)
                myFilMap.getFilamentLengthHistogram(probability_threshold = probability_threshold, verbose = False, WriteFig = True)

    #Time information
    end = time.time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'FilPHANGS took: {hours:02d}:{minutes:02d}:{seconds:02d} in total')



