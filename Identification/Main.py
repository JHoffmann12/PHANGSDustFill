#FilPHANGS Main script

#imports 
from pathlib import Path
import FilamentMap
import Modified_Constrained_Diffusion
import mainFuncs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from astropy.io import fits
matplotlib.use('Agg')

if __name__ == "__main__":

    #Params to Set
#   _____________________________________________________________________________________________
#   ____________________________________________________________________________________________

    #paths
    base_dir = Path("/Users/jakehoffmann/Documents/JHU_Research/FilPHANGS_Main/FilPHANGS")
    csv_path = Path("/Users/jakehoffmann/Documents/JHU_Research/FilPHANGS_Main/FilPHANGS/ImageData.xlsx")
    param_file_path = Path("/Users/jakehoffmann/Documents/JHU_Research/FilPHANGS_Main/FilPHANGS/SoaxParams.txt")
    batch_path = Path("/Users/jakehoffmann/Downloads/batch_soax_v3.7.0")
    #SOAX params
    min_snake_length_ss = 25
    min_fg_int = 1638
    #Non SOAX params
    probability_threshold = .3
    min_area_pix = 10
    noise_min = 10**-2 #.55 for IC5146
    flatten_perc = 90 #99 for IC5146
    min_intensity = 0
#   ____________________________________________________________________________________________
#   ____________________________________________________________________________________________

    start = time.time() #get start time

    # mainFuncs.clearAllFiles(base_dir, csv_path, param_file_path) #clear all files
    mainFuncs.renameFitsFiles(base_dir, csv_path) #apply naming convention to original files
    mainFuncs.createDirectoryStructure(base_dir, csv_path)  #Create Directory and sub folders

    for label in os.listdir(base_dir):  #Loop through Each Galaxy

        label_folder_path = os.path.join(base_dir, label)

        if not os.path.isdir(label_folder_path):  # Skip if it's not a directory
            continue

        if(label != 'OriginalMiriImages' and label != "Figures" and 'ngc0628_F770W' in label): 

            distance_Mpc,res, pixscale, min_power, max_power = mainFuncs.getInfo(label, csv_path) #get relevant information for image from csv file
            Modified_Constrained_Diffusion.decompose(label_folder_path, base_dir, label, distance_Mpc, res, pixscale, min_power, max_power) #decompose into scales
            FilamentMapList = mainFuncs.setUpGalaxy(base_dir, label_folder_path, label, distance_Mpc, res, pixscale, param_file_path, noise_min, flatten_perc, min_intensity) #Initialize Filament Map objects
            mainFuncs.CreateSNRPlot(FilamentMapList, base_dir, percentile = 99, write = True)

            for filMap in FilamentMapList: #iterate through each Filament Map object 

                #Necessary functions in order to produce a skeletonized filament map
                filMap.scaleBkgSubDivRMSMap(write_fits = True)
                filMap.runSoaxThreads(min_snake_length_ss, min_fg_int, batch_path) #Create 10 soax FITS files
                filMap.createComposite(write_fits = True) #Combine all 10 Fits files
                # filMap.blurComposite(set_blur_as_prob = True, write_fits = True) #Blur the composite
                # skelData = filMap.applyProbabilityThresholdAndSkeletonize(probability_threshold = probability_threshold, min_area_pix = min_area_pix, write_fits = True)
                # filMap.removeJunctions(skelData, probability_threshold, min_area_pix, set_as_composite = True, write_fits = True)

                #Extra plots
                # filMap.getProbIntensityPlot(use_orig_img = False, write_fig = True) #Compares probability vs intensity
                # filMap.getSyntheticFilamentMap(probability_threshold = 0, write_fits = True) # Creates a synthetic map of all filaments at a single scale from the blurred probability_map. set_as_composite = True. 
                # filMap.getNoiseLevelsHistogram(noise_min = noise_min, write_fig = True)  # Histogram of calculated noise
                # filMap.getFilamentLengthHistogram(probability_threshold = probability_threshold, write_fig = True) # Histogram of filament length in pixels

    # Display Time information
    end = time.time()
    elapsed_time = end - start
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f'FilPHANGS took: {hours:02d}:{minutes:02d}:{seconds:02d} in total')


#        1) Correct synthetic maps sharp****
#           --> Skeeltonize probability thresholded map, blow up to ~3 pix radius filament and take pixel value from 
#           the CDD image, then convolve to filament scale... Check the normalization of convolution 
#           update: Scaling issues after blurr and skeltonization issues due to width of filaments after thresholding
#        2) Min area to min length, junction removal
#        3) Find the maximum value in blurred image to base probability threshold off of. 
#        4) Work on mass/length****
# Draw a red box on the image the size of the bkg sub map box

#complete but can revisit: 
#       1) Attenuation: 
#           steps: 1) Decompose. 2) Hist eq based on scale and radius based on 2*decomposed scale. 
#           Include 8pc image to imprve other blocking scales
#           --> Works better this way
#           see if 8pc SNR map can be recovered well*