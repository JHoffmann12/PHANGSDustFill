import os
import FilamentMap
import numpy as np 
import matplotlib.pyplot as plt
from astropy.table import Table
import re


def getDistance(fits_file, dist_table_path):
        dist_table = Table.read(dist_table_path, format='ascii')
        print("Distance table loaded successfully.")

        # Dynamic galaxy name matching
        print(f'Fits file: {fits_file}')
        filename = fits_file.split('/')[-1]
        match = re.search(r'(ngc\d+)_', filename.lower())
        # if match:
        galaxytab = match.group(1)  # Extracted galaxy name (e.g., 'ngc0628')
        distance_mpc = dist_table['current_dist'][dist_table['galaxy'] == galaxytab][0]
        print(f"Extracted galaxy name: {galaxytab} at {distance_mpc} Mpc away!")
        return distance_mpc, galaxytab
        # else:
        #     print(f"Galaxy name could not be extracted from filename: {filename}")
        #     exit()


def setUp(galaxy_dir): #Assumes no data is a simulation for now
    #iterate through JWST files
    FilamentMapList = []
    for galaxy_folder in os.listdir(galaxy_dir):
        galaxy_path = os.path.join(galaxy_dir, galaxy_folder)  # Get the full path
        if not os.path.isdir(galaxy_path):  # Skip if it's not a directory
            continue
        if(galaxy_folder == "ngc0628" and galaxy_folder != 'OriginalMiriImages'): #remove later
            print(galaxy_folder)
            galaxy_folder = os.path.join(galaxy_dir, galaxy_folder)
            for fits_file in os.listdir(galaxy_folder):
                if(fits_file.endswith(".fits")): 
                    #Make object
                    print(f"Setting Up: {fits_file}")
                    dist_table_path = os.path.join(galaxy_dir, "DistanceTable.txt")
                    distance_mpc, Galaxy = getDistance(fits_file, dist_table_path)
                    ScalePix = .53328 * distance_mpc
                    MyFilMap = FilamentMap.FilamentMap(ScalePix, galaxy_folder, fits_file, Galaxy) #change name later to be specific to the image
                    MyFilMap.SetBlockData(Write = True)
                    #get BkgSub Image
                    MyFilMap.SetBkgSub(Sim = True)
                    #Block the image
                    FilamentMapList.append(MyFilMap)
    return FilamentMapList

def CreateSNRPlot(FilamentMapList, galaxy_dir, Write = False, verbose = False):

    galaxy_dict = {}

    # Iterate over each object and group them
    for myFilMap in FilamentMapList:
        galaxy = myFilMap.getGalaxy()
        if galaxy not in galaxy_dict:
            galaxy_dict[galaxy] = []
        SNRMap = myFilMap.getBkgSubMap()
        scale = myFilMap.getScale()
        scale = scale.replace('pc', "")
        scale = int(scale)
        galaxy_dict[galaxy].append((scale, np.percentile(SNRMap, 99))) 

    # Plotting scatter plots for each galaxy
    for galaxy, data in galaxy_dict.items():
        scales, percentiles = zip(*data)  # Unpack scales and percentiles
        plt.figure()
        plt.scatter(scales, percentiles, label=f"Galaxy: {galaxy}")
        plt.xlabel("Scale (pc)")
        plt.ylabel("SNR 99 percentile")
        plt.title(f"SNR Plot for Galaxy: {galaxy} without normalization")
        plt.legend()
        plt.grid(True)

        if Write:
            plt.savefig(f"{galaxy_dir}\Figures\SNRPlot_{galaxy}.png")
        if(verbose):
            plt.show()
