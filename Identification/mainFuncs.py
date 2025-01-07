import os
import FilamentMap
import numpy as np 
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for thread safety
import matplotlib.pyplot as plt
from astropy.table import Table
import re
from astropy.io import fits
from astropy.coordinates import EarthLocation
import astropy.units as u  # Add this line to import the units


def fixFits(fits_path):
    return

def getDistance(Galaxy, fits_file, dist_table_path):
    dist_table = Table.read(dist_table_path, format='ascii')
    print("Distance table loaded successfully.")

    # Dynamic galaxy name matching
    print(f'Fits file: {fits_file}')

    filename = fits_file.split('/')[-1]
    # Update the regular expression to match either "ngc\d+" or "Sim"
    match = re.search(Galaxy.lower(), filename.lower())  # Match 'ngc' followed by digits or 'sim' followed by alphanumeric before an underscore

    if match:
        galaxytab = Galaxy  # Extracted galaxy name
        print(f"Extracted galaxy name: {galaxytab}")

        # Check if galaxy exists in the distance table
        if galaxytab in dist_table['galaxy']:
            distance_mpc = dist_table['current_dist'][dist_table['galaxy'] == galaxytab][0]
            print(f"Galaxy {galaxytab} found in distance table: {distance_mpc} Mpc away!")
            return distance_mpc
        else:
            print(f"Galaxy name '{galaxytab}' not found in the distance table.")
            exit()
    else:
        print(f"Galaxy name could not be extracted from filename: {filename}")
        exit()

def setUp(galaxy_dir): #Assumes no data is a simulation for now
    #iterate through JWST files
    FilamentMapList = []
    for galaxy_folder in os.listdir(galaxy_dir):
        galaxy_path = os.path.join(galaxy_dir, galaxy_folder)  # Get the full path
        if not os.path.isdir(galaxy_path):  # Skip if it's not a directory
            continue
        if(galaxy_folder == "ngc1433" and galaxy_folder != 'OriginalMiriImages'): #remove later
            print(galaxy_folder)
            Galaxy = galaxy_folder
            galaxy_folder = os.path.join(galaxy_dir, galaxy_folder)
            for fits_file in os.listdir(galaxy_folder):
                if(fits_file.endswith(".fits")): 
                    #Make object
                    # fixFits(os.path.join(galaxy_folder, fits_file)) #avoids astropy warnings by filling header content with B.S. 
                    print(f"Setting Up: {fits_file}")
                    dist_table_path = os.path.join(galaxy_dir, "DistanceTable.txt")
                    distance_mpc = getDistance(Galaxy, fits_file, dist_table_path)
                    ScalePix = .53328 * distance_mpc
                    MyFilMap = FilamentMap.FilamentMap(ScalePix, galaxy_folder, fits_file, Galaxy) #change name later to be specific to the image
                    # MyFilMap.setBlockFactor(4)
                    MyFilMap.SetBlockData(Write = True)
                    #get BkgSub Image
                    MyFilMap.SetBkgSub()
                    #Block the image
                    FilamentMapList.append(MyFilMap)
    return FilamentMapList

def CreateSNRPlot(FilamentMapList, galaxy_dir, Write = False, verbose = False, noisePlot = False):

    galaxy_dict = {}
    # Iterate over each object and group them
    for myFilMap in FilamentMapList:
        galaxy = myFilMap.getGalaxy()
        if galaxy not in galaxy_dict:
            galaxy_dict[galaxy] = []
        SNRMap = myFilMap.getBkgSubMap()
        #to add to plot:
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
        plt.title(f"SNR Plot for Galaxy: {galaxy} without normalization and using unique masks")
        plt.legend()
        plt.grid(True)

        if Write:
            plt.savefig(f"{galaxy_dir}\Figures\SNRPlot_{galaxy}.png")
        if(verbose):
            plt.show()

    if (noisePlot):
        galaxy_dict = {}
        # Iterate over each object and group them
        for myFilMap in FilamentMapList:
            galaxy = myFilMap.getGalaxy()
            if galaxy not in galaxy_dict:
                galaxy_dict[galaxy] = []
            #to add to plot:
            scale = myFilMap.getScale()
            scale = scale.replace('pc', "")
            scale = int(scale)
            galaxy_dict[galaxy].append((scale, np.percentile(myFilMap.NoiseMap, 3))) 

        # Plotting scatter plots for each galaxy
        for galaxy, data in galaxy_dict.items():
            scales, percentiles = zip(*data)  # Unpack scales and percentiles
            plt.figure()
            plt.scatter(scales, percentiles, label=f"Galaxy: {galaxy}")
            plt.xlabel("Scale (pc)")
            plt.ylabel("Lowest 3 percentile Noise")
            plt.title(f"Noise Plot for Galaxy: {galaxy} using unique masks")
            plt.legend()
            plt.grid(True)

            if Write:
                plt.savefig(f"{galaxy_dir}\Figures\PNoisePlot_{galaxy}.png")
            if(verbose):
                plt.show()