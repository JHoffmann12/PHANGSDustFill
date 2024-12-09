import os
import FilamentMap
import numpy as np 
import matplotlib.pyplot as plt

def getDistance(fits_file):
    if('0628' in fits_file):
        return 9.84, "ngc0628"
    elif('4254' in fits_file):
        return 13.1, "ngc0254"
    elif('4303' in fits_file):
        return 16.99, "ngc4303"
    elif("SigmaHI" in fits_file):
         return 10, "Sim" #Simulation
    else:
        print("improper file")

def setUp(galaxy_dir): #Assumes no data is a simulation for now
    #iterate through JWST files
    FilamentMapList = []
    for galaxy_folder in os.listdir(galaxy_dir):
             if(galaxy_folder == "ngc0628"): #remove later
                galaxy_folder = os.path.join(galaxy_dir, galaxy_folder)
                for fits_file in os.listdir(galaxy_folder):
                    if(fits_file.endswith(".fits")): 
                        #Make object
                        print(f"Setting Up: {fits_file}")
                        distance_pc, Galaxy = getDistance(fits_file)
                        ScalePix = .53328 * distance_pc
                        MyFilMap = FilamentMap.FilamentMap(ScalePix, galaxy_folder, fits_file, Galaxy) #change name later to be specific to the image
                        MyFilMap.SetBlockData(Write = True)
                        #get BkgSub Image
                        MyFilMap.SetBkgSub(Sim = True)
                        #Block the image
                        FilamentMapList.append(MyFilMap)
    return FilamentMapList

def displaySNRPlot(FilamentMapList, galaxy_dir, Write = False, verbose = False):

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
        galaxy_dict[galaxy].append((scale, np.median(SNRMap))) 

    # Plotting scatter plots for each galaxy
    for galaxy, data in galaxy_dict.items():
        scales, percentiles = zip(*data)  # Unpack scales and percentiles
        plt.figure()
        plt.scatter(scales, percentiles, label=f"Galaxy: {galaxy}")
        plt.xlabel("Scale (pc)")
        plt.ylabel("Median Noise")
        plt.title(f"SNR Plot for Galaxy: {galaxy}")
        plt.legend()
        plt.grid(True)

        if Write:
            plt.savefig(f"{galaxy_dir}\Figures\SNRPlot_{galaxy}.png")
        if(verbose):
            plt.show()


if __name__ == "__main__":
    #Folder with JWST images
    galaxy_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2"
    FilamentMapList = setUp(galaxy_dir)
    displaySNRPlot(FilamentMapList, galaxy_dir, Write = True, verbose = True)
    #iterate through JWST files
    for myFilMap in FilamentMapList:
        myFilMap.ScaleBkgSub() #TopVal not working well
        #Run Everything to get Composite
        myFilMap.RunSoax()
        # myFilMap.CreateComposite("best_param1") #either this or run soax
        myFilMap.SetIntensityMap(Orig = False)
        myFilMap.BlurComposite(set_blur_as_prob = True)
        myFilMap.DisplayProbIntensityPlot(galaxy_dir, Orig = False, Write = True)
        myFilMap.ReHashComposite(ProbabilityThreshPercentile = 90/255, minPixBoxSize = 35)

#Increase PSF before scale decomposition?