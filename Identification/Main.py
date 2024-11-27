import os
import FilamentMap

def getDistance(fits_file):
    if('0628' in fits_file):
        return 9.84
    elif('4254' in fits_file):
        return 13.1
    elif('4303' in fits_file):
        return 16.99
    elif("SigmaHI" in fits_file):
         return 10 #Simulation
    else:
        print("improper file")

def setUp(galaxy_dir): #Assumes no data is a simulation for now
    #iterate through JWST files
    FilamentMapList = []
    for galaxy_folder in os.listdir(galaxy_dir):
            if(galaxy_folder == "ngc0628"): #remove later
                galaxy_folder = os.path.join(galaxy_dir, galaxy_folder)
                for fits_file in os.listdir(galaxy_folder):
                    if(fits_file.endswith(".fits") and "64pc" in fits_file): #remove later
                        #Make object
                        print(f"Setting Up: {fits_file}")
                        distance_pc = getDistance(fits_file)
                        ScalePix = .53328 * distance_pc
                        MyFilMap = FilamentMap.FilamentMap(ScalePix, galaxy_folder, fits_file) #change name later to be specific to the image
                        #get BkgSub Image
                        MyFilMap.SetBkgSub()
                        #Block the image
                        FilamentMapList.append(MyFilMap)
    return FilamentMapList


if __name__ == "__main__":
    #Folder with JWST images
    galaxy_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2"
    FilamentMapList = setUp(galaxy_dir)
    #iterate through JWST files
    for myFilMap in FilamentMapList:
        myFilMap.ScaleBkgSub() #TopVal not working well
        myFilMap.SetBlockData(Write = True)
        #Run Everything to get Composite
        myFilMap.RunSoax()
        myFilMap.SetIntensityMap(Orig = False)
        myFilMap.DisplayProbIntensityPlot(Orig = False)
        myFilMap.BlurComposite()
        for i in range(0, 256, 25):  # range(start, stop, step)
            myFilMap.ReHashComposite(ProbabilityThreshPercentile = i/255, minPixBoxSize = 35) 


#To Do: Update composite
