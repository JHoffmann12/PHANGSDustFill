#https://archive.stsci.edu/hlsp/phangs.html#hst_image_products_table
#https://stsci.app.box.com/s/72k2y701zd5x7v72zaymopcd4jp02ftu
import os
import FilamentMap
import numpy as np 
import matplotlib.pyplot as plt
import mainFuncs
import Modified_Constrained_Diffusion


if __name__ == "__main__":
    #Folder with JWST image
    galaxy_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2"
    decomposition_exists = True
    if(not decomposition_exists):
        Modified_Constrained_Diffusion.decompose(galaxy_dir)
    FilamentMapList = mainFuncs.setUp(galaxy_dir)
    mainFuncs.CreateSNRPlot(FilamentMapList, galaxy_dir, Write = True, verbose = False)

    #iterate through JWST files
    for myFilMap in FilamentMapList:
        #Run Everything to get Composite
        myFilMap.ScaleBkgSub()
        myFilMap.RunSoax()
        # myFilMap.CreateComposite("best_param1") #either this or run soax
        myFilMap.BlurComposite(set_blur_as_prob = True)
        myFilMap.SetIntensityMap(Orig = False)
        myFilMap.DisplayProbIntensityPlot(galaxy_dir, Orig = False, Write = True, verbose = False)
        myFilMap.ReHashComposite(ProbabilityThreshPercentile = .33, minPixBoxSize = 100) #formerly 75 increased to 100 for sim

        #comments: Parameters take way too much structure for sim, need to increase min intensity


