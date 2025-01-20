import numpy as np
import cv2
from scipy.stats import wasserstein_distance 
# Helper functions
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math 

import numpy as np

def compute_emd(hist1, hist2):

    hist1 = np.array(hist1)
    hist2 = np.array(hist2)

    # Array where angles between 45 and 90 degrees are subtracted from 180 degrees
    angles_45_to_90_1 = -180 + hist1
    angles_45_to_90_1 = angles_45_to_90_1[(hist1 >= 45) & (hist1 <= 90)]
    # Array where angles between -45 and -90 degrees are added to 180 degrees
    angles_minus45_to_minus90_1 = 180 + hist1
    angles_minus45_to_minus90_1 = angles_minus45_to_minus90_1[(hist1 >= -90) & (hist1 <= -45)]
    # Concatenate the original array with the two new arrays
    extended_1 = np.concatenate([hist1, angles_45_to_90_1, angles_minus45_to_minus90_1])
    # Convert histograms to float32 as required by cv2.EMD
    extended_1 = extended_1.astype(np.float32)

    # Array where angles between 45 and 90 degrees are subtracted from 180 degrees
    angles_45_to_90_2 = -180 + hist2
    angles_45_to_90_2 = angles_45_to_90_2[(hist2 >= 45) & (hist2 <= 90)]
    # Array where angles between -45 and -90 degrees are added to 180 degrees
    angles_minus45_to_minus90_2 = 180 + hist2
    angles_minus45_to_minus90_2 = angles_minus45_to_minus90_2[(hist2 >= -90) & (hist2 <= -45)]
    # Concatenate the original array with the two new arrays
    extended_2 = np.concatenate([hist2, angles_45_to_90_2, angles_minus45_to_minus90_2])
    # Convert histograms to float32 as required by cv2.EMD
    extended_2 = extended_2.astype(np.float32)

    emd = wasserstein_distance(extended_1, extended_2, u_weights=None, v_weights=None)
    
    return emd

def get_combinations(s, num):
    # Generate all combinations of length 'num'
    all_combs = itertools.combinations(s, num)
    # Convert each combination to a sorted tuple to remove duplicates
    unique_combs = set(tuple(sorted(c)) for c in all_combs)
    # Convert each sorted tuple back to a list
    return [list(c) for c in unique_combs]

def list_combinations(list_a, list_b):
    combinations = [(b, a) for a in list_a for b in list_b]
    return combinations

def compute_my_metric(data_dict):
    # Compute histogram
    segments = data_dict.keys()
    combinations = get_combinations(segments, 2)

    metric_list = []
    weight_list = []

    for pair in combinations:
        pair0 = pair[0]
        pair1 = pair[1]

        metric = compute_emd(data_dict[pair0][0],data_dict[pair1][0])
        weight_list.append(data_dict[pair0][1]*data_dict[pair1][1])
        metric_list.append(metric)

    # Calculate the weighted sum
    total_sum = sum(w * m for w, m in zip(metric_list, weight_list))

    # Calculate the sum of weights
    sum_weights = sum(weight_list)
    
    return (total_sum / sum_weights)



# hist1 = [50,50,50,50]

# hist2 = [-50,-50,-50,-50]
print(compute_emd([1],[np.nan]))