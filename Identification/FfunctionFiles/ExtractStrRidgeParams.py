import os
import shutil
import subprocess 
import Ffunction 
from collections import Counter
import re
import numpy as np 

# Input and output directory paths
output_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\ngc0628\SOAXOutput\64pc\best_param1"

import os
import numpy as np
import re

# Initialize a dictionary to track the lowest f_score for each (t, c) combination
f_scores = {}
count = 0

# Regular expression to match the "ridgeXX--stretchYY" pattern
pattern = r"ridge\d+\.\d+--stretch\d+\.\d+"

# Loop over all files in the output directory
for file in os.listdir(output_dir):
    if file.endswith('.txt'):  
        file_path = os.path.join(output_dir, file)
        
        # Extract the "ridgeXX--stretchYY" string from the file name
        match = re.search(pattern, file)
        if match:
            tbd_value = match.group(0)  # Extract the matched string
        else:
            tbd_value = "Unknown"  # Use "Unknown" if the pattern is not found
            print(file)
            print("Fail")
        
        for t in np.arange(1, 4, 0.5):  # t in the range [1, 4) with step 0.5
            for c in np.arange(1, 4, 0.5):  # c in the range [1, 4) with step 0.5
                # Compute f_score using your Ffunction.fFunction method
                f_score = Ffunction.fFunction(file_path, t, c)

                # If this combination of (t, c) hasn't been seen yet, or if the current f_score is lower, update the dictionary
                if (t, c) not in f_scores or f_score < f_scores[(t, c)][0]:
                    f_scores[(t, c)] = (f_score, tbd_value)  # Store both f_score and tbd_value

        count += 1
        print(f"{count} files complete")

# Count the occurrences of each "ridgeXX--stretchYY" value in the f_scores dictionary
tbd_counts = {}
for _, (_, tbd_value) in f_scores.items():
    tbd_counts[tbd_value] = tbd_counts.get(tbd_value, 0) + 1

# Sort the dictionary by count in descending order and take the top 10
top_10_tbd = sorted(tbd_counts.items(), key=lambda item: item[1], reverse=True)[:10]

# Print the results
print("Top 10 most frequent 'ridgeXX--stretchYY' values:")
for tbd_value, count in top_10_tbd:
    print(f'Params: "{tbd_value}", Count: {count}')






