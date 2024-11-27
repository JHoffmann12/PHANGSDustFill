import os
import shutil
import subprocess 
import Ffunction 
from collections import Counter
import re
import numpy as np 

# Input and output directory paths
parent_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\ngc0628\SOAXOutput\32pc"
output_dir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\ngc0628\32pcCandidates"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List to store all new filenames
new_filenames = []

# Iterate through each folder in the parent directory
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Process each .txt file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                
                # Remove ".fits" from the filename before appending the folder name
                base_name = filename.replace(".fits", "")  # Remove the ".fits" extension part
                new_filename = f"{folder_name}_{base_name}"
                
                # Ensure the final file has the ".txt" extension (in case ".fits" was embedded)
                if not new_filename.endswith(".txt"):
                    new_filename += ".txt"
                
                # Define the new file path in the output directory
                new_file_path = os.path.join(output_dir, new_filename)
                
                # Copy the original file to the new path with the new name
                shutil.copy(file_path, new_file_path)
                
                # Append the new filename to the list
                new_filenames.append(new_filename)

# Initialize a dictionary to track the lowest f_score for each (t, c) combination
f_scores = {}
count = 0
# Loop over all files in the output directory
for file in os.listdir(output_dir):
    if file.endswith('.txt'):  # Optional: filter files if needed
        # Extract the value of x from the filename using regex (e.g., "best_paramx")
        file_path = os.path.join(output_dir, file)
        match = re.search(r'best_param(\d+)', file)
        if match:
            x_value = match.group(1)  # Extracted value of x

            for t in np.arange(1, 4, 0.5):  # t in the range [1, 4) with step 0.1
                for c in np.arange(1, 4, 0.5):  # c in the range [1, 4) with step 0.1
                    # Compute f_score using your Ffunction.fFunction method
                    f_score = Ffunction.fFunction(file_path, t, c)

                    # If this combination of (t, c) hasn't been seen yet, or if the current f_score is lower, update the dictionary
                    if (t, c) not in f_scores or f_score < f_scores[(t, c)][0]:
                        f_scores[(t, c)] = (f_score, x_value)  # Store both f_score and x_value
    count+=1
    print(f"{count} files complete")
# Now we need to parse the f_scores dictionary and find the most common x_value
x_values = [x_value for _, (_, x_value) in f_scores.items()]

# Use Counter to find the most common x_value
x_counter = Counter(x_values)

# Get the most common x_value
most_common_x, most_common_count = x_counter.most_common(1)[0]

print(f"The most common x_value is: {most_common_x} with {most_common_count} occurrences")

sorted_x_counts = sorted(x_counter.items(), key=lambda x: x[1])
# Print each x_value with its count
print("x values and their counts:")
for x, count in sorted_x_counts:
    print(f"{x}: {count}")


# print("Text files have been renamed, saved to the output directory, and filenames saved to 64pcCandidateFileNames.txt.")

# #Run SOAX F function

# inputImage = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\ngc0628\BlockedPng\ngc0628_F770W_starsub_anchored_CDDss0064pc_arcsinh0p1.fits_Blocked.png"
# soaxDir = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\ngc0628\64pcCandidates"
# best_snake = r"C:\Users\HP\Downloads\best_snake_v3.7.0.exe"
# errorOut = r"C:\Users\HP\Documents\JHU_Academics\Research\Soax_results_blocking_V2\ngc0628\64pcErrorCandidates.txt"
# print("starting Soax")
# cmdString = f'"{best_snake}" -i {inputImage} -s "{soaxDir}" -o "{candidate_file_path}" -n 2 -f 5 -e {errorOut}'
# subprocess.run(cmdString, shell=True)
