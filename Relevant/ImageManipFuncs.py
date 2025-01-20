from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage.exposure as exposure
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import generic_filter
from scipy import ndimage
import copy 

def invert_image(image_path, output_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Convert image to grayscale (mode 'L' for black and white)
            img = img.convert('L')

            # Apply thresholding: pixels below 128 are black (0), pixels above or equal to 128 are white (255)
            thresholded_img = Image.eval(img, lambda px: 0 if px > 128 else 255)

            # Save the thresholded image
            thresholded_img.save(output_path)

            # Show the thresholded image (optional)
            #thresholded_img.show()

            print(f"Thresholded image saved as {output_path}")

    except IOError:
        print(f"Unable to open or process {image_path}")
        
def convert_red_to_black(image_path, output_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Convert the image to RGB mode (if not already in RGB)
            img = img.convert("RGB")
            
            # Get the image size
            width, height = img.size
            
            # Create a new image with the same size and white background
            converted_img = Image.new('RGB', (width, height), (255, 255, 255))
            
            # Iterate over each pixel in the original image
            for x in range(width):
                for y in range(height):
                    # Get the RGB values of the pixel
                    r, g, b = img.getpixel((x, y))
                    
                    # Check if the pixel is red
                    if r > 150 and g < 100 and b < 100:
                        # If red, set it to black
                        converted_img.putpixel((x, y), (0, 0, 0))
                    else:
                        # Otherwise, set it to white
                        converted_img.putpixel((x, y), (255, 255, 255))
            
            # Display both images side by side
            fig, axes = plt.subplots(1, 2, figsize=(6, 6))
            
            # Plot original image
            axes[0].imshow(img)
            axes[0].axis('off')  # Turn off axis numbers and ticks
            axes[0].set_title('Original Image')
            
            # Plot modified image (red to black)
            axes[1].imshow(converted_img)
            axes[1].axis('off')  # Turn off axis numbers and ticks
            axes[1].set_title('Modified Image (Red to Black)')
            
            plt.tight_layout()
            plt.show()
            
            # Save the modified image
            converted_img.save(output_path)
            print(f"Converted {image_path} successfully. Saved as {output_path}")
    
    except IOError:
        print(f"Unable to open or process {image_path}")

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
import networkx as nx
from skimage import measure

def skeleton_analysis(skeleton, image, prune=False, prune_criteria='length', relintens_thresh=0.5,
                      branch_thresh=None, verbose=False, save_png=False, save_name=None):
    """
    Analyze a skeletonized filament and return the longest path after optional pruning.
    
    Parameters:
    - skeleton: A 2D numpy array representing the skeletonized filament.
    - image: The original image (2D numpy array) from which the skeleton was extracted.
    - prune: Boolean flag to determine whether pruning is applied.
    - prune_criteria: Criteria for pruning ('length' or 'intensity').
    - relintens_thresh: Relative intensity threshold for pruning if pruning by intensity.
    - branch_thresh: Length threshold for pruning if pruning by length.
    - verbose: If True, display the skeleton images.
    - save_png: If True, save the images.
    - save_name: Name prefix for saved images.

    Returns:
    - longest_path_skeleton: A 2D numpy array of the longest path in the pruned skeleton.
    """

    # Ensure skeleton is binary
    skeleton = (skeleton > 0).astype(int)

    if verbose:
        plt.imshow(skeleton, cmap='gray')
        plt.title("Original Skeleton")
        plt.show()

    # Pruning based on the criteria
    if prune:
        if prune_criteria == 'length':
            if branch_thresh is not None:
                skeleton = remove_small_objects(skeleton.astype(bool), min_size=branch_thresh).astype(int)
        elif prune_criteria == 'intensity':
            labeled_skeleton, num_features = measure.label(skeleton, return_num=True)
            for region in measure.regionprops(labeled_skeleton, intensity_image=image):
                if region.mean_intensity < relintens_thresh:
                    skeleton[labeled_skeleton == region.label] = 0

        if verbose:
            plt.imshow(skeleton, cmap='gray')
            plt.title("Pruned Skeleton")
            plt.show()

    # Convert skeleton to graph
    graph = nx.Graph()
    skeleton_points = np.argwhere(skeleton)

    for point in skeleton_points:
        graph.add_node(tuple(point))

    for point in skeleton_points:
        for neighbor in [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]:
            neighbor_point = point + neighbor
            if tuple(neighbor_point) in graph:
                graph.add_edge(tuple(point), tuple(neighbor_point))

    # Find the longest path in the pruned skeleton
    longest_path_length = 0
    longest_path_coords = []

    if len(graph.nodes) > 0:
        for source in graph.nodes():
            for target in graph.nodes():
                if source != target:
                    try:
                        length = nx.dijkstra_path_length(graph, source, target)
                        if length > longest_path_length:
                            longest_path_length = length
                            longest_path_coords = nx.dijkstra_path(graph, source, target)
                    except nx.NetworkXNoPath:
                        continue

    # Create the longest path skeleton
    longest_path_skeleton = np.zeros_like(skeleton)
    for coord in longest_path_coords:
        longest_path_skeleton[coord] = 1

    if verbose:
        plt.imshow(longest_path_skeleton, cmap='gray')
        plt.title("Longest Path in Skeleton")
        plt.show()

    if save_png and save_name:
        plt.imsave(f"{save_name}_original_skeleton.png", skeleton, cmap='gray')
        if prune:
            plt.imsave(f"{save_name}_pruned_skeleton.png", skeleton, cmap='gray')
        plt.imsave(f"{save_name}_longest_path.png", longest_path_skeleton, cmap='gray')

    return longest_path_skeleton




def skeleton_Prune(skeleton, image, prune_criteria='combined', relintens_thresh=0.1,
                      branch_thresh=5, verbose=False):

    # Ensure skeleton is binary
    skeleton = (skeleton > 0).astype(int)

    if verbose:
        plt.imshow(skeleton, cmap='gray')
        plt.title("Original Skeleton")
        plt.show()

    # Pruning based on the criteria
    if prune_criteria == 'length':
        if branch_thresh is not None:
            skeleton = remove_small_objects(skeleton.astype(bool), min_size=branch_thresh).astype(int)
    elif prune_criteria == 'intensity':
        labeled_skeleton, num_features = measure.label(skeleton, return_num=True)
        for region in measure.regionprops(labeled_skeleton, intensity_image=image):
            if region.mean_intensity < relintens_thresh:
                skeleton[labeled_skeleton == region.label] = 0
    elif prune_criteria == 'combined':
        if branch_thresh is not None and relintens_thresh is not None:
            # Label the skeleton
            labeled_skeleton, num_features = measure.label(skeleton, return_num=True)
            for region in measure.regionprops(labeled_skeleton, intensity_image=image):
                # Check length (area of the region)
                if region.area < branch_thresh or region.mean_intensity < relintens_thresh:
                    skeleton[labeled_skeleton == region.label] = 0


        if verbose:
            plt.imshow(skeleton, cmap='gray')
            plt.title("Pruned Skeleton")
            plt.show()

    return skeleton


