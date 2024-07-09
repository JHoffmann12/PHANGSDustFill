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


def plot_sobel_derivatives(Gx_map, Gy_map, save_path=r'C:\Users\HP\Documents\JHU_Academics\Research\PHANGS\sobel_derivatives.png'):
    try:
        Gx_copy = copy.deepcopy(Gx_map)
        Gy_copy = copy.deepcopy(Gy_map)
        
        Gx_copy = Gx_copy / np.abs(np.max(Gx_copy))
        Gy_copy = Gy_copy / np.abs(np.max(Gy_copy))

    # Create custom colormap with specific colors and range
        custom_palette = [
            (0, 0, 0),                # -90 degrees: black
            (30/255, 144/255, 255/255),  # -72 degrees: dark blue
            (0/255, 0/255, 255/255),    # -36 degrees: blue
            (173/255, 216/255, 230/255), # -18 degrees: light blue
            (1, 1, 1),            # 0 degrees: grey
            (255/255, 192/255, 203/255), # 18 degrees: light red
            (255/255, 0/255, 0/255),    # 36 degrees: red
            (139/255, 0/255, 0/255),    # 72 degrees: dark red
            (0, 0, 0)                   # 90 degrees: black
        ]
        cmap = mcolors.ListedColormap(custom_palette)
        cmap.set_under(color='white')  # Values less than -90 degrees mapped to light gray
        norm = mcolors.Normalize(vmin=-1, vmax=1)

        # Create figure with two subplots for Gx and Gy
        fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(6, 6), gridspec_kw={'width_ratios': [1, 1, 0.05]})
        
        # Plot Gx map with custom color spectrum
        im1 = ax1.imshow(Gx_copy, cmap=cmap, norm=norm)
        ax1.set_title('X Derivatives (Gx)')
        ax1.axis('off')
        
        # Plot Gy map with custom color spectrum
        im2 = ax2.imshow(Gy_copy, cmap=cmap, norm=norm)
        ax2.set_title('Y Derivatives (Gy)')
        ax2.axis('off')
        
        # Add colorbar for interpretation on the right
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, orientation='vertical')
        cbar.set_label('Scale')
        
        # Display the plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    except IOError:
        print(f"Unable to open or process")


def circular_vector_average(angles):
    """
    Calculate the circular mean of angles.
    """
    # Remove NaN values from angles
    angles = np.array(angles)
    angles = angles[~np.isnan(angles)]

    sin_sum = np.sum(np.sin(np.radians(angles)))
    cos_sum = np.sum(np.cos(np.radians(angles)))

    return np.degrees(np.arctan(sin_sum/cos_sum))

def plot_arctan_with_smoothing(Gx, Gy, mask, filter_size=5, save_path=None):
    """
    Plot smoothed arctan(Gy / Gx) values of every pixel normalized and represented with a custom color spectrum.
    Include a scale bar to show the color mapping.

    Parameters:
    - Gx: 2D array of x derivatives (Sobel filter result)
    - Gy: 2D array of y derivatives (Sobel filter result)
    - filter_size: Size of the smoothing filter
    - save_path: Path to save the output figure (optional)
    """
    # Calculate the angle map in degrees
    angle_map = -1*np.degrees(np.arctan(Gy/Gx))
    copy_angled_map = copy.deepcopy(angle_map)
    copy_angled_map[mask == 0] = np.nan

    # Apply circular vector average in a filter_size x filter_size window
    smoothed_angle_map = generic_filter(copy_angled_map, circular_vector_average, size=filter_size, mode='reflect')

    # Create custom colormap with specific colors and range
    custom_palette = [
        (0, 0, 0),                # -90 degrees: black
        (30/255, 144/255, 255/255),  # -72 degrees: dark blue
        (0/255, 0/255, 255/255),    # -36 degrees: blue
        (173/255, 216/255, 230/255), # -18 degrees: light blue
        (0.5, 0.5, 0.5),            # 0 degrees: grey
        (255/255, 192/255, 203/255), # 18 degrees: light red
        (255/255, 0/255, 0/255),    # 36 degrees: red
        (139/255, 0/255, 0/255),    # 72 degrees: dark red
        (0, 0, 0)                   # 90 degrees: black
    ]
    cmap = mcolors.ListedColormap(custom_palette)
    cmap.set_under(color='lightgray')  # Values less than -90 degrees mapped to light gray
    norm = mcolors.Normalize(vmin=-90, vmax=90)

    # Plotting for display
    plt.figure(figsize=(8, 6))
    plt.imshow(smoothed_angle_map, cmap=cmap, norm=norm)
    plt.colorbar(label='Angle (degrees)')
    plt.title('Smoothed Arctan of Gy / Gx')
    plt.axis('off')
    plt.show()

    # Normalize smoothed_angle_map to range [0, 1] for colormap mapping
    normalized_smoothed_map = (smoothed_angle_map - (-90)) / (90 - (-90))

    # Apply colormap
    smoothed_colored_map = cmap(normalized_smoothed_map)

    # Convert to uint8 for image saving
    smoothed_colored_map = (smoothed_colored_map[:, :, :3] * 255).astype(np.uint8)

    # Create mask for NaN values
    mask_nan = np.isnan(smoothed_angle_map)

    # Set NaN values to white (255, 255, 255)
    smoothed_colored_map[mask_nan] = [255, 255, 255]

    # Save the image
    cv2.imwrite(save_path, cv2.cvtColor(smoothed_colored_map, cv2.COLOR_RGB2BGR))
    print(f"Smoothed angle map saved as {save_path}")

    return smoothed_angle_map, copy_angled_map