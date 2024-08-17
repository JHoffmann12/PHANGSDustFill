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
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_sobel_derivatives(mask_image, Gx_map, Gy_map, save_path=r'C:\Users\HP\Documents\JHU_Academics\Research\PHANGS\sobel_derivatives.png'):
    try:
        Gx_copy = copy.deepcopy(Gx_map)
        Gy_copy = copy.deepcopy(Gy_map)
        
        # Normalize Gx_copy and Gy_copy
        Gx_copy = Gx_copy / np.abs(np.nanmax(Gx_copy))
        Gy_copy = Gy_copy / np.abs(np.nanmax(Gy_copy))

        mask_binary = (mask_image > 0).astype(np.uint8)  # Convert non-zero values to 1
        Gx_copy[mask_binary==0] = np.nan
        Gy_copy[mask_binary==0] = np.nan
        # Create custom colormap with specific colors and range
        custom_palette = [
            (0, 0, 0),                # -90 degrees: black
            (30/255, 144/255, 255/255),  # -72 degrees: dark blue
            (0/255, 0/255, 255/255),    # -36 degrees: blue
            (173/255, 216/255, 230/255), # -18 degrees: light blue
            (1,1,1),            # 0 degrees: grey
            (255/255, 192/255, 203/255), # 18 degrees: light red
            (255/255, 0/255, 0/255),    # 36 degrees: red
            (139/255, 0/255, 0/255),    # 72 degrees: dark red
            (0, 0, 0)                   # 90 degrees: black
        ]
        cmap = mcolors.ListedColormap(custom_palette)
        cmap.set_bad(color="lightgray")
        norm = mcolors.Normalize(vmin=-1, vmax=1)

       
        # Create figure with two subplots for Gx and Gy
        fig, (ax1, ax2, cax) = plt.subplots(1, 3, figsize=(8, 6), gridspec_kw={'width_ratios': [1, 1, 0.05]})
        
        # Plot Gx map with custom color spectrum
        im1 = ax1.imshow(np.flipud(Gx_copy), interpolation='nearest', cmap=cmap)
        ax1.set_title('X Derivatives (Gx)')
        ax1.axis('off')
        
        # Plot Gy map with custom color spectrum
        im2 = ax2.imshow(np.flipud(Gy_copy), interpolation='nearest', cmap=cmap)
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

# Example usage (replace with your mask, Gx_map, and Gy_map variables)
# plot_sobel_derivatives(mask, Gx_map, Gy_map)

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


def plot_arctan_with_smoothing(Gx, Gy, mask_image, filter_size=5, save_path=None):
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
    angle_map = -1 * np.degrees(np.arctan(Gy / Gx))
    copy_angled_map = copy.deepcopy(angle_map)

    mask_binary = (mask_image > 0).astype(np.uint8)  # Convert non-zero values to 1
    copy_angled_map[mask_binary > 0] = np.nan


    # Apply circular vector average in a filter_size x filter_size window
    smoothed_angle_map = generic_filter(copy_angled_map, circular_vector_average, size=filter_size, mode='reflect')

    x, y = np.shape(smoothed_angle_map)
    smoothed_angle_map[mask_binary == 1] = np.nan
    smoother_angle_map = copy.deepcopy(smoothed_angle_map)
    for r in range(x):
        for c in range(y):
            if mask_binary[r, c] == 1:
                smoother_angle_map[r, c] = average_neighbors(smoothed_angle_map, (r, c), filter_size)
            else:
                smoother_angle_map[r, c] = np.nan

    # Define custom palette with specified colors and transitions
    custom_palette = [
        (0.0, (0, 0, 0)),                # -90 degrees: black
        (0.1, (0, 0, 0)),                # -80 degrees: black
        (0.2, (30 / 255, 144 / 255, 255 / 255)),  # -72 degrees: dark blue
        (0.3, (0 / 255, 0 / 255, 255 / 255)),    # -36 degrees: blue
        (0.4, (173 / 255, 216 / 255, 230 / 255)), # -18 degrees: light blue
        (0.45, (1, 1, 1)),              # -10 degrees: white
        (0.55, (1, 1, 1)),              # 10 degrees: white
        (0.6, (255 / 255, 192 / 255, 203 / 255)), # 18 degrees: light red
        (0.7, (255 / 255, 0 / 255, 0 / 255)),    # 36 degrees: red
        (0.8, (139 / 255, 0 / 255, 0 / 255)),    # 72 degrees: dark red
        (0.9, (0, 0, 0)),                # 80 degrees: black
        (1.0, (0, 0, 0))                 # 90 degrees: black
    ]

    # Create a cyclic color map
    cyclic_cmap = mcolors.LinearSegmentedColormap.from_list("cyclic_custom", custom_palette, N=256)
    cyclic_cmap.set_bad("lightgrey")

    # Plotting for display
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot angle_map
    im0 = axs[0].imshow(np.flipud(angle_map), interpolation='nearest', cmap=cyclic_cmap)
    axs[0].set_title('Original Angle Map')
    axs[0].axis('off')

    # Plot smoothed_angle_map
    im1 = axs[1].imshow(np.flipud(smoothed_angle_map), interpolation='nearest', cmap=cyclic_cmap)
    axs[1].set_title('Smoothed Angle Map')
    axs[1].axis('off')

    # Plot smoother_angle_map
    im2 = axs[2].imshow(np.flipud(smoother_angle_map), interpolation='nearest', cmap=cyclic_cmap)
    axs[2].set_title('Infilled and Smoothed Angle Map')
    axs[2].axis('off')

    # Add one colorbar for all subplots to the right of the third plot
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im2, cax=cax, orientation='vertical').set_label('Angle (degrees)')

    plt.tight_layout()
    plt.show()

    # Normalize smoother_angle_map to range [0, 1] for colormap mapping
    normalized_smoother_map = (smoother_angle_map - (-90)) / (90 - (-90))

    # Apply colormap
    smoother_colored_map = cyclic_cmap(normalized_smoother_map)

    # Convert to uint8 for image saving
    smoother_colored_map = (smoother_colored_map[:, :, :3] * 255).astype(np.uint8)

    # Save the image if save_path is provided
    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(np.flipud(smoother_colored_map), cv2.COLOR_RGB2BGR))
        print(f"Smoothed angle map saved as {save_path}")

    return smoother_angle_map, smoothed_angle_map

def average_neighbors(data, point, half_size):
    x = point[0]
    y = point[1]
   # assert(data[x,y]==np.nan)
    sum = 0
    count = 0
    width,height = np.shape(data)
    # Iterate over a box_size x box_size box centered at (x, y)
    for i in range(-half_size, half_size + 1):  # Range from -half_size to half_size (inclusive)
        for j in range(-half_size, half_size + 1):  # Range from -half_size to half_size (inclusive)
            # Calculate the actual pixel coordinates
            pixel_x = x + i
            pixel_y = y + j
            # Check if the pixel is within bounds of the image
            if (pixel_x != x and pixel_y != y) and 0 <= pixel_x < width and 0 <= pixel_y < height and not np.isnan(data[pixel_x,pixel_y]):
                if (data[pixel_x,pixel_y] > 80 and sum < 0) or (data[pixel_x,pixel_y] < -80 and sum > 0): 
                    data[pixel_x,pixel_y] = -1*data[pixel_x,pixel_y]
                sum+=data[pixel_x,pixel_y]
                count+=1
    if count > 0:
        return sum/count
    else: 
        return np.nan
