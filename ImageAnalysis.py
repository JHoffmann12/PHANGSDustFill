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

def identify_intersects(image_path, output_path,dot_size=8,box_size=10,perc = .4, RGBA_color = (100, 255, 100, 200), title = 'Processed Image with Green Dots'):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Convert the image to grayscale if not already
            img = img.convert("L")
            
            # Get image size
            width, height = img.size
            print(img.size)
            # Create a new image for processing
            processed_img = img.copy()
            draw = ImageDraw.Draw(processed_img)
            
            # List to store coordinates of green dots
            green_dots = []
            
            # Identify crosses or tridents (black regions)
            for x in range(width):
                for y in range(height):
                    pixel_value = img.getpixel((x, y))
                    if pixel_value == 0:  # Black pixel
                        # Check for a cross or trident pattern
                        if is_intersect(img, x, y, width, height,box_size,perc):
                            # Store coordinates of green dot
                            green_dots.append((x, y))
            
            # Create a new image with transparent background
            green_dot_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw_green = ImageDraw.Draw(green_dot_img)
            
            # Draw green dots on transparent image
            for dot in green_dots:
                x, y = dot
                draw_green.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill= RGBA_color)
            
            # Merge processed_img and green_dot_img
            processed_img = Image.alpha_composite(processed_img.convert('RGBA'), green_dot_img)
            processed_img = processed_img.convert('RGB')
            
            # Display both images side by side
            fig, axes = plt.subplots(1, 2, figsize=(6, 6))
            
            # Plot original image
            axes[0].imshow(img, cmap='gray')
            axes[0].axis('off')  # Turn off axis numbers and ticks
            axes[0].set_title('Original Image')
            
            # Plot processed image with green dots
            axes[1].imshow(processed_img)
            axes[1].axis('off')  # Turn off axis numbers and ticks
            axes[1].set_title(title)
            
            plt.tight_layout()
            plt.show()
            
            # Save the processed image with green dots
            processed_img.save(output_path)
            print(f"Processed {image_path} successfully. Saved as {output_path}")
    
    except IOError:
        print(f"Unable to open or process {image_path}")

def is_intersect(image, x, y, width, height, box_size=15, perc=.5):
    half_size = box_size // 2  # Half of the box size
    count = 0
    
    # Iterate over a box_size x box_size box centered at (x, y)
    for i in range(-half_size, half_size + 1):  # Range from -half_size to half_size (inclusive)
        for j in range(-half_size, half_size + 1):  # Range from -half_size to half_size (inclusive)
            # Calculate the actual pixel coordinates
            pixel_x = x + i
            pixel_y = y + j
            
            # Check if the pixel is within bounds of the image
            if (pixel_x != x and pixel_y != y) and 0 <= pixel_x < width and 0 <= pixel_y < height:
                # Check if the pixel is black (value 0)
                if image.getpixel((pixel_x, pixel_y)) == 0:
                    count += 1
    
    # Return True if the count of black pixels exceeds the threshold
    return count >= box_size * box_size*perc


def identify_connected_components(image_path):

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Perform connected components labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    
    # Convert grayscale image to RGB for drawing colored outlines
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Define RGBA color for the outline (light orange with transparency)
    color_orange = (255, 165, 0, 150)  # (B, G, R, Alpha)
    
    # Draw outlines around each connected component
    for label in range(1, num_labels):
        # Find region coordinates
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        
        # Draw rectangle around the connected component with RGBA color
        cv2.rectangle(image_rgb, (left, top), (left + width, top + height), color_orange, 2)
    
    # Create a figure with two subplots (original and with outlines)
    fig, axes = plt.subplots(1, 2, figsize=(6, 6))
    
    # Display the original image on the left subplot
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display the image with outlines on the right subplot
    axes[1].imshow(image_rgb)
    axes[1].set_title('Connected Components Outlined')
    axes[1].axis('off')
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    
    # Return labels and stats
    return labels, stats



def apply_sobel_filter_to_components(image_path, labels, stats):
    """
    Apply Sobel filter to each connected component identified by labels and stats.
    Calculate Gx and Gy for every pixel in each component.

    Parameters:
    - image_path: Path to the grayscale image
    - labels: Image where each connected component is labeled
    - stats: Statistics about each connected component (leftmost, topmost, width, height, area)

    Returns:
    - Gx_total: Total x derivatives (Sobel filter result) for the entire image
    - Gy_total: Total y derivatives (Sobel filter result) for the entire image
    """
    # Read the image
    # read the image
    img = cv2.imread(r'C:\Users\HP\Documents\JHU_Academics\Research\PHANGS\ThinSkeleton1Intersects.png')

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (0,0), 1.3, 1.3)

    # Initialize Gx and Gy matrices to store results for the entire image
    Gx_total = np.zeros_like(blur, dtype=np.float32)
    Gy_total = np.zeros_like(blur, dtype=np.float32)

    # Loop through each connected component
    for label_id in range(1, np.max(labels) + 1):
        # Extract the bounding box coordinates
        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
        # Create a mask for the current connected component
        mask = (labels == label_id).astype(np.uint8)
        
        # Apply Sobel filter to the masked component
        masked_image = blur * (~mask)
        
        # Apply Sobel filter to every pixel in the masked component
        sobelx = cv2.Sobel(masked_image,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(masked_image,cv2.CV_64F,0,1,ksize=3)
        
        # Accumulate Gx and Gy values into the total matrices
        Gx_total[top:top+height, left:left+width] = sobelx[top:top+height, left:left+width]
        Gy_total[top:top+height, left:left+width] = sobely[top:top+height, left:left+width]
        
    fig, axs = plt.subplots(1, 2, figsize=(8, 8))
    print(min(Gx_total.flatten()))
    print(max(Gy_total.flatten()))
    axs[0].imshow(Gx_total)
    axs[1].imshow(Gy_total)

    return (Gx_total), (Gy_total)