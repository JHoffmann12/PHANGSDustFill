from PIL import Image, ImageDraw
import math
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
            # Create a new image for processing
            processed_img = img.copy()
            draw = ImageDraw.Draw(processed_img)
            
            # List to store coordinates of green dots
            green_dots = []
            
            # Identify crosses or tridents (black regions)
            for x in range(width):
                for y in range(height):
                    pixel_value = img.getpixel((x, y))
                    if pixel_value > .9:  # White pixel
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
            #axes[0].set_title('Original Image')
            
            # Plot processed image with green dots
            axes[1].imshow(processed_img)
            axes[1].axis('off')  # Turn off axis numbers and ticks
            #axes[1].set_title(title)
            
            plt.tight_layout()
            plt.show()
            
            # Save the processed image with green dots
            processed_img.save(output_path)
            print(f"Processed {image_path} successfully. Saved as {output_path}")

            return green_dots
    
    except IOError:
        print(f"Unable to open or process {image_path}")





from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# def remove_junctions1(junctions, img, output_path, dot_size=8):
#         # Get image size
#         width, height = img.size
#         # Create a new image for processing
#         processed_img = img.copy()
#         processed_img = Image.fromarray(processed_img )

#         draw = ImageDraw.Draw(processed_img)
        
#         # Draw circles and make pixels black within the circle
#         for dot in junctions:
#             x, y = dot
#             draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=255)
            
#             # Make any white pixels within the circle black
#             for i in range(x - dot_size, x + dot_size + 1):
#                 for j in range(y - dot_size, y + dot_size + 1):
#                     if (i - x) ** 2 + (j - y) ** 2 <= dot_size ** 2:
#                         if 0 <= i < width and 0 <= j < height:  # Check boundaries
#                             if img.getpixel((i, j)) == 255:  # Check if pixel is white
#                                 draw.point((i, j), fill=127)  # Make pixel black

#         # Display both images side by side
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
#         # Plot original image
#         axes[0].imshow(np.flipud(img), cmap='gray')
#         axes[0].axis('off')  # Turn off axis numbers and ticks
#         axes[0].set_title('Original Image')
        
#         # Plot processed image with circles
#         axes[1].imshow(np.flipud(processed_img), cmap='gray')
#         axes[1].axis('off')  # Turn off axis numbers and ticks
#         axes[1].set_title('Processed Image')
        
#         plt.tight_layout()
#         plt.show()
        
#         # Save the processed image
#         processed_img.save(output_path)  
        # return processed_img  


def remove_junctions(junctions, img, output_path,dot_size=8):
            # Get image size
            height, width = np.shape(img)
            # Create a new image for processing
            processed_img = copy.deepcopy(img)
            processed_img = Image.fromarray(processed_img)

            processed_img = processed_img.convert("L")   

            # Create a new image with transparent background
            green_dot_img = Image.new('RGBA', (width, height), (255,255,255,255))

            # assert(np.shape(green_dot_img)==np.shape(img))
            draw_green = ImageDraw.Draw(green_dot_img)
            
            # Draw green dots on transparent image
            for dot in junctions:
                x, y = dot
                draw_green.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill= (255, 255,255,255))
            
            # Merge processed_img and green_dot_img
            processed_img = Image.alpha_composite(processed_img.convert('RGBA'), green_dot_img)
            processed_img = processed_img.convert('L')

            # Display both images side by side
            fig, axes = plt.subplots(1, 2, figsize=(6, 6))
            
            # Plot original image
            axes[0].imshow(np.flipud(img), cmap='gray')
            axes[0].axis('off')  # Turn off axis numbers and ticks
            #axes[0].set_title('Original Image')
            
            # Plot processed image with green dots
            axes[1].imshow(np.flipud(processed_img), cmap = 'gray')
            axes[1].axis('off')  # Turn off axis numbers and ticks
            #axes[1].set_title(title)
            
            plt.tight_layout()
            plt.show()
            
            # Save the processed image with green dots
            processed_img.save(output_path)
            return processed_img    

def is_intersect(image, x, y, width, height, box_size=15, perc=.5):
    box_size = int(math.sqrt(box_size)-1)
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
                if image.getpixel((pixel_x, pixel_y)) > .9:
                    count += 1
    
    # Return True if the count of black pixels exceeds the threshold
    return count >= box_size*box_size*perc


def identify_connected_components(image):

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
    # fig, axes = plt.subplots(1, 2, figsize=(6, 6))
    
    # Display the original image on the left subplot
    # axes[0].imshow(np.flipud(image), cmap='gray')
    # #axes[0].set_title('Original Image')
    # axes[0].axis('off')
    
    # # Display the image with outlines on the right subplot
    # axes[1].imshow(np.flipud(image_rgb))
    # #axes[1].set_title('Connected Components Outlined')
    # axes[1].axis('off')
    
    # # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()
    
    # Return labels and stats
    return labels, stats, num_labels



def apply_sobel_filter_to_components(img, labels, stats, num_labels):
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

    # blur
    blur = cv2.GaussianBlur(img, (0,0), 1.3, 1.3)

    # Initialize Gx and Gy matrices to store results for the entire image
    Gx_total = np.zeros_like(blur, dtype=np.float32)
    Gy_total = np.zeros_like(blur, dtype=np.float32)
    
    #label_id_sort = sort_label_id(num_labels, stats)
    
    # Loop through each connected component
    for label_id in range(1,num_labels):
        # Extract the bounding box coordinates with added padding of 50 pixels
        left = max(stats[label_id, cv2.CC_STAT_LEFT] - 50, 0)
        top = max(stats[label_id, cv2.CC_STAT_TOP] - 50, 0)
        width = min(stats[label_id, cv2.CC_STAT_WIDTH] + 100, img.shape[1] - left)
        height = min(stats[label_id, cv2.CC_STAT_HEIGHT] + 100, img.shape[0] - top)


        # Create a mask for the current connected component
        mask = (labels == label_id).astype(np.uint8)
        
        # Apply Sobel filter to the masked component
        masked_image = blur * (~mask)
        
        # Apply Sobel filter to every pixel in the masked component
        sobelx = cv2.Sobel(masked_image,cv2.CV_64F,1,0,ksize=3)
        sobely = cv2.Sobel(masked_image,cv2.CV_64F,0,1,ksize=3)
        
        # Accumulate Gx and Gy values into the total matrices
        for x in range(top,top+height):
            for y in range(left,left+width):
                if(Gx_total[x,y]==0):
                    Gx_total[x,y] = sobelx[x,y]
                if(Gy_total[x,y]==0):
                    Gy_total[x,y] = sobely[x,y]

        # Gx_total[top:top+height, left:left+width] = sobelx[top:top+height, left:left+width]
        # Gy_total[top:top+height, left:left+width] = sobely[top:top+height, left:left+width]

    return (Gx_total), (Gy_total)

def sort_label_id(num_labels, stats, size):
    small_areas = []
    for label_id in range(1, num_labels):
        # Extract the bounding box coordinates
        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        
        # Compute the area
        area = width * height
        if area < size: 
        # Append the label ID and area to the list
            small_areas.append(label_id)

    return small_areas


 