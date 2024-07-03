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
            thresholded_img.show()

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

