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


def neighbours(x, y, image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]

def getSkeletonIntersection(skeleton):
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.
    
    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of
    
    Returns: 
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]]
    image = skeleton.copy() / 255
    intersections = []
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            # If we have a white pixel
            if image[x][y] == 1:
                neighbors = neighbours(x, y, image)
                if neighbors in validIntersection:
                    intersections.append((y, x))
    
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    filtered_intersections = []
    for point1 in intersections:
        add_point = True
        for point2 in filtered_intersections:
            if ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2:
                add_point = False
                break
        if add_point:
            filtered_intersections.append(point1)
    
    return filtered_intersections


def removeJunctions(junctions, img, dot_size):
    # Convert NumPy array to PIL Image (grayscale mode)
    # Make a mask of the junctions to be removed
    mask = np.ones_like(img, dtype=bool)
    for x, y in junctions:
        x0 = max(0, x - dot_size)
        y0 = max(0, y - dot_size)
        x1 = min(mask.shape[1], x + dot_size)
        y1 = min(mask.shape[0], y + dot_size)
        mask[y0:y1, x0:x1] = False  # Mask out region
    
    # Convert back to NumPy array for returning
    return img * mask

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
    
    return labels, stats, num_labels

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
