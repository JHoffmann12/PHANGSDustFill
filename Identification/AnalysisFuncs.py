import logging

import cv2
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def neighbours(x, y, image):
    """Return the 8 clockwise neighbours of pixel (x, y) in image.

    Parameters
    ----------
    - x (int): Row index of the target pixel.
    - y (int): Column index of the target pixel.
    - image (ndarray): 2D image array.

    Returns
    -------
    - neighbours (list): 8 pixel values in clockwise order starting from (x-1, y).
    """
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [
        image[x_1][y], image[x_1][y1], image[x][y1], image[x1][y1],
        image[x1][y], image[x1][y_1], image[x][y_1], image[x_1][y_1],
    ]


def getSkeletonIntersection(skeleton):
    """Return pixel coordinates of all junction points in a binary skeleton image.

    Parameters
    ----------
    - skeleton (ndarray): Skeletonised binary image scaled to [0, 255].

    Returns
    -------
    - filtered_intersections (list): List of (x, y) tuples identifying junction pixels,
      deduplicated so no two returned points are within 10 pixels of each other.
    """
    validIntersection = [
        [0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
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
        [1,0,1,1,0,1,0,0],
    ]

    image = skeleton.copy() / 255
    intersections = []
    for x in range(1, len(image) - 1):
        for y in range(1, len(image[x]) - 1):
            if image[x][y] == 1:
                if neighbours(x, y, image) in validIntersection:
                    intersections.append((y, x))

    # Deduplicate: suppress any point within 10 pixels of an already-accepted one.
    # O(n²) over intersection count, which is typically small (tens to low hundreds).
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
    """Zero out a square patch of radius dot_size around each junction pixel.

    Parameters
    ----------
    - junctions (list): List of (x, y) junction coordinates from getSkeletonIntersection.
    - img (ndarray): Binary image to remove junctions from.
    - dot_size (int): Half-width in pixels of the patch erased around each junction.

    Returns
    -------
    - result (ndarray): Copy of img with junction patches set to zero.
    """
    mask = np.ones_like(img, dtype=bool)
    for x, y in junctions:
        x, y = int(x), int(y)
        x0 = max(0, x - dot_size)
        y0 = max(0, y - dot_size)
        x1 = min(mask.shape[1], x + dot_size)
        y1 = min(mask.shape[0], y + dot_size)
        mask[y0:y1, x0:x1] = False
    return img * mask


def identify_connected_components(image):
    """Label connected components and draw bounding rectangles on an RGB copy.

    Parameters
    ----------
    - image (ndarray): Grayscale uint8 binary image.

    Returns
    -------
    - labels (ndarray): Label map from cv2.connectedComponentsWithStats.
    - stats (ndarray): Per-component statistics array.
    - num_labels (int): Total number of components including background.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    color_orange = (255, 165, 0, 150)
    for label in range(1, num_labels):
        left   = stats[label, cv2.CC_STAT_LEFT]
        top    = stats[label, cv2.CC_STAT_TOP]
        width  = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        cv2.rectangle(image_rgb, (left, top), (left + width, top + height), color_orange, 2)
    return labels, stats, num_labels


def sort_label_id(num_labels, stats, size):
    """Return label IDs whose bounding-box area is smaller than size.

    Parameters
    ----------
    - num_labels (int): Total number of components including background.
    - stats (ndarray): Per-component statistics from cv2.connectedComponentsWithStats.
    - size (int): Area threshold in pixels; components strictly below this are returned.

    Returns
    -------
    - small_areas (list): List of label IDs whose bounding-box area is < size.
    """
    small_areas = []
    for label_id in range(1, num_labels):
        width  = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        if width * height < size:
            small_areas.append(label_id)
    return small_areas
