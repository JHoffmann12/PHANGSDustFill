import logging

import cv2
import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


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

    # Build lookup set of valid 8-bit pattern integers (same bit order as neighbours())
    powers = (1 << np.arange(7, -1, -1, dtype=np.uint16))
    valid_set = np.array(
        [(np.array(p, dtype=np.uint16) * powers).sum() for p in validIntersection],
        dtype=np.uint16,
    )

    image = (skeleton > 0).astype(np.uint8)

    # Find all interior skeleton pixels at once
    xs, ys = np.where(image[1:-1, 1:-1])
    xs += 1  # adjust for removed border
    ys += 1

    if len(xs) == 0:
        return []

    # Extract 8 clockwise neighbours for every skeleton pixel simultaneously.
    # Order matches neighbours(): above, above-right, right, below-right,
    #                             below, below-left, left, above-left.
    n = np.stack([
        image[xs - 1, ys    ],   # above
        image[xs - 1, ys + 1],   # above-right
        image[xs,     ys + 1],   # right
        image[xs + 1, ys + 1],   # below-right
        image[xs + 1, ys    ],   # below
        image[xs + 1, ys - 1],   # below-left
        image[xs,     ys - 1],   # left
        image[xs - 1, ys - 1],   # above-left
    ], axis=1).astype(np.uint16)  # (N, 8)

    pattern_ints = (n * powers[np.newaxis]).sum(axis=1)   # (N,)
    is_junc = np.isin(pattern_ints, valid_set)

    junc_xs = xs[is_junc]
    junc_ys = ys[is_junc]
    intersections = list(zip(junc_ys.tolist(), junc_xs.tolist()))  # (col, row) = (y, x)

    if len(intersections) <= 1:
        return intersections

    # Deduplicate: suppress any point within 10 px of an already-accepted one.
    pts = np.array(intersections)
    tree = cKDTree(pts)
    used = np.zeros(len(pts), dtype=bool)
    filtered = []
    for i in range(len(pts)):
        if not used[i]:
            filtered.append(intersections[i])
            for j in tree.query_ball_point(pts[i], r=10.0):
                used[j] = True
    return filtered


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


def removeAllJunctions(binary_mask, dot_size=1):
    """Remove junction pixels using two sequential passes.

    Stage 1 — matrix method: Uses the neighbourhood pattern lookup table in
    getSkeletonIntersection to identify T/Y/X branch points, then zeros a
    square patch of radius dot_size around each.

    Stage 2 — graph method: Re-skeletonizes the stage-1 result, finds every
    pixel with three or more skeleton neighbours (degree >= 3), clusters nearby
    branch points by centroid proximity, and erases a patch around each cluster
    medoid.  Clustering avoids punching multiple overlapping holes at dense
    branch regions.

    Parameters
    ----------
    - binary_mask (ndarray): 2D binary image (bool or uint8, values 0/1).
    - dot_size (int): Half-width in pixels of the patch erased around each junction.

    Returns
    -------
    - result (ndarray): Float copy of binary_mask with all junction regions zeroed.
    """
    binary = (binary_mask > 0).astype(np.uint8)

    # --- Stage 1: matrix / lookup-table method ---
    junctions = getSkeletonIntersection(binary * 255)
    result = removeJunctions(junctions, binary.astype(float), dot_size)

    # --- Stage 2: graph / degree-counting method ---
    skel = skeletonize(result > 0)
    if not np.any(skel):
        return result

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = ndimage.convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    degree_img = neighbor_count * skel.astype(np.uint8)

    raw_junctions = list(map(tuple, np.argwhere(degree_img >= 3)))
    if not raw_junctions:
        return result

    cluster_radius = max(4, dot_size * 4)
    raw_arr = np.array(raw_junctions)

    # Replace O(n^2) cluster-merging while-loop with single-linkage hierarchical clustering
    if len(raw_arr) == 1:
        labels = np.array([1])
    else:
        Z = linkage(raw_arr, method='single', metric='euclidean')
        labels = fcluster(Z, t=float(cluster_radius), criterion='distance')

    for lab in np.unique(labels):
        members = raw_arr[labels == lab]
        mean = members.mean(axis=0)
        medoid = members[np.argmin(np.linalg.norm(members - mean, axis=1))]
        r, c = int(round(medoid[0])), int(round(medoid[1]))
        r0 = max(0, r - dot_size);  r1 = min(result.shape[0], r + dot_size + 1)
        c0 = max(0, c - dot_size);  c1 = min(result.shape[1], c + dot_size + 1)
        result[r0:r1, c0:c1] = 0.0

    return result


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
