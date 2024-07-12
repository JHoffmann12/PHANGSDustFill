import importlib
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import ipywidgets as widgets
from IPython.display import display
import math
from PIL import Image, ImageDraw, ImageTk, ImageOps
import tkinter as tk
from tkinter import ttk

import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import tkinter as tk
from tkinter import ttk

# Helper functions
def hexagon_corners(center, size):
    x, y = center
    w = math.sqrt(3) * size
    h = 2 * size

    return [
        (x - w / 2, y - h / 4),
        (x, y - h / 2),
        (x + w / 2, y - h / 4),
        (x + w / 2, y + h / 4),
        (x, y + h / 2),
        (x - w / 2, y + h / 4)
    ]

def hexagonify_with_outline(image, hexagon_size):
    im_copy = image.copy()
    draw = ImageDraw.Draw(im_copy)

    w = math.sqrt(3) * hexagon_size
    h = 2 * hexagon_size

    num_hor = int(im_copy.size[0] / w) + 2
    num_ver = int(im_copy.size[1] / h * 4 / 3) + 2

    hexagons = []

    for i in range(num_hor * num_ver):
        column = i % num_hor
        row = i // num_hor
        even = row % 2

        center = (column * w + even * w / 2, row * h * 3 / 4)
        p = hexagon_corners(center, hexagon_size)
        p_clipped = [(int(np.clip(point[0], 0, im_copy.size[0] - 1)), int(np.clip(point[1], 0, im_copy.size[1] - 1))) for point in p]

        draw.polygon(p_clipped, outline="purple")
        hexagons.append((center, p_clipped))

    return im_copy, hexagons

def point_in_polygon(point, polygon):
    num_vertices = len(polygon)
    x, y = point
    inside = False

    p1x, p1y = polygon[0]
    for i in range(num_vertices + 1):
        p2x, p2y = polygon[i % num_vertices]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersection:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def update_image(hexagon_size, original_image):
    modified_image, hexagons = hexagonify_with_outline(original_image, hexagon_size)
    im_display = np.array(modified_image)
    height, width = im_display.shape[:2]
    scale = min(800 / width, 800 / height)
    im_display_resized = cv2.resize(im_display, (int(width * scale), int(height * scale)))
    cv2.imshow('image', im_display_resized)
    return hexagons, im_display

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hexagons, im_display, data_array = param['hexagons'], param['im_display'], param['data_array']
        height, width = im_display.shape[:2]
        scale = min(800 / width, 800 / height)
        x = int(x / scale)
        y = int(y / scale)
        param['hex_data'].clear()

        # Reset image to original before highlighting the new hexagon
        im_display = param['im_display_base'].copy()
        for center, hexagon in hexagons:
            if point_in_polygon((x, y), hexagon):
                mask = Image.new("RGBA", (im_display.shape[1], im_display.shape[0]), (0, 0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.polygon(hexagon, fill=(128, 0, 128, 128))
                im_overlay = Image.alpha_composite(Image.fromarray(im_display), mask)
                im_display = np.array(im_overlay)
                im_display = cv2.cvtColor(im_display, cv2.COLOR_RGBA2BGRA)  # Ensure BGRA format after overlay
                height, width = im_display.shape[:2]
                scale = min(800 / width, 800 / height)
                im_display_resized = cv2.resize(im_display, (int(width * scale), int(height * scale)))
                cv2.imshow('image', im_display_resized)

                # Append corresponding data to hex_data
                min_x = int(min([px for px, _ in hexagon]))
                max_x = int(max([px for px, _ in hexagon]))
                min_y = int(min([py for _, py in hexagon]))
                max_y = int(max([py for _, py in hexagon]))

                for px in range(min_x, max_x + 1):
                    for py in range(min_y, max_y + 1):
                        if point_in_polygon((px, py), hexagon):
                            if 0 <= py < 1042 and 0 <= px < 1038:
                                param['hex_data'].append(data_array[py][px])

                param['im_display'] = im_display
                param['hex_center'] = center  # Store the center of the hexagon
                break

def analyze_hexagon(hex_data, center):
    hex_data = [x for x in hex_data if not math.isnan(x)]  # Remove NaN values
    if len(hex_data) == 0: 
        print('No data in hexagon!')
    else:
        plt.hist(hex_data, bins=30, alpha=0.75)
        center = tuple(int(x) for x in center)
        plt.title(f'Histogram of Hexagon Data centered at {center}')
        plt.xlabel('Pixel Orientation')
        plt.ylabel('Frequency')
        plt.show()
        print(f'Number of Pixels {len(hex_data)}')

def increase_hexagon_size(param):
    param['hexagon_size'] += 1
    param['hexagons'], param['im_display_base'] = update_image(param['hexagon_size'], param['original_image'])

def decrease_hexagon_size(param):
    param['hexagon_size'] = max(1, param['hexagon_size'] - 1)
    param['hexagons'], param['im_display_base'] = update_image(param['hexagon_size'], param['original_image'])

