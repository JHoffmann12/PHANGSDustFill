import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import ttk
import csv

def display_image_with_hexagons(image_path, hexagon_size):
    original_image = Image.open(image_path).convert("RGBA")
    hexagon_image, hexagons = hexagonify_with_outline(original_image, hexagon_size)

    return hexagons, hexagon_image 

def get_hexagon_mask(image_shape, corners):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Create a single channel mask
    cv2.fillPoly(mask, [np.array(corners, dtype=np.int32)], 255)
    return mask

def find_row_by_tuple(input_tuple, csv_path):
    input_value = f"({input_tuple[0]}, {input_tuple[1]})"
    first_column_values = []
    rows = []
    headers = []

    with open(csv_path, newline='') as o:
        myData = csv.reader(o)
        headers = next(myData)
        for row in myData:
            first_column_values.append(row[0])
            rows.append(row)

    try:
        for index, value in enumerate(first_column_values):
            if value == input_value:
                matching_row = rows[index]
                for header, data in zip(headers, matching_row):
                    print(f"{header}: {data}")
                return
        print("Input value not found in the first column.")
    except ValueError:
        print("Error occurred while processing the CSV file.")

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
    return hexagons, im_display_resized

def update_display_image(param):
    im_display_resized = cv2.resize(np.array(param['im_display_with_borders']), (800, 800))
    tk_image = ImageTk.PhotoImage(Image.fromarray(im_display_resized))
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.image = tk_image

def click_event(event, x, y, param):
    hexagons, im_display, data_array = param['hexagons'], param['im_display'], param['data_array']
    height, width = im_display.shape[:2]
    scale = min(800 / width, 800 / height)
    x = int(x / scale)
    y = int(y / scale)
    param['hex_data'].clear()

    im_display_base = Image.fromarray(param['im_display_base']).convert("RGBA")
    mask = Image.new("RGBA", im_display_base.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)

    for center, hexagon in hexagons:
        if point_in_polygon((x, y), hexagon):
            mask_draw.polygon(hexagon, fill=(128, 0, 128, 128))
            param['hex_center'] = center

            min_x = int(min([px for px, _ in hexagon]))
            max_x = int(max([px for px, _ in hexagon]))
            min_y = int(min([py for _, py in hexagon]))
            max_y = int(max([py for _, py in hexagon]))

            for px in range(min_x, max_x + 1):
                for py in range(min_y, max_y + 1):
                    if point_in_polygon((px, py), hexagon):
                        if 0 <= py < data_array.shape[0] and 0 <= px < data_array.shape[1]:
                            param['hex_data'].append(data_array[py][px])

            im_overlay = Image.alpha_composite(im_display_base, mask)
            border_overlay = Image.new("RGBA", im_display_base.size, (0, 0, 0, 0))
            border_draw = ImageDraw.Draw(border_overlay)
            for _, hexagon in hexagons:
                border_draw.polygon(hexagon, outline="purple")
            
            im_display_with_border = Image.alpha_composite(im_overlay, border_overlay)
            param['im_display'] = np.array(im_display_with_border)
            param['im_display_with_borders'] = im_display_with_border

            update_display_image(param)
            break

def analyze_hexagon(param, csv_path):
    if param['hex_data'] and param['hex_center']:
        hex_data = [x for x in param['hex_data'] if not math.isnan(x)]
        if len(hex_data) == 0:
            print('No data in hexagon!')
        else:
            # plt.hist(hex_data, bins=30, alpha=0.75)
            # center = tuple(int(x) for x in param['hex_center'])
            # plt.title(f'Histogram of Hexagon Data centered at {center}')
            # plt.xlabel('Pixel Orientation')
            # plt.ylabel('Frequency')
            # plt.show()
            find_row_by_tuple(param['hex_center'], csv_path)

def run_hexagon_analysis(original_image_path, csv_path, data_array, hexagon_size=100):
    original_image = Image.open(original_image_path).convert("RGBA")
    hexagons, im_display_resized = update_image(hexagon_size, original_image)

    param = {
        'hexagon_size': hexagon_size,
        'original_image': original_image,
        'hexagons': hexagons,
        'im_display': np.array(original_image),
        'im_display_base': np.array(original_image),
        'im_display_with_borders': np.array(original_image),
        'im_display_resized': im_display_resized,
        'data_array': data_array,
        'hex_data': [],
        'hex_center': None,
        'csv_data': csv_path
    }

    global canvas
    root = tk.Tk()
    root.title("Hexagon Analysis")

    canvas = tk.Canvas(root, width=800, height=800)
    canvas.pack()

    tk_image = ImageTk.PhotoImage(Image.fromarray(im_display_resized))
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

    def on_click(event):
        click_event(event, event.x, event.y, param)

    canvas.bind("<Button-1>", on_click)

    analyze_button = ttk.Button(root, text="Extract Hexagon Info", command=lambda: analyze_hexagon(param, csv_path))
    analyze_button.pack(side=tk.BOTTOM, pady=10)

    def on_closing():
        cv2.destroyAllWindows()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    original_image_path = r'C:\Users\HP\Documents\JHU_Academics\Research\PHANGS\smoothed_angles.png'
    csv_path = r"C:\Users\HP\Documents\JHU_Academics\Research\PHANGS\PHANGSDustFill\thinskeletondata.csv"

    data_array = np.random.uniform(0, 100, (1042, 1038))
    run_hexagon_analysis(original_image_path, csv_path, data_array)
