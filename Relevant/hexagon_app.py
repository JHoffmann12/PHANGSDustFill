import math
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import matplotlib.pyplot as plt

class HexagonApp:
    def __init__(self, root, image_path, data):
        self.root = root
        self.data = data
        self.image_path = image_path
        self.root.title("Hexagon Outlines")
        self.hexagon_size = 50
        
        # Load and resize the original image for display
        self.original_image = Image.open(self.image_path)
        self.resized_image = self.original_image.resize((476, 476))
        self.resized_image_tk = ImageTk.PhotoImage(self.resized_image)
        
        self.img_label = tk.Label(root)
        self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.img_label.configure(image=self.resized_image_tk)
        self.img_label.image = self.resized_image_tk

        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        increase_button = tk.Button(button_frame, text="+", command=self.on_increase_clicked)
        increase_button.pack(side=tk.LEFT, padx=5, pady=5)

        decrease_button = tk.Button(button_frame, text="-", command=self.on_decrease_clicked)
        decrease_button.pack(side=tk.LEFT, padx=5, pady=5)

        open_file_button = tk.Button(button_frame, text="Open Image", command=self.on_open_file)
        open_file_button.pack(side=tk.LEFT, padx=5, pady=5)

        analyze_button = tk.Button(button_frame, text="Analyze Hexagon", command=self.analyze_hexagon)
        analyze_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clicked_point = None
        self.clicked_hexagon = None

        # Calculate hexagons based on the original image size
        self.hexagons = self.calculate_hexagons()
        self.update_image()
        self.img_label.bind("<Button-1>", self.on_image_click)

    def calculate_hexagons(self):
        hexagons = []
        im = Image.open(self.image_path)
        w = math.sqrt(3) * self.hexagon_size
        h = 2 * self.hexagon_size
        num_hor = int(im.size[0] / w) + 2
        num_ver = int(im.size[1] / h * 4 / 3) + 2
        
        for row in range(num_ver):
            even = row % 2
            for column in range(num_hor):
                p = self.hexagon_corners((column * w + even * w / 2, row * h * 3 / 4))
                p_clipped = [(int(np.clip(point[0], 0, im.size[0] - 1)), int(np.clip(point[1], 0, im.size[1] - 1))) for point in p]
                hexagons.append(p_clipped)
        
        return hexagons

    def hexagon_corners(self, center):
        x, y = center
        w = math.sqrt(3) * self.hexagon_size
        h = 2 * self.hexagon_size
        return [
            (x - w / 2, y - h / 4),
            (x, y - h / 2),
            (x + w / 2, y - h / 4),
            (x + w / 2, y + h / 4),
            (x, y + h / 2),
            (x - w / 2, y + h / 4)
        ]

    def hexagonify_with_outline(self):
        im = Image.open(self.image_path)
        im_copy = im.copy()
        draw = ImageDraw.Draw(im_copy)
        w = math.sqrt(3) * self.hexagon_size
        h = 2 * self.hexagon_size
        num_hor = int(im.size[0] / w) + 2
        num_ver = int(im.size[1] / h * 4 / 3) + 2
        
        for row in range(num_ver):
            even = row % 2
            for column in range(num_hor):
                p = self.hexagon_corners((column * w + even * w / 2, row * h * 3 / 4))
                p_clipped = [(int(np.clip(point[0], 0, im.size[0] - 1)), int(np.clip(point[1], 0, im.size[1] - 1))) for point in p]
                draw.polygon(p_clipped, outline="purple")
        
        return im_copy

    def update_image(self):
        modified_image = self.hexagonify_with_outline()
        if self.clicked_hexagon:
            overlay = Image.new('RGBA', modified_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.polygon(self.clicked_hexagon, fill=(128, 0, 128, 128))  # Transparent purple
            modified_image = Image.alpha_composite(modified_image.convert('RGBA'), overlay)
        
        # Resize modified image to display in the GUI
        resized_modified_image = modified_image.resize((476, 476))
        self.resized_image_tk = ImageTk.PhotoImage(resized_modified_image)
        self.img_label.configure(image=self.resized_image_tk)
        self.img_label.image = self.resized_image_tk

    def on_increase_clicked(self):
        self.hexagon_size += 1
        self.update_image()

    def on_decrease_clicked(self):
        self.hexagon_size = max(1, self.hexagon_size - 1)
        self.update_image()

    def on_open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(self.image_path)
            self.hexagons = self.calculate_hexagons()
            self.update_image()

    def on_image_click(self, event):
        original_x = event.x * (self.original_image.width / self.resized_image.width)
        original_y = event.y * (self.original_image.height / self.resized_image.height)
        
        self.clicked_point = (original_x, original_y)
        self.clicked_hexagon = None
        for hexagon in self.hexagons:
            if self.point_in_polygon(self.clicked_point, hexagon):
                self.clicked_hexagon = hexagon
                break
        self.update_image()

    def point_in_polygon(self, point, polygon):
        x, y = point
        num_vertices = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(num_vertices + 1):
            p2x, p2y = polygon[i % num_vertices]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def analyze_hexagon(self):
        plt.clf()
        hex_data = []
        if self.clicked_hexagon is None:
            messagebox.showinfo("Error", "Please click on an image to select a hexagon first.")
            return
        
        img = cv2.imread(self.image_path)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(self.clicked_hexagon)], 1)

        data = self.data
        num_col = len(self.data[0])
        num_pixels = np.size(self.data)
        for r in range(num_pixels//num_col):
            for c in range(num_col):
                point = (r,c)
                if(self.point_in_polygon(point, self.clicked_hexagon)):
                    hex_data.append(self.data[r][c])

        hex_data = np.array(hex_data)
        hex_data = hex_data[~np.isnan(hex_data)]
        if(len(hex_data == 0)):
            print('No data in Hexagon')
        else: 
            plt.hist(hex_data, bins=30, edgecolor='black')
            plt.title('Histogram of Pixel Orientations')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

if __name__ == "__main__":
    # Replace with your image path and data
    image_path = "path_to_your_image.jpg"
    data = np.random.randn(1042, 1038)  # Example data matching original image size
    root = tk.Tk()
    app = HexagonApp(root, image_path=image_path, data=data)
    root.mainloop()
