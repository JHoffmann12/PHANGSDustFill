import math
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog

class HexagonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hexagon Outlines")
        self.hexagon_size = 50
        self.original_image_path = r'C:\Users\HP\Documents\JHU_Academics\Research\PHANGS\smoothed_angles.png'
        
        self.img_label = tk.Label(root)
        self.img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        increase_button = tk.Button(button_frame, text="+", command=self.on_increase_clicked)
        increase_button.pack(side=tk.LEFT, padx=5, pady=5)

        decrease_button = tk.Button(button_frame, text="-", command=self.on_decrease_clicked)
        decrease_button.pack(side=tk.LEFT, padx=5, pady=5)

        open_file_button = tk.Button(button_frame, text="Open Image", command=self.on_open_file)
        open_file_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.update_image()

    def hexagon_corners(self, center, size):
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

    def hexagonify_with_outline(self, original_image_path, hexagon_size):
        im = Image.open(original_image_path)
        im_copy = im.copy()
        draw = ImageDraw.Draw(im_copy)

        w = math.sqrt(3) * hexagon_size
        h = 2 * hexagon_size

        num_hor = int(im.size[0] / w) + 2
        num_ver = int(im.size[1] / h * 4 / 3) + 2

        for row in range(num_ver):
            even = row % 2
            for column in range(num_hor):
                p = self.hexagon_corners((column * w + even * w / 2, row * h * 3 / 4), hexagon_size)
                p_clipped = [(int(np.clip(point[0], 0, im.size[0] - 1)), int(np.clip(point[1], 0, im.size[1] - 1))) for point in p]
                draw.polygon(p_clipped, outline="purple")

        return im_copy

    def update_image(self):
        modified_image = self.hexagonify_with_outline(self.original_image_path, self.hexagon_size)
        modified_image_tk = ImageTk.PhotoImage(modified_image)

        self.img_label.configure(image=modified_image_tk)
        self.img_label.image = modified_image_tk

    def on_increase_clicked(self):
        self.hexagon_size += 1
        self.update_image()

    def on_decrease_clicked(self):
        self.hexagon_size = max(1, self.hexagon_size - 1)
        self.update_image()

    def on_open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image_path = file_path
            self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = HexagonApp(root)
    root.mainloop()
