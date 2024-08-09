import numpy as np
import json
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from config import *

class ImageDataset(Dataset):
    def __init__(self, dataframe, encoder, transform, pred = False) -> None:
        super(ImageDataset).__init__()
        self.dataframe = dataframe
        self.encoder = encoder
        self.transforms = transform
        self.pred = pred

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        image = self.vector_to_image(self.dataframe.iloc[index]['drawing'])
        if self.pred:
          return self.transforms(image).to(device), self.dataframe.iloc[index]['key_id']
        label = self.encoder.transform([self.dataframe.iloc[index]['word']])
        return self.transforms(image).to(device), torch.from_numpy(label).to(device)

    def vector_to_numpy(self, drawing, side=256):
        image = self.vector_to_image(drawing, side)
        image_array = np.array(image)
        return image_array

    def vector_to_image(self, drawing, side=256):
        drawing = json.loads(drawing)
        min_x, min_y, max_x, max_y = self.calculate_bounding_box(drawing)

        # Calculate the offset to center the drawing within the canvas
        offset_x = (side - (max_x - min_x + 1)) // 2
        offset_y = (side - (max_y - min_y + 1)) // 2

        image = Image.new('RGB', (side, side), color='white')  # Create a white canvas
        m = int(255 / len(drawing))
        draw = ImageDraw.Draw(image)
        for i, (x, y) in enumerate(drawing):
            xy = [(x0 - min_x + offset_x, y0 - min_y + offset_y) for x0, y0 in zip(x, y)]
            draw.line(xy, fill=(255 - i*m, i*m,255 - i*m), width=1)

        return image

    def calculate_bounding_box(self, drawing):
        all_x = [x for x, _ in drawing]
        all_y = [y for _, y in drawing]

        min_x = min(min(x) for x in all_x)
        min_y = min(min(y) for y in all_y)
        max_x = max(max(x) for x in all_x)
        max_y = max(max(y) for y in all_y)

        return min_x, min_y, max_x, max_y
    