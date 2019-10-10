# Copyright (C) 2019 Klaas Dijkstra
#
# This file is part of OpenCentroidNet.
#
# OpenCentroidNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# OpenCentroidNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with OpenCentroidNet.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from skimage.draw import circle
import os
import cv2
import random

def create_dataset(folder, file: str = "training.csv", n: int = 50, height: int = 200, width: int = 300, n_circles: int = 5, min_r: int = 15, max_r: int = 20, prefix="train", n_classes=3):
    boxes = []
    for i in range(n):
        out_fn = f"{prefix}_{i}.png"
        img = np.random.randint(10, 30, (3, height, width))
        for a in range(n_circles):
            if min_r < max_r:
                r = np.random.randint(min_r, max_r)
            else:
                r = min_r
            y = np.random.randint(r, height-r)
            x = np.random.randint(r, width-r)
            rr, cc = circle(y, x, r)
            d = (r - ((((rr - y) ** 2) + ((cc - x) ** 2)) ** 0.5)) * (150 / r) + 50
            id = a % min(n_classes, 3)
            img[id, rr, cc] = d
            xmin,xmax,ymin,ymax = x-r,x+r,y-r,y+r
            boxes.append(f"{out_fn},{xmin},{xmax},{ymin},{ymax},{id}\n")
        n_points = round(height * width * 0.2)
        y_rand = np.random.randint(0, height, n_points)
        x_rand = np.random.randint(0, width, n_points)
        img[:, y_rand, x_rand] = np.random.randint(0, 64, size=(3, n_points))
        img = np.transpose(img.astype(np.uint8), (1, 2, 0))
        cv2.imwrite(os.path.join(folder, out_fn), img)
    f = open(os.path.join(folder, file), "w")
    f.writelines(boxes)


random.seed(42)
folder = os.path.join("..", "data", "dataset")
train_folder = os.path.join(folder, "training")
validation_folder = os.path.join(folder, "validation")
os.makedirs(folder, exist_ok=True)
create_dataset(folder, file="training.csv", prefix="train", n=50)
create_dataset(folder, file="validation.csv", prefix="valid", n=10)


