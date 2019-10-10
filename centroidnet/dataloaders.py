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

import os
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
import centroidnet

class CentroidNetDataset(Dataset):
    """
    CentroidNetDataset Dataset
        Load centroids from txt file and apply vector aware data augmentation

    Arguments:
        filename: filename of the input data format: (image_file_name,xmin,ymin,xmax,ymax,class_id\lf)
        crop (h, w): Random crop size
        transpose ((dim2, dim3)): List of random transposes to choose from
        stride ((dim2, dim3)): List of random strides to choose from
    """
    def convert_path(self, path):
        if os.path.isabs(path):
            return path
        else:
            return os.path.join(self.data_path, path)

    def load_and_convert_data(self, filename, max_dist=None):
        with open(filename) as f:
            lines = f.readlines()
        lines = [x.strip().split(",") for x in lines]
        self.count = len(lines)
        img_ctrs = {}
        for line in lines:
            fn, xmin, xmax, ymin, ymax, id = line
            xmin, xmax, ymin, ymax, id = int(xmin), int(xmax), int(ymin), int(ymax), int(id)
            x, y = (xmin + xmax) // 2, (ymin + ymax) // 2
            if not fn in img_ctrs:
                img_ctrs[fn] = list()

            self.num_classes = id+2 if id+2 > self.num_classes else self.num_classes #including background class
            img_ctrs[fn].append(np.array([y, x, ymin, ymax, xmin, xmax, id], dtype=int))

        for key in img_ctrs.keys():
            img_ctrs[key] = np.stack(img_ctrs[key])
            fn = self.convert_path(key)
            img = cv2.imread(fn)

            if img is None:
                raise Exception("Could not read {}".format(fn))

            crop = min(img.shape[0], img.shape[1], self.crop[0], self.crop[1])
            if crop != self.crop[0]:
                print(f"Warning: random crop adjusted to {[crop, crop]}")
                self.set_crop([crop, crop])

            target = centroidnet.encode(img_ctrs[key], img.shape[0], img.shape[1], max_dist, self.num_classes)
            img = (np.transpose(img, [2, 0, 1]).astype(np.float32) - self.sub) / self.div
            target = target.astype(np.float32)

            self.images.append(img)
            self.targets.append(target)


    def __init__(self, filename: str, crop=(256, 256), max_dist=100, repeat=1, sub=127, div=256, transpose=np.array([[0, 1], [1, 0]]), stride=np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]]), data_path=None):
        self.count = 0
        if data_path is None:
            self.data_path = os.path.dirname(filename)
        else:
            self.data_path = data_path

        self.filename = filename
        self.images = list()
        self.targets = list()

        self.sub = sub
        self.div = div
        self.repeat = repeat
        self.set_crop(crop)
        self.set_repeat(repeat)
        self.set_transpose(transpose)
        self.set_stride(stride)
        self.num_classes = 0
        self.train()
        self.load_and_convert_data(filename, max_dist=max_dist)

    def eval(self):
        self.train_mode = False

    def train(self):
        self.train_mode = True

    def set_repeat(self, repeat):
        if repeat < 0:
            self.repeat = 1
        self.repeat = repeat

    def set_crop(self, crop):
        self.crop = crop

    def set_transpose(self, transpose):
        if np.all(transpose == np.array([[0, 1]])):
            self.transpose = None
        else:
            self.transpose = transpose

    def set_stride(self, stride):
        if np.all(stride == np.array([[1, 1]])):
            self.stride = None
        else:
            self.stride = stride

    def adjust_vectors(self, img, transpose, stride):
        if not transpose is None:
            img2 = img.copy()
            img[0] = img2[transpose[0]]
            img[1] = img2[transpose[1]]
        if not stride is None:
            img2 = img.copy()
            img[0] = img2[0] * stride[0]
            img[1] = img2[1] * stride[1]
        return img

    def adjust_image(self, img, transpose, slice, crop, stride):
        if not transpose is None:
            img = np.transpose(img, (0, transpose[0] + 1, transpose[1] + 1))
        if not slice is None:
            img = img[:, slice[0]:slice[0] + crop[0], slice[1]:slice[1] + crop[1]]
        if not stride is None:
            img = img[:, ::stride[0], ::stride[1]]
        return img

    def get_target(self, img: np.array, transpose, slice, crop, stride):
        img[0:2] = self.adjust_vectors(img[0:2], transpose, stride)
        img = self.adjust_image(img, transpose, slice, crop, stride)
        return img

    def get_input(self, img: np.array, transpose, slice, crop, stride):
        img = self.adjust_image(img, transpose, slice, crop, stride)
        return img

    def __getitem__(self, index):
        index = index // self.repeat
        input, target = self.images[index], self.targets[index]

        if self.stride is None and self.transpose is None and self.crop is None or not self.train_mode:
            return input.astype(np.float32), target.astype(np.float32)

        if not self.transpose is None:
            transpose = random.choice(self.transpose)
        else:
            transpose = None

        if not self.stride is None:
            stride = random.choice(self.stride)
        else:
            stride = None

        if not self.crop is None:
            min = np.array([0, 0])
            if not transpose is None:
                max = np.array([input.shape[transpose[0] + 1], input.shape[transpose[1] + 1]], dtype=int) - self.crop
            else:
                max = np.array([input.shape[1] - self.crop[0], input.shape[2] - self.crop[1]])
            slice = [random.randint(mn, mx) for mn, mx in zip(min, max)]
        else:
            slice = None

        input = self.get_input(input, transpose, slice, self.crop, stride).astype(np.float32)
        target = self.get_target(target, transpose, slice, self.crop, stride).astype(np.float32)

        return input, target

    def __len__(self):
        return len(self.images) * self.repeat
