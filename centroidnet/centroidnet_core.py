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

import torch
import torch.nn
import torch.autograd
import torch.nn.modules.loss
import torch.nn.functional as F
from centroidnet.backbones import UNet
import numpy as np
from typing import List
from skimage.draw import ellipse
from skimage.feature import peak_local_max

class CentroidNet(torch.nn.Module):
    def __init__(self, num_classes, num_channels):
        torch.nn.Module.__init__(self)
        self.backbone = UNet(num_classes=num_classes+2, in_channels=num_channels, depth=5, start_filts=64)
        self.num_classes = num_classes
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def __str__(self):
        return f"CentroidNet: {self.backbone}"


class CentroidLoss(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.loss = 0

    def forward(self, result, target):
        loss = F.mse_loss(result, target, size_average=True, reduce=True)
        self.loss = loss.item()
        return loss

    def __str__(self):
        return f"{self.loss}"

def encode(coords, image_height: int, image_width: int, max_dist: int, num_classes: int):
    y_coords, x_coords, _, _, _, _, _  = np.transpose(coords)

    #Encode vectors
    target_vectors = calc_vector_distance(y_coords, x_coords, image_height, image_width, max_dist)
    if not max_dist is None:
        target_vectors /= max_dist
    target_vectors = np.transpose(target_vectors, [2, 0, 1])

    #Encode logits (bounding box is drawn as ellipses)
    target_logits = np.zeros((num_classes, image_height, image_width))
    target_logits[0] = 1
    for (y, x, ymin, ymax, xmin, xmax, id) in coords:
        ymin, ymax, xmin, xmax = min(ymin, ymax), max(ymin, ymax), min(xmin, xmax), max(xmin, xmax)
        rr, cc = ellipse(y, x, (ymax - ymin) // 2, (xmax - xmin) // 2, shape=(image_height, image_width))
        target_logits[id + 1][rr, cc] = 1
        target_logits[0][rr, cc] = 0

    target = np.concatenate((target_vectors, target_logits))
    return target


def decode(input : np.ndarray, max_dist: int, binning: int, nm_size: int, centroid_threshold: int):
    _, image_height, image_width = input.shape
    centroid_vectors = input[0:2] * max_dist
    logits = input[2:]

    #Calculate class ids and class probabilities
    class_ids = np.expand_dims(np.argmax(logits, axis=0), axis=0)
    sum_logits = np.expand_dims(np.sum(logits, axis=0), axis=0)
    class_probs = np.expand_dims(np.max((logits / sum_logits), axis=0), axis=0)
    class_probs = np.clip(class_probs, 0, 1)

    # Calculate the centroid images
    votes = calc_vote_image(centroid_vectors, binning)
    votes_nm = peak_local_max(votes[0], min_distance=nm_size, threshold_abs=centroid_threshold, indices=False)
    votes_nm = np.expand_dims(votes_nm, axis=0)

    # Calculate list of centroid statistics
    coords = np.transpose(np.where(votes_nm[0] > 0))
    centroids = [[y * binning, x * binning, class_ids[0, y * binning, x * binning] - 1, class_probs[0, y * binning, x * binning]] for (y, x) in coords]
    return centroid_vectors, votes, class_ids, class_probs, votes_nm, centroids


def calc_vector_distance(y_coords: List[int], x_coords: List[int], image_height: int, image_width: int, max_dist) -> np.ndarray:
    assert (len(y_coords) == len(x_coords)), "list of coordinates should be the same"
    assert (len(y_coords) > 0), "No centroids in source image"

    # Prepare datastructures
    shape = [image_height, image_width]
    image_coords = np.indices(shape)
    image_coords_planar = np.transpose(image_coords, [1, 2, 0])

    dist_cube = np.empty([len(y_coords), image_height, image_width])
    vec_cube = np.empty([len(y_coords), image_height, image_width, 2])

    # Create multichannel image with distances and vectors
    for (i, (y, x)) in enumerate(zip(y_coords, x_coords)):
        vec = np.array([y, x]) - image_coords_planar
        vec_cube[i] = vec
        dist = vec ** 2
        dist = np.sum(dist, axis=2)
        dist = np.sqrt(dist)
        dist_cube[i] = dist

    # Get the smallest centroid distance index
    dist_ctr_labels = np.argmin(dist_cube, axis=0)

    # Get the smallest distance vectors [h, w, yx]
    vec_ctr = vec_cube[dist_ctr_labels, image_coords[0], image_coords[1]]

    # Clip vectors
    if not max_dist is None:
        active = np.sqrt(np.sum(vec_ctr ** 2, axis=2)) > max_dist
        vec_ctr[active, :] = 0

    return vec_ctr


def calc_vote_image(centroid_vectors: np.array, f) -> np.ndarray:
    channels, height, width = centroid_vectors.shape

    size = np.array(np.array((height, width), dtype=np.float) * ((1/f), (1/f)), dtype=np.int)
    indices = np.indices((height, width), dtype=centroid_vectors.dtype)

    # Calculate absolute vectors
    vectors = ((centroid_vectors + indices) * (1/f)).astype(np.int)
    nimage = np.zeros((size[0], size[1]))

    # Clip pixels
    logic = np.logical_and(np.logical_and(vectors[0] >= 0, vectors[1] >= 0), np.logical_and(vectors[0] < size[0], vectors[1] < size[1]))
    coords = vectors[:, logic]

    # Accumulate
    np.add.at(nimage, (coords[0], coords[1]), 1)
    return np.expand_dims(nimage, axis=0)
