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

import torch.utils.data
from config import Config

from centroidnet import *

dev = "cuda:1"

def create_centroidnet(num_channels, num_classes):
    model = centroidnet.CentroidNet(num_classes, num_channels)
    return model


def load_model(filename, model):
    print(f"Load snapshot from: {os.path.abspath(filename)}")
    with open(filename, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    return model


def predict(image, model, max_dist, binning, nm_size, centroid_threshold, sub, div):
    # Prepare network input
    inputs = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0).astype(np.float32)
    inputs = torch.Tensor((inputs - sub) / div)

    # Upload to device
    inputs = inputs.to(Config.dev)
    model.to(Config.dev)

    # Do inference and decoding
    outputs = model(inputs)[0].cpu().detach().numpy()
    centroid_vectors, votes, class_ids, class_probs, votes_nm, centroids = centroidnet.decode(outputs, max_dist, binning, nm_size, centroid_threshold)

    # Only return the list of centroids
    return centroids


def main():
    file = "data/dataset/valid_0.png"

    print(f"Load image: {file}")
    image = cv2.imread(file)
    assert image is not None, f"Image {os.path.abspath(file)} not found"

    print(f"Predicting.")
    model = create_centroidnet(num_channels=Config.num_channels, num_classes=Config.num_classes)
    model = load_model(os.path.join("data", "CentroidNet.pth"), model)
    centroids = predict(image, model, Config.max_dist, Config.binning, Config.nm_size, Config.centroid_threshold, Config.sub, Config.div)
    centroids = np.stack(centroids, axis=0)
    print(f"Found {centroids.shape[0]} centroids (y, x, class_id, probability):\n {centroids}")


if __name__ == '__main__':
    main()
