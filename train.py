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
import torch.optim as optim
from tqdm import tqdm
import copy
from config import Config
from centroidnet import *


def create_data_loader(training_file, validation_file, crop, max_dist, repeat, sub, div):
    training_set = centroidnet.CentroidNetDataset(training_file, crop=crop, max_dist=max_dist, repeat=repeat, sub=sub, div=div)
    validation_set = centroidnet.CentroidNetDataset(validation_file, crop=crop, max_dist=max_dist, sub=sub, div=div)
    return training_set, validation_set


def create_centroidnet(num_channels, num_classes):
    model = centroidnet.CentroidNet(num_classes, num_channels)
    return model


def create_centroidnet_loss():
    loss = centroidnet.CentroidLoss()
    return loss


def validate(validation_loss, epoch, validation_set_loader, model, loss, validation_interval=10):
    if epoch % validation_interval == 0:
        with torch.no_grad():
            # Validate using validation data loader
            model.eval() # put in evaluation mode
            validation_loss = 0
            idx = 0
            for inputs, targets in validation_set_loader:
                inputs = inputs.to(Config.dev)
                targets = targets.to(Config.dev)
                outputs = model(inputs)
                mse = loss(outputs, targets)
                validation_loss += mse.item()
                idx += 1
            model.train() # put back in training mode
            return validation_loss / idx
    else:
        return validation_loss


def save_model(filename, model):
    print(f"Save snapshot to: {os.path.abspath(filename)}")
    with open(filename, "wb") as f:
        torch.save(model.state_dict(), f)


def load_model(filename, model):
    print(f"Load snapshot from: {os.path.abspath(filename)}")
    with open(filename, "rb") as f:
        state_dict = torch.load(f)
    model.load_state_dict(state_dict)
    return model


def train(training_set, validation_set, model, loss, epochs, batch_size, learn_rate, validation_interval):
    print(f"Training {len(training_set)} images for {epochs} epochs with a batch size of {batch_size}.\n"
          f"Validate {len(validation_set)} images each {validation_interval} epochs and learning rate {learn_rate}.\n")

    #training_set.eval()

    best_model = copy.deepcopy(model)
    model.to(Config.dev)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    training_set_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)
    validation_set_loader = torch.utils.data.DataLoader(validation_set, batch_size=min(len(validation_set), batch_size), shuffle=True, num_workers=10, drop_last=True)

    if len(training_set_loader) == 0:
        raise Exception("The training dataset does no contain any samples. Is the minibatch larger than the amount of samples?")
    if len(training_set_loader) == 0:
        raise Exception("The validation dataset does no contain any samples. Is the minibatch larger than the amount of samples?")

    bar = tqdm(range(1, epochs))
    validation_loss = 9999
    best_loss = 9999
    for epoch in bar:
        training_loss = 0
        idx = 0
        # Train one minibatch
        for (inputs, targets) in training_set_loader:
            inputs = inputs.to(Config.dev)
            targets = targets.to(Config.dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            ls = loss(outputs, targets)
            ls.backward()
            optimizer.step()
            training_loss += ls.item()
            idx += 1

        scheduler.step(epoch)

        # Update progress bar
        bar.set_description("Epoch {}/{} Loss(T): {:5f} and Loss(V): {:.5f}".format(epoch, epochs, training_loss / idx, validation_loss))
        bar.refresh()

        # Validate and save
        validation_loss = validate(validation_loss, epoch, validation_set_loader, model, loss, validation_interval)
        if validation_loss < best_loss:
            print(f"Update model with loss {validation_loss}")
            best_loss = validation_loss
            best_model.load_state_dict(model.state_dict())

    return best_model


def predict(data_set, model, loss, max_dist, binning, nm_size, centroid_threshold):
    print(f"Predicting {len(data_set)} files with loss {type(loss)}")
    with torch.no_grad():
        data_set.eval()
        model.eval()
        model.to(Config.dev)
        loss_value = 0
        idx = 0
        set_loader = torch.utils.data.DataLoader(data_set, batch_size=5, shuffle=False, num_workers=1, drop_last=False)
        result_images = []
        result_centroids = []
        for inputs, targets in tqdm(set_loader):
            inputs = inputs.to(Config.dev)
            targets = targets.to(Config.dev)
            outputs = model(inputs)
            ls = loss(outputs, targets)
            loss_value += ls.item()
            decoded = [centroidnet.decode(img, max_dist, binning, nm_size, centroid_threshold) for img in outputs.cpu().numpy()]

            # Add all numpy arrays to a list
            result_images.extend([{"inputs": i.cpu().numpy(),
                                   "targets": t.cpu().numpy(),
                                   "vectors": d[0],
                                   "votes": d[1],
                                   "class_ids": d[2],
                                   "class_probs": d[3],
                                   "centroids": d[4]} for i, t, o, d in zip(inputs, targets, outputs, decoded)])

            # Add image_id to centroid locations and add to list
            result_centroids.extend([np.stack(ctr for ctr in d[5]) for d in decoded])
            idx = idx + 1
    print("Aggregated loss is {:.5f}".format(loss_value / idx))
    return result_images, result_centroids


def output(folder, result_images, result_centroids):
    os.makedirs(folder, exist_ok=True)
    print(f"Created output folder {os.path.abspath(folder)}")
    for i, sample in enumerate(result_images):
        for name, arr in sample.items():
            np.save(os.path.join(folder, f"{i}_{name}.npy"), arr)

    lines = ["image_nr centroid_y centroid_x class_id probability \r\n"]
    with open(os.path.join(folder, "validation.txt"), "w") as f:
        for i, image in enumerate(result_centroids):
            for line in image:
                line_str = [str(i), *[str(elm) for elm in line]]
                lines.append(" ".join(line_str) + "\r\n")
        f.writelines(lines)


def main():
    # Perform retraining.
    do_train = True
    # Perform loading of the model.
    do_load = True
    # Perform final prediction and export data.
    do_predict = True

    # Start script
    assert (do_train or do_load), "Enable do_train and/or do_load"

    # Load datasets
    training_set, validation_set = create_data_loader(os.path.join("data", "dataset", "training.csv"),
                                                      os.path.join("data", "dataset", "validation.csv"),
                                                      crop=Config.crop, max_dist=Config.max_dist, repeat=1, sub=Config.sub, div=Config.div)

    assert training_set.num_classes == Config.num_classes, f"Number of classes on config.py is incorrect. Should be {training_set.num_classes}"

    # Create loss function
    loss = create_centroidnet_loss()
    model = None

    # Train network
    if do_train:
        # Create network and load snapshots
        model = create_centroidnet(num_channels=Config.num_channels, num_classes=Config.num_classes)
        model = train(training_set, validation_set, model, loss, epochs=Config.epochs, batch_size=Config.batch_size, learn_rate=Config.learn_rate, validation_interval=Config.validation_interval)
        save_model(os.path.join("data", "CentroidNet.pth"), model)

    # Load model
    if do_load:
        model = create_centroidnet(num_channels=3, num_classes=training_set.num_classes)
        model = load_model(os.path.join("data", "CentroidNet.pth"), model)

    # Predict
    if do_predict:
        result_images, result_centroids = predict(validation_set, model, loss, max_dist=Config.max_dist, binning=Config.binning, nm_size=Config.nm_size, centroid_threshold=Config.centroid_threshold)
        output(os.path.join("data", "validation_result"), result_images, result_centroids)


if __name__ == '__main__':
    main()
