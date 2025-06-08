import os
import numpy as np
from pathlib import Path
from keras.datasets import cifar10
from keras.utils import to_categorical
from cifar10_resnet.config import CONFIG


def preprocess_func():
    """
    Description: Preprocesses the CIFAR-10 dataset by loading the data, normalizing the labels, splitting it into training and validation sets, and converting the labels to one-hot vectors.
    Returns:
    train_X (np.ndarray): Numpy array of shape (num_train_samples, image_height, image_width, num_channels) containing the training data.
    val_X (np.ndarray): Numpy array of shape (num_val_samples, image_height, image_width, num_channels) containing the validation data.
    train_Y (np.ndarray): Numpy array of shape (num_train_samples, num_classes) containing the one-hot encoded training labels.
    val_Y (np.ndarray): Numpy array of shape (num_val_samples, num_classes) containing the one-hot encoded validation labels.
    """
    (data_X, data_Y), (test_X, test_Y) = cifar10.load_data()

    data = np.load(Path(os.path.join(CONFIG["RUN_PATH"],'dataset_split.npz')).expanduser())
    train_idxs = data["train_labeled_idxs"]
    unlabeled_idxs = data["train_unlabeled_idxs"]

    mean = np.array([0.4914, 0.4822, 0.4465])[None, None, None, :]
    std = np.array([0.2470, 0.2435, 0.2616])[None, None, None, :]

    data_X = data_X / 255
    data_X = (data_X - mean) / std
    test_X = test_X / 255
    test_X = (test_X - mean) / std

    data_Y, test_Y = np.squeeze(data_Y), np.squeeze(test_Y)
    data_Y, test_Y = to_categorical(data_Y), to_categorical(test_Y)  # Hot Vector

    return data_X, data_Y, test_X, test_Y, train_idxs, unlabeled_idxs
