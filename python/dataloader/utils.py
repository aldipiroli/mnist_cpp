import os

import numpy as np
import tensorflow as tf


def download_mnist_dataset(folder_path):
    """
    Downloads the MNIST dataset and saves it to the specified folder.

    Parameters:
    folder_path (str): The path to the folder where the dataset will be saved.
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Download the MNIST dataset to the default location
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print(x_train.shape)
    print(y_train.shape)

    # Save the dataset as numpy arrays in the specified folder
    np.save(os.path.join(folder_path, "x_train.npy"), x_train)
    np.save(os.path.join(folder_path, "y_train.npy"), y_train)
    np.save(os.path.join(folder_path, "x_test.npy"), x_test)
    np.save(os.path.join(folder_path, "y_test.npy"), y_test)

    print(f"MNIST dataset downloaded and saved to {folder_path}")


download_mnist_dataset("data/mnist/")
