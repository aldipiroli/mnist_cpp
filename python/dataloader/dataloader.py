import os

import numpy as np
import tensorflow as tf

from python.model.losses import indices_to_one_hot


class Dataloader:
    def __init__(self, config, root_path):
        super().__init__()
        self.root_path = root_path
        self.config = config["DATA"]
        self.config["PATH"] = os.path.join(self.root_path, self.config["PATH"])

    def load_data(self):
        x_train = np.load(os.path.join(self.config["PATH"], "x_train.npy")).reshape(-1, 28, 28, 1)
        x_test = np.load(os.path.join(self.config["PATH"], "x_test.npy")).reshape(-1, 28, 28, 1)

        y_train = np.load(os.path.join(self.config["PATH"], "y_train.npy"))
        y_train = indices_to_one_hot(y_train, N=10)

        y_test = np.load(os.path.join(self.config["PATH"], "y_test.npy"))
        y_test = indices_to_one_hot(y_test, N=10)
        return x_train, y_train, x_test, y_test

    def get_dataloaders(self):
        x_train, y_train, x_test, y_test = self.load_data()

        print(x_train.shape, y_train.shape)
        train_dataloader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataloader = self.prepare_dataset(train_dataloader, augment=True)

        test_dataloader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataloader = self.prepare_dataset(test_dataloader, augment=True)
        return train_dataloader, test_dataloader

    def shift_image(self, image):
        # Get the size of the image
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]

        # Generate a random shift
        shift_x = tf.random.uniform([], minval=-width, maxval=width, dtype=tf.int32)
        shift_y = tf.random.uniform([], minval=-height, maxval=height, dtype=tf.int32)

        # Roll the image by the random shift
        shifted_image = tf.roll(image, shift=[shift_y, shift_x, 0], axis=[0, 1, 2])
        return shifted_image

    def apply_augmentation(self, image, label):
        image = self.shift_image(image)
        return image, label

    def prepare_dataset(self, dataset, augment=False):
        if augment:
            dataset = dataset.map(self.apply_augmentation)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self.config["BATCH_SIZE"])
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
