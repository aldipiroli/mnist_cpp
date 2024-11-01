import tensorflow as tf
from tensorflow.keras import layers, models


class SimpleCNN(tf.keras.Model):
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.config = config["MODEL"]
        self.model = models.Sequential()
        input_size = self.config["INPUT_SIZES"]

        # First convolutional layer
        self.model.add(
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                input_shape=(input_size[0], input_size[1], input_size[2]),
            )
        )
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Second convolutional layer
        self.model.add(
            layers.Conv2D(
                64,
                (3, 3),
                activation="relu",
            )
        )
        self.model.add(layers.MaxPooling2D((2, 2)))

        # Classification
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(layers.Dense(10, activation="softmax"))

    def call(self, data):
        return self.model(data)


if __name__ == "__main__":
    config = {"INPUT_SIZES": [28, 28, 1]}
    data = tf.random.uniform((2, 28, 28, 1))
    model = SimpleCNN(config)
