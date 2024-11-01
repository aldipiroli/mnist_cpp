import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from python.model.model import SimpleCNN
from python.utils import utils


def open_img_opencv(img_path):
    image = cv2.imread(img_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, (28, 28), interpolation=cv2.INTER_AREA)
    image_array = image_resized.reshape((1, 28, 28, 1))
    return image_array


def open_img_pil(img_path):
    image = Image.open(img_path)
    # image = image.convert("L")
    # image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape((1, 28, 28, 3))
    return image


def shift_image(image, p=1):
    import random

    height, width, _ = image.shape

    shift_x = random.randint(-width // 2, width // 2)
    shift_y = random.randint(-height // 2, height // 2)

    shifted_image = np.zeros_like(image)

    new_x = np.clip(np.arange(width) + shift_x, 0, width - 1)
    new_y = np.clip(np.arange(height) + shift_y, 0, height - 1)

    shifted_image[new_y[:, None], new_x] = image
    return shifted_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    assert os.path.isdir(args.path)

    config = utils.load_yaml(os.path.join(args.path, "python/config/model.yaml"))
    model = SimpleCNN(config)
    latest = tf.train.latest_checkpoint(os.path.join(args.path, "assets/ckpts"))
    model.load_weights(latest)

    img_path = os.path.join(args.path, "assets/imgs/single.png")
    image = open_img_opencv(img_path)

    pred = model(image)
    print("PRED:", np.argmax(pred, -1), np.max(pred))


if __name__ == "__main__":
    main()
