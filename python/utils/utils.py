import os

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml


def load_yaml(path):
    assert os.path.isfile(path)
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def tensorboard_callback(log_path):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
    return tensorboard_callback


def get_argmax_pred(y_pred):
    y_pred_argmax = tf.cast(tf.math.argmax(y_pred, axis=1), tf.int32)
    return y_pred_argmax


def save_image_with_text(img, text, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(text)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, file_name + ".png"), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
