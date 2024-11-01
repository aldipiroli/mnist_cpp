import tensorflow as tf

from python.model.losses import classification_loss, indices_to_one_hot


def test_classification_loss():
    y_pred = tf.random.uniform((2, 10))
    y_gt = tf.random.uniform(shape=(2, 1), minval=0, maxval=9, dtype=tf.int32)
    y_gt_one_hot = indices_to_one_hot(y_gt, 10)
    loss = classification_loss(y_pred, y_gt_one_hot)
    assert loss > 0
