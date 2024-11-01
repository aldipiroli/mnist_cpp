import tensorflow as tf


def classification_loss(y_pred, y_gt):
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    loss = loss_fn(y_pred, y_gt)
    return loss


def indices_to_one_hot(y, N):
    y_one_hot = tf.one_hot(y, N, axis=-1)
    # y_one_hot = tf.squeeze(y_one_hot, axis=1)
    return y_one_hot


if __name__ == "__main__":
    y_pred = tf.random.uniform((2, 10))
    y_gt = tf.random.uniform(shape=(2, 1), minval=0, maxval=9, dtype=tf.int32)
    y_gt_one_hot = indices_to_one_hot(y_gt, 10)
    print("y_pred", y_pred.shape)
    print("y_gt", y_gt.shape)
    print("y_gt_one_hot", y_gt_one_hot.shape)
    loss = classification_loss(y_pred, y_gt_one_hot)
    print("loss", loss)
