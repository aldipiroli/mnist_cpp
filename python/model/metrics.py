import tensorflow as tf


def custom_accuracy(y_true, y_pred):
    y_pred_argmax = tf.cast(tf.math.argmax(y_pred, axis=1), tf.int32)
    y_true_argmax = tf.cast(tf.math.argmax(y_true, axis=1), tf.int32)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_argmax, y_true_argmax), tf.float32))
    return accuracy


class CustomEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data):
        super(CustomEvalCallback, self).__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        res = self.model.evaluate(self.val_data)
        print(f"Epoch: {epoch}, val_set accuracy: {res[1]:.4f}")
        # all_accuracies = []
        # for data, y_true in self.val_data:
        #     y_pred = self.model.predict(data)
        #     acc = custom_accuracy(y_true, y_pred)
        #     all_accuracies.append(acc)
        # print(
        #     f"\nEpoch {epoch + 1}: val_loss = {res[0]:.4f}, val_accuracy = {res[1]:.4f}, custom_accuracy = {tf.reduce_mean(all_accuracies) }"
        # )
