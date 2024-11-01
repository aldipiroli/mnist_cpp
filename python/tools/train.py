import argparse
import os
import shutil

import tensorflow as tf

from python.dataloader.dataloader import Dataloader
from python.model.losses import classification_loss
from python.model.metrics import CustomEvalCallback, custom_accuracy
from python.model.model import SimpleCNN
from python.utils import utils


class Trainer:
    def __init__(self, config, model, train_dataloader, val_dataloader, log_path):
        super().__init__()
        self.config = config
        self.log_path = log_path
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tb_callback = utils.tensorboard_callback(os.path.join(log_path, "tb_logs"))
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.log_path, "ckpts", "checkpoint_epoch_{epoch:02d}.ckpt"),
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        )

    def train(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=classification_loss,
            metrics=["accuracy", custom_accuracy],
        )
        custom_val_eval = CustomEvalCallback(self.val_dataloader)
        self.model.fit(
            self.train_dataloader,
            epochs=20,
            callbacks=[
                self.tb_callback,
                self.checkpoint_callback,
                custom_val_eval,
            ],
            validation_data=self.val_dataloader,
        )
        self.model.save(os.path.join(self.log_path, "ckpts"))

    def visual_evaluation(self, max_num_samples=20):
        eval_save_path = os.path.join(self.log_path, "val_img")
        next_batch = next(iter(self.val_dataloader))
        data, y_gt = next_batch
        y_pred = self.model(data)
        y_pred = utils.get_argmax_pred(y_pred)
        y_gt = utils.get_argmax_pred(y_gt)

        for idx, (curr_data, curr_y_gt, curr_y_pred) in enumerate(zip(data, y_gt, y_pred)):
            title = f"y_pred: {str(curr_y_pred.numpy())}, y_gt: {str(curr_y_gt.numpy())}"
            curr_data = curr_data[..., 0].numpy()
            utils.save_image_with_text(img=curr_data, text=title, save_path=eval_save_path, file_name=str(idx))
            if idx > max_num_samples:
                break

    def save_checkpoint(self):
        self.checkpoint.save(os.path.join(self.log_path, "last"))
        tf.saved_model.save(self.model, self.log_path)


def create_log_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)  # Removes the folder
    os.makedirs(folder_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()
    assert os.path.isdir(args.path)

    config = utils.load_yaml(os.path.join(args.path, "python/config/model.yaml"))
    log_path = os.path.join(args.path, "logs")
    create_log_folder(log_path)

    model = SimpleCNN(config)
    dataloader = Dataloader(config=config, root_path=args.path)
    train_dataloader, val_dataloader = dataloader.get_dataloaders()
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        log_path=log_path,
    )
    trainer.train()
    trainer.save_checkpoint()
    trainer.visual_evaluation()


if __name__ == "__main__":
    main()
