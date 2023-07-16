import logging
import pathlib
import tensorflow as tf
import sys
from utils import parse_configuration
import matplotlib.pyplot as plt
import numpy as np


class load_data():
    def __init__(self) -> None:
        self.config = parse_configuration.parse()
        self.datadir = pathlib.Path("./preprocessed_dataset/train")
        self.test_datadir = pathlib.Path("./preprocessed_dataset/test")
        logging.info("DATASET PATH: {}".format(self.datadir))
        print("DATASET PATH: {}".format(self.datadir))
        image_count = len(list(self.datadir.glob('*/*.jpg')))
        logging.info("FOUND {} IMAGES".format(image_count))
        print(f"FOUND {image_count} IMAGES")

        self.VALIDATION_SPLIT = self.config.getfloat(
            "TRAINING", "VALIDATION_SPLIT")
        self.SEED = self.config.getint("TRAINING", "SEED")
        self.IMG_HEIGHT = self.config.getint("DATASET", "IMAGE_SIZE")
        self.IMG_WIDTH = self.config.getint("DATASET", "IMAGE_SIZE")
        self.BATCH_SIZE = self.config.getint("TRAINING", "BATCH_SIZE")

    def prepare(self, ds, shuffle=False, augment=False):
        AUTOTUNE = tf.data.AUTOTUNE
        # use buffered prefetching to load images from disk without having I/O become blocking
        resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(self.IMG_HEIGHT, self.IMG_WIDTH),
            tf.keras.layers.Rescaling(1./255)
        ])
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])
        # Resize and rescale all datasets.
        ds = ds.map(lambda x, y: (resize_and_rescale(x), y))

        if shuffle:
            ds = ds.shuffle(1000)

        # Use data augmentation only on the training set.
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        # Configure the dataset for efficent memory usage
        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)

    def create_train_dataloader(self):
        # training
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.datadir,
            validation_split=self.VALIDATION_SPLIT,
            subset="training",
            seed=self.SEED,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE)
        logging.info("TRAIN DATASET CREATED")
        print("TRAIN DATASET CREATED")
        return train_ds

    def create_validation_dataloader(self):
        # validation
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.datadir,
            validation_split=self.VALIDATION_SPLIT,
            subset="validation",
            seed=self.SEED,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE)
        logging.info("VALIDATION DATASET CREATED")
        print("VALIDATION DATASET CREATED")
        return val_ds

    def create_test_dataloader(self):
        # test
        test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_datadir,
            seed=self.SEED,
            image_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.BATCH_SIZE)
        logging.info("TEST DATASET CREATED")
        print("TEST DATASET CREATED")
        test_ds = self.prepare(test_ds)
        return test_ds

    def create_training_chart(self, num_epochs, history, save_path):
        # plot the training loss and accuracy
        N = len(history.history["loss"])
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N),
                 history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N),
                 history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N),
                 history.history["val_accuracy"], label="val_acc")
        plt.plot(np.arange(0, N),
                 history.history["recall"], label="train_recall")
        plt.plot(np.arange(0, N),
                 history.history["val_recall"], label="val_recall")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(save_path)
