# class for preprocessing the dataset
import os
import pathlib
from PIL import Image
import numpy as np
from . import parse_configuration
import splitfolders


class prepr_dat():
    def __init__(self):
        self.original_dataset_dir = pathlib.Path("./dataset")
        self.destination_dir = pathlib.Path("./preprocessed_dataset")
        self.config = parse_configuration.parse()

    def split_dataset(self):
        # ratio of split are in order of train/val/test. You can change to whatever you want. For train/val sets only, you could do .75, .25 for example.
        splitfolders.ratio(self.original_dataset_dir,
                           output=self.destination_dir, seed=42, ratio=(.7, 0, .3))
