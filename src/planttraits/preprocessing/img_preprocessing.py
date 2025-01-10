import os

import torch

from planttraits.config import TEST_IMAGES_FOLDER, TRAIN_IMAGES_FOLDER
from planttraits.utils import DTYPE
from torchvision.io import read_image


class ImagePreprocessing:
    def __init__(
        self,
        is_train: bool,  # add more parameters if necessary
    ):
        self.imgs_folder = TRAIN_IMAGES_FOLDER if is_train else TEST_IMAGES_FOLDER
        self.imgs_paths = os.listdir(self.imgs_folder)
        self.is_train = is_train
        # list of removed sample indexes from imgs_paths (don't delete it nor set to None if not used)
        self.drop_idxs = []

        self.preprocess()

    def preprocess(self):
        # inner computations like computing means itd if necessary
        # add indexes of removed samples to the list self.drop_idxs
        # store final version under self.imgs_paths
        pass

    def transform(self, idx) -> torch.Tensor:
        img_path = self.imgs_folder / self.imgs_paths[idx]
        image = read_image(img_path)

        # here perform some transformations on a single image
        # you can use functions from here: https://pytorch.org/vision/0.11/transforms.html

        return torch.tensor(image, dtype=DTYPE)
