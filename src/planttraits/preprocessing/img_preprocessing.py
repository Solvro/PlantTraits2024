import os

import torch

from planttraits.utils import DTYPE
from torchvision.io import read_image


class ImagePreprocessing:
    def __init__(
        self,
        imgs_folder,  # add more parameters if necessary
    ):
        self.imgs_paths = None
        self.imgs_folder = imgs_folder  # To będa wczytywane zawsze treningowe.
        # do not remove any image path from the list
        # self.imgs_paths = {int(p.split('.')[0]): p for p in os.listdir(self.imgs_folder)}
        # list of removed sample indexes from imgs_paths (don't delete it nor set to None if not used)
        self.drop_idxs = []
        self.prepare_data(self.imgs_folder)
        self._fit_preprocessing()

    def prepare_data(self, data):
        self.imgs_paths = {int(p.split('.')[0]): p for p in os.listdir(self.imgs_folder)}
        """
        Unifying and “cleaning” the data, which makes sense to do regardless of whether
         the data comes from a training or test collection.

        Deleting unnecessary columns or rows for both train and test.
        Unification data types ex. conversion to appropriate num-categorical types.
        Standardization of format ex. size of letters, deleting spaces.
        Other cleansing operations, which don't rely on "learning parameters", only on transforming
        raw data to concise format.
        """
        pass

    def _fit_preprocessing(
        self,
    ):  # Tylko treningowe, zastosowanie fit i scalera i wyliczanie wartosci tylko raz
        """
        Calculate parameters (e.g., mean, variance, coding maps, PCA matrix, etc.)
        that are needed for subsequent data transformation.

        Computation of statistics on cleaned data ex. avg, std_dv for scaler.
        Teaching coder for categorical variables. (we don't have any I think)
        Computation of reduction dimensionality parameters which are learning on data distribution.
        """
        pass

    def _transform_preprocessing(self, idx):  # Wspólny dla testowych i treningowych
        """
        The use of common transformations, as well as the use of previously learned parameters
         to process a single line (sample).

        Perform type conversion (e.g., convert values from Pandas Series to floating-point numbers).
        Normalize or scale the data using parameters learned earlier on the training set.
        Extract or calculate additional features from existing data.
        Apply possible augmentation operations (if working with images and want to introduce random
        transformations for the training data).
        """
        pass

    def transform(self, idx) -> torch.Tensor:
        self._transform_preprocessing(idx)  # Tutaj wedle uznania, w jaki sposób użycie tego
        img_path = self.imgs_folder / self.imgs_paths[idx]
        image = read_image(img_path)
        # quick rescaling for testing purposes
        image = image / 255.0

        # here perform some transformations on a single image
        # you can use functions from here: https://pytorch.org/vision/0.11/transforms.html

        return torch.tensor(image, dtype=DTYPE)
