import os

import torch

from torchvision import transforms
from planttraits.utils import DTYPE
from torchvision.io import read_image


class ImagePreprocessing:
    def __init__(
        self,
        imgs_paths,  # add more parameters if necessary
    ):
        # zdjęcia treningowe
        self.imgs_paths = imgs_paths
        # To będa wczytywane zawsze treningowe.
        self._fit_preprocessing()

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


        image = image / 255.0

        transform_list = []

        transform_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]

        transform_list.append(transforms.ConvertImageDtype(DTYPE))  # Ensure correct dtype

        transform_pipeline = transforms.Compose(transform_list)

        image = transform_pipeline(image)
        pass

    def _transform_preprocessing(self, image):  # Wspólny dla testowych i treningowych
        # quick rescaling for testing purposes
        image = image / 255.0
        """
        The use of common transformations, as well as the use of previously learned parameters
         to process a single line (sample).

        Perform type conversion (e.g., convert values from Pandas Series to floating-point numbers).
        Normalize or scale the data using parameters learned earlier on the training set.
        Extract or calculate additional features from existing data.
        Apply possible augmentation operations (if working with images and want to introduce random
        transformations for the training data).
        """


        return torch.tensor(image, dtype=DTYPE) 

    def transform(self, img_path) -> torch.Tensor:
        image = read_image(img_path)
        image = self._transform_preprocessing(image)  
        return image