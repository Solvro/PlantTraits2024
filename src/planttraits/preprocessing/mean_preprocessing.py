import numpy as np
import pandas as pd
import torch

from planttraits.utils import DTYPE, STD_COLUMN_NAMES, TARGET_COLUMN_NAMES


class MeanPreprocessing:
    def __init__(
        self,
        data,  # add more parameters if necessary
    ):
        self.csv_file = data
        # filter only necessary columns
        self.prepare_data(self.csv_file)._fit_preprocessing(self.csv_file).transform_preprocessing(self.csv_file)

    def prepare_data(self, data):
        """
        Unifying and “cleaning” the data, which makes sense to do regardless
         of whether the data comes from a training or test collection.

        Deleting unnecessary columns or rows for both train and test.
        Unification data types ex. conversion to appropriate num-categorical types.
        Standardization of format ex. size of letters, deleting spaces.
        Other cleansing operations, which don't rely on "learning parameters", only on transforming
        raw data to concise format.
        """
        return self

    def _fit_preprocessing(self, data):  # Tylko treningowe, zastosowanie fit i scalera i wyliczanie wartosci tylko raz
        """
        Calculate parameters (e.g., mean, variance, coding maps, PCA matrix, etc.)
        that are needed for subsequent data transformation.

        Computation of statistics on cleaned data ex. avg, std_dv for scaler.
        Teaching coder for categorical variables. (we don't have any I think)
        Computation of reduction dimensionality parameters which are learning on data distribution.
        """
        return self

    def transform_preprocessing(self, data):  # Wspólny dla testowych i treningowych
        """
        The use of common transformations, as well as the use of previously learned parameters
         to process a single line (sample).

        Perform type conversion (e.g., converrowt values from Pandas Series to floating-point numbers).
        Normalize or scale the data using parameters learned earlier on the training set.
        Extract or calculate additional features from existing data.
        Apply possible augmentation operations (if working with images and want to introduce
        random transformations for the training data).
        """
        pass

    def select(self, row: pd.Series) -> torch.Tensor:
        mean = torch.tensor(row[TARGET_COLUMN_NAMES].values, dtype=DTYPE)
        # even if you decide not to use std, still don't delete the line below
        std = torch.tensor(np.zeros_like(row[STD_COLUMN_NAMES].values), dtype=DTYPE)
        mean = torch.normal(mean, std)
        return mean, std

    def reverse_transform(self, preds: torch.Tensor) -> torch.Tensor:
        # transform means back to the original scale (reverse log operations etc.) but don't remove any rows
        return preds
