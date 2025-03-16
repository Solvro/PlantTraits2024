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
        self._fit_transform_preprocessing(self.csv_file)

    def _fit_transform_preprocessing(
        self, data
    ):  # Tylko treningowe, zastosowanie fit i scalera i wyliczanie wartosci tylko raz
        """
        Calculate parameters (e.g., mean, variance, coding maps, PCA matrix, etc.)
        that are needed for subsequent data transformation.

        Computation of statistics on cleaned data ex. avg, std_dv for scaler.
        Teaching coder for categorical variables. (we don't have any I think)
        Computation of reduction dimensionality parameters which are learning on data distribution.
        """
        return self

    def select(self, row: pd.Series) -> torch.Tensor:
        mean = torch.tensor(row[TARGET_COLUMN_NAMES].values, dtype=DTYPE)
        # even if you decide not to use std, still don't delete the line below
        std = torch.tensor(np.zeros_like(row[STD_COLUMN_NAMES].values), dtype=DTYPE)
        mean = torch.normal(mean, std)
        return mean, std

    def reverse_transform(self, preds: torch.Tensor) -> torch.Tensor:
        # transform means back to the original scale (reverse log operations etc.) but don't remove any rows
        return preds
