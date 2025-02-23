import numpy as np
import pandas as pd
import torch

from planttraits.config import TEST_CSV_FILE, TRAIN_CSV_FILE
from planttraits.utils import DTYPE, STD_COLUMN_NAMES, TARGET_COLUMN_NAMES


class MeanPreprocessing:
    def __init__(
        self,
        is_train: bool,  # add more parameters if necessary
    ):
        self.csv_file = pd.read_csv(TRAIN_CSV_FILE if is_train else TEST_CSV_FILE, index_col='id')
        self.is_train = is_train
        # list of removed sample indexes (don't delete it nor set to None if not used)
        self.drop_idxs = []
        # filter only necessary columns
        self.df_mean = self.csv_file[TARGET_COLUMN_NAMES].copy()
        self.df_sd = self.csv_file[STD_COLUMN_NAMES].copy()

        self.preprocess()

    def preprocess(self):
        # inner computations on the whole set in necessary
        # add indexes of removed samples to the list self.drop_idxs

        # Check for missing values
        self.df_mean.isnull().sum()
        self.df_sd.isnull().sum()

        # Remove invalid X4_mean values 
        to_drop = self.df_mean[(self.df_mean['X4_mean'] < 0) | (self.df_mean['X4_mean'] > 3)].index
        self.drop_idxs = to_drop.tolist()

        # Drop invalid rows
        self.df_mean.drop(to_drop, inplace=True)
        self.df_sd.drop(to_drop, inplace=True)
        
        # Fill missing values in sd columns
        self.df_sd.fillna(self.df_sd.mean(), inplace=True)
        
        # Log transform selected columns (excluding X4_mean)
        columns_to_transform = [col for col in TARGET_COLUMN_NAMES if col != 'X4_mean']
        self.df_mean[columns_to_transform] = np.log1p(self.df_mean[columns_to_transform])

        pass

    def transform(self, row: pd.Series) -> torch.Tensor:
        mean = torch.tensor(row[TARGET_COLUMN_NAMES].values, dtype=DTYPE)

        # some transformations on a single row of data if necessary
        mean = torch.tensor(row[TARGET_COLUMN_NAMES].values, dtype=DTYPE)
        # even if you decide not to use std, still don't delete the line below
        std = torch.tensor(np.zeros_like(row[STD_COLUMN_NAMES].values), dtype=DTYPE)
        mean = torch.normal(mean, std)
        return mean, std
