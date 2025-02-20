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

        self.preprocess()

    def preprocess(self):
        # inner computations on the whole set in necessary
        # add indexes of removed samples to the list self.drop_idxs
        pass

    def transform(self, row: pd.Series) -> torch.Tensor:
        mean = torch.tensor(row[TARGET_COLUMN_NAMES].values, dtype=DTYPE)

        # some transformations on a single row of data if necessary

        # even if you decide not to use std, still don't delete the line below
        std = torch.tensor(np.zeros_like(row[STD_COLUMN_NAMES].values), dtype=DTYPE)
        mean = torch.normal(mean, std)
        return mean, std
