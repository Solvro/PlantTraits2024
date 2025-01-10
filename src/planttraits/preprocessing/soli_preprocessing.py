import pandas as pd
import torch

from planttraits.config import TEST_CSV_FILE, TRAIN_CSV_FILE
from planttraits.utils import DTYPE


class SoilPreprocessing:
    def __init__(
        self,
        is_train: bool,  # add more parameters if necessary
    ):
        self.csv_file = pd.read_csv(TRAIN_CSV_FILE if is_train else TEST_CSV_FILE)
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
        # row = ... # filter on columns

        # some transformations on a single row of data if necessary

        return torch.tensor(row.values, dtype=DTYPE)
