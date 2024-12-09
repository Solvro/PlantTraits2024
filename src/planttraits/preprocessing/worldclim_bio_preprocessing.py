from planttraits.config import TRAIN_CSV_FILE, TEST_CSV_FILE, TRAIN_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from planttraits.utils import DTYPE
import pandas as pd 
import torch
import os 

class WorldClimBioPreprocessing:

    def __init__(self, is_train : bool, # add more parameters if necessary
                 ):
        self.csv_file = pd.read_csv(TRAIN_CSV_FILE if is_train else TEST_CSV_FILE)
        self.is_train = is_train

        # filter only necessary columns
         
        self.preprocess()

    def preprocess(self):
        # inner computations on the whole set in necessary
        pass 

    def transform(self, idx) -> torch.Tensor:
        row = self.csv_file.iloc[idx]

        # some transformations on a single row of data if necessary

        return torch.tensor(row.values, dtype=DTYPE)