from planttraits.config import TRAIN_CSV_FILE, TEST_CSV_FILE, TRAIN_IMAGES_FOLDER, TEST_IMAGES_FOLDER
import pandas as pd 
import torch
import os 

class StdPreprocessing:

    def __init__(self, train : bool, # add more parameters if necessary
                 ):
        self.csv_file = pd.read_csv(TRAIN_CSV_FILE if train else TEST_CSV_FILE)

        # filter only necessary columns
         
        self.preprocess()

    def preprocess(self):
        # inner computations on the whole set in necessary
        pass 

    def transform(self, idx) -> torch.Tensor:
        row = self.csv_file.iloc[idx]

        # some transformations on a single row of data if necessary

        return torch.tensor(row.values)