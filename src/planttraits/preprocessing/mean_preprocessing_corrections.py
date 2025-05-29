import numpy as np
import pandas as pd
import torch
# nw czy ci ten import zadziala, jak cos to tutaj sobie te zmienne wklej do skryptu
from planttraits.utils import DTYPE, STD_COLUMN_NAMES, TARGET_COLUMN_NAMES


class MeanPreprocessing:
    def __init__(
        self,
        data,  # add more parameters if necessary
    ):
        self.csv_file = data
        
        # filter only necessary columns
        self.prepare_data(self.csv_file)._fit_preprocessing(self.csv_file).transform_preprocessing(self.csv_file)

    def prepare_data(self, data=None):

        # Drop rows where X4_mean < 0 or > 3
        to_drop = data[(data['X4_mean'] < 0) | (data['X4_mean'] > 3)].index
        data.drop(to_drop, inplace=True)

        # Log transform selected columns (excluding X4_mean)
        self.columns_to_log = [col for col in TARGET_COLUMN_NAMES if col != 'X4_mean']
        data[self.columns_to_log] = np.log1p(data[self.columns_to_log])

        # Fill missing values in STD columns
        data[STD_COLUMN_NAMES] = data[STD_COLUMN_NAMES].fillna(data[STD_COLUMN_NAMES].mean())

        return self

    def _fit_preprocessing(self, data): 
        return self

    def transform_preprocessing(self, data):  
        pass

    def select(self, row: pd.Series) -> torch.Tensor:
        
        mean = torch.tensor(row[TARGET_COLUMN_NAMES].values, dtype=DTYPE)
        # even if you decide not to use std, still don't delete the line below
        std = torch.tensor(np.zeros_like(row[STD_COLUMN_NAMES].values), dtype=DTYPE)
        mean = torch.normal(mean, std)
        return mean, std

    def reverse_transform(self, preds: torch.Tensor) -> torch.Tensor:
        
        # Move tensor to CPU to convert it to a NumPy array
        preds_np = preds.cpu().numpy()
    
        # Convert to DataFrame
        df_preds = pd.DataFrame(preds_np, columns=TARGET_COLUMN_NAMES)

        # Apply inverse of log1p to the columns that were log-transformed previously
        df_preds[self.columns_to_log] = np.expm1(df_preds[self.columns_to_log])

        # Convert DataFrame back to a torch tensor
        return torch.tensor(df_preds.values, dtype=DTYPE)
