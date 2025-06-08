import pandas as pd
import torch

from planttraits.utils import DTYPE
from sklearn.preprocessing import RobustScaler


class WorldClimBioPreprocessing:
    def __init__(
        self,
        data,
    ):
        self.selected_columns = [col for col in data.columns if col.startswith('WORLDCLIM_BIO')]
        self.data = data[self.selected_columns]
        self.scaler = None
        self.prepare_data(self.data)._fit_preprocessing(self.data).transform_preprocessing(self.data)

    def prepare_data(self, test_data: pd.DataFrame = None):
        """
        Unifying and “cleaning” the data, which makes sense to do regardless of whether
         the data comes from a training or test collection.

        Deleting unnecessary columns or rows for both train and test.
        Unification data types ex. conversion to appropriate num-categorical types.
        Standardization of format ex. size of letters, deleting spaces.
        Other cleansing operations, which don't rely on "learning parameters", only on transforming
        raw data to concise format.
        """
        if test_data is None:
            df = self.data
        else:
            df = test_data

        df.drop(df.columns[5], axis=1, inplace=True)
        return self

    def _fit_preprocessing(self, data):  # Tylko treningowe, zastosowanie fit i scalera i wyliczanie wartosci tylko raz
        """
        Calculate parameters (e.g., mean, variance, coding maps, PCA matrix, etc.)
         that are needed for subsequent data transformation.

        Computation of statistics on cleaned data ex. avg, std_dv for scaler.
        Teaching coder for categorical variables. (we don't have any I think)
        Computation of reduction dimensionality parameters which are learning on data distribution.
        """
        self.scaler = RobustScaler()
        self.scaler.fit(data)
        return self

    def transform_preprocessing(self, test_data: pd.DataFrame = None):  # Wspólny dla testowych i treningowych
        """
        The use of common transformations, as well as the use of previously learned parameters.

        Perform type conversion (e.g., convert values from Pandas Series to floating-point numbers).
        Normalize or scale the data using parameters learned earlier on the training set.
        Extract or calculate additional features from existing data.
        Apply possible augmentation operations (if working with images and want to introduce
         random transformations for the training data).
        """
        if test_data is None:
            df = self.data
        else:
            df = test_data

        scaled_array = self.scaler.transform(df[self.selected_columns])
        df.loc[:, self.selected_columns] = pd.DataFrame(scaled_array, columns=self.selected_columns, index=df.index)
        return self

    def select(self, row: pd.Series) -> torch.Tensor:
        # row = ... # filter on columns
        row = row[self.selected_columns]
        return torch.tensor(row.values, dtype=DTYPE)
