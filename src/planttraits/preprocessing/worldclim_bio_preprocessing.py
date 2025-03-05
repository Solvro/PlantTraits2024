import pandas as pd
import torch

from planttraits.utils import DTYPE


class WorldClimBioPreprocessing:
    def __init__(
        self,
        data,  # add more parameters if necessary
    ):
        self.csv_file = data
        # list of removed sample indexes (don't delete it nor set to None if not used)
        self.drop_idxs = []
        # filter only necessary columns
        self.prepare_data(self.csv_file)
        self._fit_preprocessing()

    def prepare_data(self, data):
        """
        Unifying and “cleaning” the data, which makes sense to do regardless of whether
         the data comes from a training or test collection.

        Deleting unnecessary columns or rows for both train and test.
        Unification data types ex. conversion to appropriate num-categorical types.
        Standardization of format ex. size of letters, deleting spaces.
        Other cleansing operations, which don't rely on "learning parameters", only on transforming
        raw data to concise format.
        """
        pass

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
        pass

    def _transform_preprocessing(self, row):  # Wspólny dla testowych i treningowych
        """
        The use of common transformations, as well as the use of previously
         learned parameters to process a single line (sample).

        Perform type conversion (e.g., convert values from Pandas Series to floating-point numbers).
        Normalize or scale the data using parameters learned earlier on the training set.
        Extract or calculate additional features from existing data.
        Apply possible augmentation operations (if working with images and want to introduce
         random transformations for the training data).
        """
        pass

    def transform(self, row: pd.Series) -> torch.Tensor:
        # row = ... # filter on columns
        self._transform_preprocessing(row)
        # some transformations on a single row of data if necessary

        return torch.tensor(row.values, dtype=DTYPE)
