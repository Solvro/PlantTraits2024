import numpy as np
import pandas as pd
import torch

from planttraits.utils import DTYPE
from sklearn.preprocessing import RobustScaler

# To co robimy to jest świadome mutowanie obiektów i uzyskiwanie side_effectu


class SoilPreprocessing:
    def __init__(self, data, log_transform: bool = False):
        self.data = data  # Zmieniłem nazwę na data // poprzednio csv_file

        # Store the user’s choice
        self.log_transform = log_transform

        # Miejsce na parametry fitu - inicjalizujemy pustymi wartościami // potrzebujemy tego, bo później instancja
        # testowa będzie z nich korzystać na zewnątrz
        self.columns = None
        self.clip_values = None
        self.modis_cols = None
        self.scaler = None

        # Pipeline treningowy
        self.prepare_data()._fit_preprocessing().transform_preprocessing()

    def prepare_data(self, test_data: pd.DataFrame = None):
        """
        Unifying and “cleaning” the data, which makes sense to do regardless
         of whether the data comes from a training or test collection.

        Deleting unnecessary columns or rows for both train and test.
        Unification data types ex. conversion to appropriate num-categorical types.
        Standardization of format ex. size of letters, deleting spaces.
        Other cleansing operations, which don't rely on "learning parameters", only on transforming
        raw data to concise format.
        """

        if test_data is None:
            df = self.data  # referencja
        else:
            df = test_data

        return self  # Bo łączymy kaskadowo wywołania w konstruktorze

    def _fit_preprocessing(self):  # Tylko treningowe, zastosowanie fit i scalera i wyliczanie wartosci tylko raz
        """
        Calculate parameters (e.g., mean, variance, coding maps, PCA matrix, etc.)
        that are needed for subsequent data transformation.

        Computation of statistics on cleaned data ex. avg, std_dv for scaler.
        Teaching coder for categorical variables. (we don't have any I think)
        Computation of reduction dimensionality parameters which are learning on data distribution.
        """

        df_clean = self.data  # Tutaj już wyczyszczony, bo najpierw wywołany jest prepare_data

        # Ustalamy na jakich kolumnach pracujemy
        self.columns = [c for c in df_clean.columns if c.startswith('SOIL_')]
        # Dzięki temu w transformacji testu, wiemy dokładnie, z jakich kolumn zbudować macierz danych do
        # scaler.transform

        # Progi outlierów (1-99%)

        self.clip_values = {col: (df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)) for col in self.columns}
        # To słownik dla każdej kolumny wartość 1%, 99% percentyla na treningu
        # Gdyby przycinać test według innych, albo nie przycinać wcale, to model zobaczyłby niepożądane wartości
        # odstające lub zupełnie inny rozkład niż podczas uczenia

        # Lista kolumn SOIL do log1p
        if self.log_transoform:
            self.soil_cols = [c for c in self.columns if c.startswith('SOIL_')]
        else:
            self.soil_cols = []
        # W testowym pipeline, trzeba wiedzieć, które kolumny logarytmować, a które pozostawić surowe.

        # 5) Przygotuj kopię tylko cech, zastosuj przycinanie i log1p do fitowania scalera
        df_for_scaler = df_clean[
            self.columns
        ].copy()  # kopia, dlatego, że chce zmodyfikować tylko to do scalera, a nie zmieniać wewnątrzenie z outside
        # effectem
        for col, (low, high) in self.clip_values.items():
            df_for_scaler[col].clip(lower=low, upper=high, inplace=True)

        df_for_scaler[self.soil_cols] = np.log1p(
            df_for_scaler[self.soil_cols]
        )  # Niby nowy obiekt, ale inplace zmienia się po przypisaniu ten nowy obiekt inplace

        #  Scaler na przetworzonych danych
        # Tutaj, można zrobić w konstuktorze na podstawie wyboru jaki scaler, ale na razie na sztywno
        self.scaler = RobustScaler().fit(df_for_scaler.values)

        return self

    def transform_preprocessing(self, test_data: pd.DataFrame = None):  # Wspólny dla testowych i treningowych
        """
        The use of common transformations, as well as the use of previously learned parameters.

        Perform type conversion (e.g., convert values from Pandas Series to floating-point numbers).
        Normalize or scale the data using parameters learned earlier on the training set.
        Extract or calculate additional features from existing data.
        Apply possible augmentation operations (if working with images and want to introduce random
        transformations for the training data).
        """
        if test_data is None:
            df = self.data
        else:
            df = test_data

        for col, (low, high) in self.clip_values.items():
            df[col].clip(lower=low, upper=high, inplace=True)

        # Apply log1p if requested
        if self.log_transform and self.soil_cols:
            df[self.soil_cols] = np.log1p(df[self.soil_cols])

        df[self.soil_cols] = np.log1p(df[self.soil_cols])

        X = df[self.columns].values

        scaled_vals = self.scaler.transform(X)

        df.loc[:, self.columns] = scaled_vals

        # Używamy metody .loc, która bierze wszystkie wiersze o kolumnach z self.modis_cols i nadpisuje je tablicą
        # scaled_vals (o takiej samej liczbie wierszy i kolumn)

        return self

    def select(self, row: pd.Series) -> torch.Tensor:
        new_row = row[self.columns]

        return torch.tensor(new_row.values, dtype=DTYPE)
