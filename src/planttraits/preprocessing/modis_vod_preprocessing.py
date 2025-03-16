import numpy as np
import pandas as pd
import torch

from planttraits.utils import DTYPE
from sklearn.preprocessing import StandardScaler

# To co robimy to jest świadome mutowanie obiektów i uzyskiwanie side_effectu


class ModisVodPreprocessing:
    def __init__(self, data):
        self.data = data  # Zmieniłem nazwę na data // poprzednio csv_file

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

        # Łącze VOD_C i VOD_X w VOD_CX_month
        months = [f'm{str(i).zfill(2)}' for i in range(1, 13)]
        for m in months:
            c_col = next((c for c in df if 'VOD_C' in c and m in c), None)
            x_col = next((c for c in df if 'VOD_X' in c and m in c), None)
            if c_col and x_col:
                df[f'VOD_CX_{m}'] = df[[c_col, x_col]].mean(axis=1)
                df.drop(columns=[c_col, x_col], inplace=True)

        # Agregacja na pory roku

        seasons = {
            'Winter': ['m12', 'm01', 'm02'],
            'Spring': ['m03', 'm04', 'm05'],
            'Summer': ['m06', 'm07', 'm08'],
            'Autumn': ['m09', 'm10', 'm11'],
        }

        for season, ms in seasons.items():
            # VOD_CX sezon
            cx_cols = [f'VOD_CX_{m}' for m in ms if f'VOD_CX_{m}' in df.columns]
            if cx_cols:
                df[f'VOD_CX_{season}'] = df[cx_cols].mean(axis=1)
                df.drop(columns=cx_cols, inplace=True)  # usuwam miesięczne
            # VOD_Ku sezon (jeśli masz jakieś VOD_Ku_* miesięczne)
            ku_cols = [c for c in df.columns if 'VOD_Ku' in c and any(m in c for m in ms)]
            if ku_cols:
                df[f'VOD_Ku_{season}'] = df[ku_cols].mean(axis=1)
                df.drop(columns=ku_cols, inplace=True)

        # Scalanie MODIS: band_01 i band_04 -> nowa kolumna band_14
        months_simple = [f'{i}' for i in range(1, 13)]
        for month in months_simple:
            band_01_col = next((c for c in df.columns if 'band_01' in c and f'_month_m{month}' in c), None)
            band_04_col = next((c for c in df.columns if 'band_04' in c and f'_month_m{month}' in c), None)
            if band_01_col and band_04_col:
                new_col = f'MODIS_2000.2020_monthly_mean_surface_reflectance_band_14_._month_m{month}'
                df[new_col] = df[[band_01_col, band_04_col]].mean(axis=1)
                df.drop(columns=[band_01_col, band_04_col], inplace=True)

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
        self.columns = [c for c in df_clean.columns if c.startswith('VOD_') or c.startswith('MODIS_')]
        # Dzięki temu w transformacji testu, wiemy dokładnie, z jakich kolumn zbudować macierz danych do
        # scaler.transform

        # Progi outlierów (1-99%)

        self.clip_values = {col: (df_clean[col].quantile(0.01), df_clean[col].quantile(0.99)) for col in self.columns}
        # To słownik dla każdej kolumny wartość 1%, 99% percentyla na treningu
        # Gdyby przycinać test według innych, albo nie przycinać wcale, to model zobaczyłby niepożądane wartości
        # odstające lub zupełnie inny rozkład niż podczas uczenia

        # Lista kolumn MODIS do log1p
        self.modis_cols = [c for c in self.columns if c.startswith('MODIS_')]
        # W testowym pipeline, trzeba wiedzieć, które kolumny logarytmować, a które pozostawić surowe.

        # 5) Przygotuj kopię tylko cech, zastosuj przycinanie i log1p do fitowania scalera
        df_for_scaler = df_clean[
            self.columns
        ].copy()  # kopia, dlatego, że chce zmodyfikować tylko to do scalera, a nie zmieniać wewnątrzenie z outside
        # effectem
        for col, (low, high) in self.clip_values.items():
            df_for_scaler[col].clip(lower=low, upper=high, inplace=True)
        df_for_scaler[self.modis_cols] = np.log1p(
            df_for_scaler[self.modis_cols]
        )  # Niby nowy obiekt, ale inplace zmienia się po przypisaniu ten nowy obiekt inplace

        #  Scaler na przetworzonych danych
        # Tutaj, można zrobić w konstuktorze na podstawie wyboru jaki scaler, ale na razie na sztywno
        self.scaler = StandardScaler().fit(df_for_scaler.values)

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

        df[self.modis_cols] = np.log1p(df[self.modis_cols])

        X = df[self.columns].values

        scaled_vals = self.scaler.transform(X)

        df.loc[:, self.columns] = scaled_vals

        # Używamy metody .loc, która bierze wszystkie wiersze o kolumnach z self.modis_cols i nadpisuje je tablicą
        # scaled_vals (o takiej samej liczbie wierszy i kolumn)

        return self

    def select(self, row: pd.Series) -> torch.Tensor:
        new_row = row[self.columns]

        return torch.tensor(new_row.values, dtype=DTYPE)
