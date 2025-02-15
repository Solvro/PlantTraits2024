import numpy as np
import pandas as pd
import torch

from planttraits.config import TEST_CSV_FILE, TRAIN_CSV_FILE
from planttraits.preprocessing.img_preprocessing import ImagePreprocessing
from planttraits.preprocessing.mean_preprocessing import MeanPreprocessing
from planttraits.preprocessing.modis_vod_preprocessing import ModisVodPreprocessing
from planttraits.preprocessing.soli_preprocessing import SoilPreprocessing
from planttraits.preprocessing.worldclim_bio_preprocessing import WorldClimBioPreprocessing
from torch.utils.data import Dataset


class PlantTraitsDataset(Dataset):
    def __init__(
        self,
        is_train: bool,  # add here more arguments if needed
    ):
        super().__init__()
        self.is_train = is_train
        self.img_preprocess = ImagePreprocessing(self.is_train)
        self.modis_vod_preprocess = ModisVodPreprocessing(self.is_train)
        self.soil_preprocess = SoilPreprocessing(self.is_train)
        self.worldclimbio_preprocess = WorldClimBioPreprocessing(self.is_train)
        self.mean_preprocess = MeanPreprocessing(self.is_train)

        # Keep all the data
        self.data = pd.read_csv(TRAIN_CSV_FILE if self.is_train else TEST_CSV_FILE, index_col='id')

        self.indexes = self.data.index.to_numpy()
        self._adjust_indexes()

    def _adjust_indexes(self):
        self.drop_idxs = []
        for transform in [
            self.img_preprocess,
            self.modis_vod_preprocess,
            self.soil_preprocess,
            self.worldclimbio_preprocess,
            self.mean_preprocess,
        ]:
            self.drop_idxs.extend(transform.drop_idxs)
        self.indexes = np.setdiff1d(self.indexes, self.drop_idxs)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        true_idx = self.indexes[idx]
        row = self.data.loc[true_idx]
        img = self.img_preprocess.transform(true_idx)
        modisvod_row = self.modis_vod_preprocess.transform(row)
        soil_row = self.soil_preprocess.transform(row)
        worldclimbio_row = self.worldclimbio_preprocess.transform(row)
        mean_row, std_row = (
            self.mean_preprocess.transform(row) if self.is_train else (torch.zeros(1), torch.zeros(1))
        )  # mean_row, std_row ??

        return img, modisvod_row, soil_row, worldclimbio_row, std_row, mean_row
