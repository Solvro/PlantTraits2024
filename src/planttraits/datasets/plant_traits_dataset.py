import numpy as np
import pandas as pd
import torch

from planttraits.config import TEST_CSV_FILE, TRAIN_CSV_FILE
from planttraits.preprocessing.img_preprocessing import ImagePreprocessing
from planttraits.preprocessing.modis_vod_preprocessing import ModisVodPreprocessing
from planttraits.preprocessing.soli_preprocessing import SoilPreprocessing
from planttraits.preprocessing.std_preprocessing import StdPreprocessing
from planttraits.preprocessing.worldclim_bio_preprocessing import WorldClimBioPreprocessing
from planttraits.utils import TARGET_COLUMN_NAMES
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
        self.std_preprocess = StdPreprocessing(self.is_train)

        # Keep all the data
        self.data = pd.read_csv(TRAIN_CSV_FILE if self.is_train else TEST_CSV_FILE)
        self.Y = self.data[TARGET_COLUMN_NAMES] if self.is_train else None

        self.indexes = np.arange(len(self.data))
        self._adjust_indexes()

    def _adjust_indexes(self):
        self.drop_idxs = []
        for transform in [
            self.img_preprocess,
            self.modis_vod_preprocess,
            self.soil_preprocess,
            self.worldclimbio_preprocess,
            self.std_preprocess,
        ]:
            self.drop_idxs.extend(transform.drop_idxs)
        self.indexes = np.setdiff1d(self.indexes, self.drop_idxs)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        true_idx = self.indexes[idx]
        row = self.data.iloc[true_idx]
        img = self.img_preprocess.transform(true_idx)
        modisvod_row = self.modis_vod_preprocess.transform(row)
        soil_row = self.soil_preprocess.transform(row)
        worldclimbio_row = self.worldclimbio_preprocess.transform(row)
        std_row = self.std_preprocess.transform(row)

        y = torch.tensor(self.Y.iloc[true_idx].values) if self.is_train else torch.empty(1)

        return img, modisvod_row, soil_row, worldclimbio_row, std_row, y
