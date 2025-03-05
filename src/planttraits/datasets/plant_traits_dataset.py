import numpy as np
import pandas as pd
import torch

from planttraits.config import (
    TEST_CSV_FILE,
    TEST_IMAGES_FOLDER,
    TRAIN_CSV_FILE,
    TRAIN_IMAGES_FOLDER,
)
from planttraits.preprocessing.img_preprocessing import ImagePreprocessing
from planttraits.preprocessing.mean_preprocessing import MeanPreprocessing
from planttraits.preprocessing.modis_vod_preprocessing import ModisVodPreprocessing
from planttraits.preprocessing.soli_preprocessing import SoilPreprocessing
from planttraits.preprocessing.worldclim_bio_preprocessing import (
    WorldClimBioPreprocessing,
)
from torch.utils.data import Dataset


class PlantTraitsDataset(Dataset):
    def __init__(
        self,
        preprocessors=None,  # add here more arguments if needed
    ):
        super().__init__()
        self._preprocessors = {} if preprocessors is None else preprocessors
        self.is_train = preprocessors is None
        self.data = pd.read_csv(TRAIN_CSV_FILE if self.is_train else TEST_CSV_FILE, index_col='id')
        self.imgs_folder = TRAIN_IMAGES_FOLDER if self.is_train else TEST_IMAGES_FOLDER

        if self.is_train:
            # Tutaj też implicitly dokona się prepare_data i fit_preprocessing
            self._preprocessors = {
                'img': ImagePreprocessing(
                    self.imgs_folder
                ),  # Tutaj nie nie chciałem ingerować za bardzo w zdjęcia, więc przekazuję imgs_folder
                'modis_vod': ModisVodPreprocessing(self.data),  # tutaj im przekazuje data, to będzie train zawsze
                'soil': SoilPreprocessing(self.data),
                'worldclimbio': WorldClimBioPreprocessing(self.data),
                'mean': MeanPreprocessing(self.data),
            }
        else:
            for key, preprocessor in self._preprocessors.items():
                if key == 'img':
                    preprocessor.prepare_data(self.imgs_folder)
                else:
                    preprocessor.prepare_data(
                        self.data
                    )  # Tutaj też przekazuje data, ale to będzie typowo test i tylko prepare

        # Keep all the data
        self._adjust_indexes()

    def return_preprocessors(self):
        return self._preprocessors

    def _adjust_indexes(self):
        self.indexes = self.data.index.to_numpy()
        self.drop_idxs = []
        for transform in self._preprocessors.values():
            self.drop_idxs.extend(transform.drop_idxs)
        self.indexes = np.setdiff1d(self.indexes, self.drop_idxs)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        true_idx = self.indexes[idx]
        row = self.data.loc[true_idx]
        img = self._preprocessors['img'].transform(true_idx)
        modisvod_row = self._preprocessors['modis_vod'].transform(row)
        soil_row = self._preprocessors['soil'].transform(row)
        worldclimbio_row = self._preprocessors['worldclimbio'].transform(row)
        mean_row, std_row = (
            self._preprocessors['mean'].transform(row) if self.is_train else (torch.zeros(1), torch.zeros(1))
        )

        return img, modisvod_row, soil_row, worldclimbio_row, std_row, mean_row
