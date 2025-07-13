import os
from typing import List

import pandas as pd
import torch

from planttraits.config import (
    TEST_CSV_FILE,
    TEST_IMAGES_FOLDER,
    TRAIN_CSV_FILE,
    TRAIN_IMAGES_FOLDER,
    VAL_CSV_FILE,
    VAL_IMAGES_FOLDER
)
from planttraits.utils import TARGET_COLUMNS_MAPPING, SUBMISSION_COLUMNS, TARGET_COLUMN_NAMES
from planttraits.preprocessing.img_preprocessing import ImagePreprocessing
from planttraits.preprocessing.mean_preprocessing import MeanPreprocessing
from planttraits.preprocessing.modis_vod_preprocessing import ModisVodPreprocessing
from planttraits.preprocessing.soil_preprocessing import SoilPreprocessing
from planttraits.preprocessing.worldclim_bio_preprocessing import (
    WorldClimBioPreprocessing,
)
from torch.utils.data import Dataset


class PlantTraitsDataset(Dataset):
    def __init__(
        self,
        preprocessors=None,  # add here more arguments if needed
        eval=False
    ):
        super().__init__()
        self._preprocessors = {} if preprocessors is None else dict(preprocessors)
        self.is_eval = eval
        self.is_train = preprocessors is None
        self.data = pd.read_csv(TRAIN_CSV_FILE if self.is_train else (VAL_CSV_FILE if eval else TEST_CSV_FILE), index_col='id')

        self.imgs_folder = TRAIN_IMAGES_FOLDER if self.is_train else (VAL_IMAGES_FOLDER if eval else TEST_IMAGES_FOLDER)
        self.imgs_paths = {int(p.split('.')[0]): self.imgs_folder / p for p in os.listdir(self.imgs_folder)}

        if self.is_train:
            # Tutaj też implicitly dokona się prepare_data i fit_preprocessing
            self._preprocessors = {
                'modis_vod': ModisVodPreprocessing(self.data),  # tutaj im przekazuje data, to będzie train zawsze
                'soil': SoilPreprocessing(self.data),
                'worldclimbio': WorldClimBioPreprocessing(self.data),
                'mean': MeanPreprocessing(self.data),
            }
            self.img_preprocessor = ImagePreprocessing(
                self.imgs_paths
            )
        else:
            for key, preprocessor in self._preprocessors.items():
                if key != 'mean':
                    preprocessor.prepare_data(self.data).transform_preprocessing(
                        self.data
                    )  # Tutaj też przekazuje data, ale to będzie typowo test i tylko prepare
            self.img_preprocessor = ImagePreprocessing(
                self.imgs_paths, is_train=False
            )
            if self.is_eval:
                self._preprocessors['mean'] = MeanPreprocessing(self.data)

    def return_preprocessors(self):
        return self._preprocessors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        true_idx = self.data.index[idx]
        row = self.data.loc[true_idx]
        img = self.img_preprocessor.transform(self.imgs_paths[true_idx])
        # transformacje są wykonane w miejscu na danych więc tylko trzeba odpowiednie kolumny wybrać
        modisvod_row = self._preprocessors['modis_vod'].select(row)
        soil_row = self._preprocessors['soil'].select(row)
        worldclimbio_row = self._preprocessors['worldclimbio'].select(row)
        mean_row, std_row = (
            self._preprocessors['mean'].select(row) if (self.is_train or self.is_eval) else (torch.zeros(1), torch.zeros(1))
        )

        return img, modisvod_row, soil_row, worldclimbio_row, std_row, mean_row

    def transform_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        return self._preprocessors['mean'].reverse_transform(preds)

    def save_submission(self, predictions: List[torch.Tensor], path: str):
        predictions = self.transform_predictions(torch.concat(predictions))
        submission = pd.DataFrame(predictions, index=self.data.index, 
                                  columns=TARGET_COLUMN_NAMES).reset_index().rename({'index': 'id'}, axis=1)
        submission = submission.rename(TARGET_COLUMNS_MAPPING, axis=1)
        submission = submission[SUBMISSION_COLUMNS]
        submission.to_csv(path, index=False)
        return submission
        