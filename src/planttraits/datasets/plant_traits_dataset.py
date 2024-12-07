from torch.utils.data import Dataset
from planttraits.preprocessing.img_preprocessing import ImagePreprocessing
from planttraits.preprocessing.modis_vod_preprocessing import ModisVodPreprocessing
from planttraits.preprocessing.soli_preprocessing import SoilPreprocessing
from planttraits.preprocessing.worldclim_bio_preprocessing import WorldClimBioPreprocessing
from planttraits.preprocessing.std_preprocessing import StdPreprocessing
from planttraits.config import TRAIN_IMAGES_FOLDER, TEST_IMAGES_FOLDER, TRAIN_CSV_FILE, TEST_CSV_FILE
from planttraits.utils import TARGET_COLUMN_NAMES
import os 
import torch
import pandas as pd 

class PlantTraitsDataset(Dataset):

    def __init__(self, train : bool, # add here more arguments if needed
                 ):
        super().__init__()
        self.train = train 
        self.img_preprocess = ImagePreprocessing(train)
        self.modis_vod_preprocess = ModisVodPreprocessing(train)
        self.soil_preprocess = SoilPreprocessing(train)
        self.worldclimbio_preprocess = WorldClimBioPreprocessing(train)
        self.std_preprocess = StdPreprocessing(train)
        self.Y = pd.read_csv(TRAIN_CSV_FILE if train else TEST_CSV_FILE, usecols=TARGET_COLUMN_NAMES)

    def __len__(self):
        return len(os.listdir(TRAIN_IMAGES_FOLDER if self.train else TEST_IMAGES_FOLDER))
    
    def __getitem__(self, idx):
        img = self.img_preprocess.transform(idx)
        modisvod_row = self.modis_vod_preprocess.transform(idx)
        soil_row = self.soil_preprocess.transform(idx)
        worldclimbio_row = self.worldclimbio_preprocess.transform(idx)
        std_row = self.std_preprocess.transform(idx)

        y = torch.tensor(self.Y.iloc[idx].values)

        return img, modisvod_row, soil_row, worldclimbio_row, std_row, y