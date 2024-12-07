from planttraits.config import TRAIN_CSV_FILE, TEST_CSV_FILE, TRAIN_IMAGES_FOLDER, TEST_IMAGES_FOLDER
from torchvision.io import read_image
import os 
import torch 

class ImagePreprocessing:

    def __init__(self, train : bool, # add more parameters if necessary
                 ):
        self.imgs_folder = TRAIN_IMAGES_FOLDER if train else TEST_IMAGES_FOLDER
        self.imgs_paths = os.listdir(self.imgs_folder)

        self.preprocess()

    def preprocess(self):
        # inner computations like computing means itd if necessary
        pass 
    
    def transform(self, idx) -> torch.Tensor:
        img_path = self.imgs_folder / self.imgs_paths[idx]
        image = read_image(img_path)

        # here some transformations on a single image
        # https://pytorch.org/vision/0.11/transforms.html

        return torch.tensor(image)