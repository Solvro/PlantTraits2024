import torch
from torchvision import transforms
from torchvision.io import read_image
from planttraits.utils import DTYPE

class ImagePreprocessing:
    def __init__(self, imgs_paths, is_train = True):
        self.imgs_paths = imgs_paths
        self._fit_preprocessing()
        self.is_train = is_train
        
    def _fit_preprocessing(self):
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])

        self.test_transform = transforms.Compose([
            transforms.ConvertImageDtype(DTYPE)
        ])


    def _transform_preprocessing(self, img_path):
        image = read_image(img_path)
        image = image.float() / 255.0
        if self.is_train:
            image = self.train_transform(image)
        image = self.test_transform(image)
        return image

    def transform(self, img_path) -> torch.Tensor:
        return self._transform_preprocessing(img_path)