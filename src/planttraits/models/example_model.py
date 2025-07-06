import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

from src.planttraits.notebook_utils import ConvLayer, BaseCNN
from torchmetrics.functional import mean_squared_error, r2_score
from torchvision import transforms

# Suppress all warnings
warnings.filterwarnings('ignore')

train_transforms = transforms.Compose(
    [
        transforms.Resize(size=(232, 232)),
        transforms.ConvertImageDtype(torch.float),
        transforms.CenterCrop(size=224),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

conv_layers = [
    ConvLayer(3, 32, 3, padding=1, stride=1, batch_norm=True),  # C, W, H = ()
    ConvLayer(32, 64, 3, padding=1, stride=2, batch_norm=True),
    ConvLayer(64, 128, 3, padding=1, stride=1, batch_norm=True),
    ConvLayer(128, 256, 3, padding=1, stride=2, batch_norm=True),
    ConvLayer(256, 512, 3, padding=1, stride=2, batch_norm=True),
    nn.AdaptiveAvgPool2d(output_size=1)
]

efficientnet_b2 = torchvision.models.efficientnet_b2() # too heavy

custom_backbone_net = BaseCNN(conv_layers, linear_layers=[])

for param in efficientnet_b2.parameters():
    param.requires_grad_(False)

kwargs = {'n_feat': 122, 'backbone_net': custom_backbone_net, 'criteria': mean_squared_error}


class PTNN(pl.LightningModule):
    def __init__(self, n_feat, criteria, lr=1e-3, backbone_net=efficientnet_b2, optim_alg=torch.optim.Adam):
        super().__init__()
        self.lr = lr
        self.optim_alg = optim_alg
        self.n_feat = n_feat
        self.criteria = criteria
        self.climate_encoder = nn.Sequential(
            nn.Linear(n_feat, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
        )

        self.img_encoder = nn.Sequential(
            backbone_net, nn.Linear(512, 256), nn.LayerNorm(256)
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    def training_step(self, batch, batch_idx):
        predictions, std_row, mean_row = self.run_forward(batch)
        loss = self.criteria(predictions, mean_row)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_score', r2_score(predictions, mean_row), prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        predictions, std_row, mean_row = self.run_forward(batch)
        loss = self.criteria(predictions, mean_row)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_score', r2_score(predictions, mean_row), prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, *_ = self.run_forward(batch)
        return predictions

    def forward(self, img, row):
        climate_features = self.climate_encoder(row)
        image_features = self.img_encoder(img)
        features = torch.concat([climate_features, image_features], dim=1)
        pred = self.prediction_head(features)
        return pred

    def run_forward(self, batch):
        img, modisvod_row, soil_row, worldclimbio_row, std_row, mean_row = batch
        row = torch.concat([modisvod_row, soil_row, worldclimbio_row], axis=1)
        img = train_transforms(img)
        return self.forward(img, row), std_row, mean_row

    def configure_optimizers(self):
        # W przypadku skomplikowanych modeli można dodać więcej optymizatorów
        optimizer = self.optim_alg(self.parameters(), lr=self.lr)
        return optimizer
