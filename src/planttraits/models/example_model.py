import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision

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


efficientnet_b2 = torchvision.models.efficientnet_b2()

for param in efficientnet_b2.parameters():
    param.requires_grad_(False)

kwargs = {'n_feat': 163, 'backbone_net': efficientnet_b2, 'criteria': mean_squared_error}


class PTNN(pl.LightningModule):
    def __init__(self, n_feat, criteria, lr=1e-3, backbone_net=efficientnet_b2, optim_alg=torch.optim.Adam):
        super().__init__()
        self.lr = lr
        self.optim_alg = optim_alg
        self.n_feat = n_feat
        self.criteria = criteria
        self.climate_encoder = nn.Sequential(
            nn.Linear(n_feat, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(1000, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
        )

        self.img_encoder = nn.Sequential(
            backbone_net, nn.Linear(1000, 512), nn.LayerNorm(512), nn.ReLU(), nn.Linear(512, 256), nn.LayerNorm(256)
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )

    def training_step(self, batch, batch_idx):
        predictions, mean_row = self.run_forward(batch)
        loss = self.criteria(predictions, mean_row)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_score', r2_score(predictions, mean_row), prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        predictions, mean_row = self.run_forward(batch)
        loss = self.criteria(predictions, mean_row)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_score', r2_score(predictions, mean_row), prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, _ = self.run_forward(batch)
        return predictions

    def forward(self, img, row):
        climate_features = self.climate_encoder(torch.concat([row], dim=1))
        image_features = self.img_encoder(img)
        features = torch.concat([climate_features, image_features], dim=1)
        pred = self.prediction_head(features)
        return pred

    def run_forward(self, batch):
        img, row, *_, mean_row = batch
        row = row[:, : self.n_feat]
        img = train_transforms(img)
        return self.forward(img, row), mean_row

    def configure_optimizers(self):
        # W przypadku skomplikowanych modeli można dodać więcej optymizatorów
        optimizer = self.optim_alg(self.parameters(), lr=self.lr)
        return optimizer
