from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from mpl_toolkits.axes_grid1 import ImageGrid
from torchmetrics.classification import Accuracy
from torchvision import transforms
from torchvision.transforms import v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = transforms.Compose(
    [transforms.PILToTensor(), transforms.Resize(size=(256, 256)), transforms.ConvertImageDtype(torch.float)]
)

type_transforms = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.Resize(size=(256, 256)),
        transforms.ConvertImageDtype(torch.float),
    ]
)


def compute_dim(d_in, padding, kernel, stride, pooling=1):
    return (int((d_in + 2 * padding - (kernel - 1) - 1) / stride) + 1) // pooling


class ConvLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=2, stride=1, padding=0, pooling=None, batch_norm=True, activ=nn.ReLU
    ):
        super(ConvLayer, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding)]

        if pooling:
            layers.append(nn.MaxPool2d(kernel_size=pooling))

        if activ:
            layers.append(activ())

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BaseCNN(nn.Module):
    def __init__(self, conv_layers: List, linear_layers: List[nn.Linear], act_fun=nn.ReLU):
        super(BaseCNN, self).__init__()
        self.conv_layers = nn.Sequential(*conv_layers)
        self.lin_layers = nn.ModuleList(linear_layers)
        self.act_fun = act_fun()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        for lin_layer in self.lin_layers[:-1]:
            x = self.act_fun(lin_layer(x))
        # Po ostatniej warstwie nie dodajemy już funkcji aktywacji bo chcemy, żeby sieć zwracała logit'y
        if self.lin_layers:
            x = self.lin_layers[-1](x)
        # x = self.lin_layers[-1](x)
        return x


class CNN(pl.LightningModule):
    def __init__(self, cnn, lr=1e-3, optim_alg=torch.optim.Adam):
        super().__init__()
        self.cnn = cnn
        self.lr = lr
        self.optim_alg = optim_alg
        self.metric = Accuracy(task='multiclass', num_classes=102)

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        train_acc = self.metric(predictions, y)
        self.log('train_acc', train_acc, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        val_acc = self.metric(predictions, y)
        self.log('val_acc', val_acc, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        predictions = self(x)
        predicted_labels = torch.argmax(predictions, dim=-1)
        return predicted_labels

    def configure_optimizers(self):
        # W przypadku skomplikowanych modeli można dodać więcej optymizatorów
        optimizer = self.optim_alg(self.parameters(), lr=self.lr)
        return optimizer


class GoogleNetModule(pl.LightningModule):
    def __init__(self):
        super(GoogleNetModule, self).__init__()
        self.gnet = torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights).to(device)
        # zamrożenie parametrów oryginalnego modelu
        for param in self.gnet.parameters():
            param.requires_grad_(False)
        # zamiana oryignalnego modułu klasyfikatora na nowy klasyfikator, który będzie trenowany do danego zadania
        self.gnet.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 102),
        )
        self.metric = Accuracy(task='multiclass', num_classes=102)

    def forward(self, x):
        return self.gnet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        train_acc = self.metric(predictions, y)
        self.log('train_acc', train_acc, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        val_acc = self.metric(predictions, y)
        self.log('val_acc', val_acc, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        predictions = self(x)
        predicted_labels = torch.argmax(predictions, dim=-1)
        return predicted_labels

    def configure_optimizers(self):
        # W przypadku skomplikowanych modeli można dodać więcej optymizatorów
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


class CNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=2, stride=1, padding=0, batch_norm=True, activ=nn.ReLU):
        super(CNLayer, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding)]

        if activ:
            layers.append(activ())

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TCNLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=2, stride=1, padding=0, output_padding=0, batch_norm=True, activ=nn.ReLU
    ):
        super(TCNLayer, self).__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel, stride=stride, padding=padding, output_padding=output_padding
            )
        ]

        if activ:
            layers.append(activ())

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CNNAutoencoder(pl.LightningModule):
    def __init__(self, act_fn=F.relu, lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        # N, 1, 256, 256
        self.encoder = nn.Sequential(
            CNLayer(3, 32, 3, stride=2, padding=1),  # N, 32, 128, 128
            CNLayer(32, 64, 3, stride=2, padding=1),  # N, 64, 64, 64
            CNLayer(64, 128, 3, stride=2, padding=1),  # N, 128, 32, 32
            CNLayer(128, 256, 3, stride=2, padding=1),  # N, 256, 16, 16
            CNLayer(256, 256, 3, stride=2, padding=1, activ=None),  # N, 256, 8, 8
            nn.AdaptiveAvgPool2d(output_size=2),  # N, 256, 2, 2
        )

        # now decoder operations are the reverse
        # some other operations, like MaxPool have their reverse
        self.decoder = nn.Sequential(
            TCNLayer(256, 256, 3, stride=2, padding=1, output_padding=1),  # N, 256, 4, 4
            TCNLayer(256, 128, 3, stride=2, padding=1, output_padding=1),  # N, 128, 8, 8
            TCNLayer(128, 64, 3, stride=2, padding=1, output_padding=1),  # N, 64, 16, 16
            TCNLayer(64, 64, 3, stride=2, padding=1, output_padding=1),  # N, 64, 32, 32
            TCNLayer(64, 32, 3, stride=2, padding=1, output_padding=1),  # N, 32, 64, 64
            TCNLayer(32, 3, 3, stride=2, padding=1, output_padding=1),  # N, 3, 128, 128
            TCNLayer(3, 3, 3, stride=2, padding=1, output_padding=1, activ=None),  # N, 3, 256, 256
            nn.Sigmoid(),
        )

        self.criterion = nn.BCELoss()  # Binary Cross Entropy
        self.act_fn = act_fn
        self.noise_transform = v2.GaussianNoise()

    def forward(self, x):
        x = self.noise_transform(x)
        x = self.decoder(self.encoder(x))
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        prediction = self(x)
        loss = self.criterion(prediction, x)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        prediction = self(x)
        loss = self.criterion(prediction, x)
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            fig = plt.figure(figsize=(18, 8))
            grid = ImageGrid(
                fig,
                111,  # similar to subplot(111)
                nrows_ncols=(2, 9),  # creates 2x9 grid of Axes
                axes_pad=0.1,  # pad between Axes in inch.
            )

            for ax, im in zip(grid, [*x[:9].detach().cpu().numpy(), *prediction[:9].detach().cpu().numpy()]):
                # Iterating over the grid returns the Axes.
                ax.imshow(im.transpose(1, 2, 0))
            tensorboard.add_figure(f'Step {self.current_epoch}', fig)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def kl_loss(z_var_log, z_mean):
    return -0.5 * torch.sum(1.0 + z_var_log - z_mean**2 - torch.exp(z_var_log))


class CNNVAE(pl.LightningModule):
    def __init__(self, z_dim, lr=1e-3, beta=1, **kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        self.beta = beta
        # N, 1, 256, 256
        self.encoder = nn.Sequential(
            CNLayer(3, 16, 3, stride=2, padding=1),  # N, 32, 128, 128
            CNLayer(16, 32, 3, stride=2, padding=1),  # N, 64, 64, 64
            CNLayer(32, 64, 3, stride=2, padding=1),  # N, 128, 32, 32
            CNLayer(64, 128, 3, stride=2, padding=1),  # N, 128, 16, 16
            CNLayer(128, 256, 3, stride=2, padding=1),  # N, 256, 8, 8
            CNLayer(256, 256, 3, padding=1),  # N, 256, 8, 8
            CNLayer(256, 512, 3, stride=2, padding=1),  # N, 256, 4, 4
            CNLayer(512, 512, 3, stride=2, padding=1),  # N, 256, 2, 2
            nn.Flatten(),
        )

        output_dim = 512 * 2 * 2
        self.m_linear = nn.Linear(output_dim, z_dim)
        self.v_linear = nn.Linear(output_dim, z_dim)

        self.linear = nn.Sequential(nn.Linear(z_dim, output_dim), nn.ReLU(), nn.BatchNorm1d(output_dim))

        # now decoder operations are the reverse
        self.decoder = nn.Sequential(
            TCNLayer(512, 512, 3, padding=1),  # N, 512, 2, 2
            TCNLayer(512, 256, 3, stride=2, padding=1, output_padding=1),  # N, 256, 4, 4
            TCNLayer(256, 256, 3, padding=1),  # N, 128, 8, 8
            TCNLayer(256, 128, 3, stride=2, padding=1, output_padding=1),  # N, 128, 8, 8
            TCNLayer(128, 64, 3, stride=2, padding=1, output_padding=1),  # N, 64, 16, 16
            TCNLayer(64, 64, 3, stride=2, padding=1, output_padding=1),  # N, 64, 32, 32
            TCNLayer(64, 32, 3, stride=2, padding=1, output_padding=1),  # N, 32, 64, 64
            TCNLayer(32, 16, 3, stride=2, padding=1, output_padding=1),  # N, 3, 128, 128
            TCNLayer(
                16, 3, 3, stride=2, padding=1, output_padding=1, activ=nn.Sigmoid, batch_norm=False
            ),  # N, 3, 256, 256
        )

        self.criterion = nn.BCELoss()  # Binary Cross Entropy,

    def forward(self, x):
        x = self.encoder(x)
        z_mean = self.m_linear(x)
        z_var = self.v_linear(x)
        # nie próbkujemy bezpośrednio z rozkładu,
        # ale korzystamy z przekształceń możliwych na rozkładach prawdopodobieństwa
        sample = torch.randn_like(z_var) * torch.exp(0.5 * z_var) + z_mean
        sample = self.linear(sample)
        sample = sample.view(-1, 512, 2, 2)
        decoded = self.decoder(sample)
        return decoded, z_mean, z_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        prediction, z_mean, z_var = self(x)
        recon_loss = self.criterion(prediction, x)
        kldiv_loss = kl_loss(z_var, z_mean)
        loss = self.beta * recon_loss + kldiv_loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_recon_loss', recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_kldiv_loss', kldiv_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        prediction, z_mean, z_var = self(x)
        recon_loss = self.criterion(prediction, x)
        kldiv_loss = kl_loss(z_var, z_mean)
        loss = self.beta * recon_loss + kldiv_loss
        if batch_idx == 0:
            tensorboard = self.logger.experiment
            fig = plt.figure(figsize=(18, 8))
            grid = ImageGrid(
                fig,
                111,  # similar to subplot(111)
                nrows_ncols=(2, 9),  # creates 2x9 grid of Axes
                axes_pad=0.1,  # pad between Axes in inch.
            )

            for ax, im in zip(grid, [*x[:9].detach().cpu().numpy(), *prediction[:9].detach().cpu().numpy()]):
                # Iterating over the grid returns the Axes.
                ax.imshow(im.transpose(1, 2, 0))
            tensorboard.add_figure(f'Step {self.current_epoch}', fig)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_recon_loss', recon_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_kldiv_loss', kldiv_loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        prediction, *_ = self(x)
        return prediction

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
