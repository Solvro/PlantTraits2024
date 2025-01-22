import torch

from planttraits.datasets.plant_traits_dataset import PlantTraitsDataset
from planttraits.utils import TARGET_COLUMN_NAMES
from torch.utils.data import DataLoader


def test_train_dataset_input_shapes():
    train_dataset = PlantTraitsDataset(is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    img, modisvod_row, soil_row, worldclimbio_row, std_row, y = next(iter(train_dataloader))
    assert len(img.shape) == 4
    assert len(modisvod_row.shape) == 2
    assert len(soil_row.shape) == 2
    assert len(worldclimbio_row.shape) == 2
    assert len(std_row.shape) == 2
    assert len(y.shape) == 2
    assert y.shape[1] == len(TARGET_COLUMN_NAMES)


def test_train_dataset_output_shapes():
    train_dataset = PlantTraitsDataset(is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    *_, std, y = next(iter(train_dataloader))
    assert len(y.shape) == 2
    assert len(std.shape) == 2
    assert y.shape[1] == len(TARGET_COLUMN_NAMES)
    assert std.shape[1] == len(TARGET_COLUMN_NAMES)


def test_test_dataset_output_shapes():
    test_dataset = PlantTraitsDataset(is_train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    *_, std, y = next(iter(test_dataloader))
    assert torch.allclose(y[0], torch.tensor([0.0]))
    assert torch.allclose(std[0], torch.tensor([0.0]))
