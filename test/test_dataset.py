from planttraits.datasets.plant_traits_dataset import PlantTraitsDataset
from planttraits.utils import TARGET_COLUMN_NAMES
from torch.utils.data import DataLoader

def test_train_dataset():
    train_dataset = PlantTraitsDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    img, modisvod_row, soil_row, worldclimbio_row, std_row, y = next(iter(train_dataloader))
    assert len(img.shape) == 4
    assert len(modisvod_row.shape) == 2
    assert len(soil_row.shape) == 2
    assert len(worldclimbio_row.shape) == 2
    assert len(std_row.shape) == 2
    assert len(y.shape) == 2
    assert y.shape[1] == len(TARGET_COLUMN_NAMES)