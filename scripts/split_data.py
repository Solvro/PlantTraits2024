from pathlib import Path
import pandas as pd
from planttraits.config import (
    TRAIN_CSV_FILE,
    TRAIN_IMAGES_FOLDER,
    VAL_CSV_FILE, 
    VAL_IMAGES_FOLDER
)
from sklearn.model_selection import train_test_split
import argparse
import os 
import shutil

def split_indices(val_size, random_state=None):
    data = pd.read_csv(TRAIN_CSV_FILE, index_col='id')
    indices = data.index.values
    train_indices, test_indices = train_test_split(indices, test_size=val_size)
    return train_indices, test_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split train dataset to train and val sets")
    parser.add_argument("--val_size", type=float, help="validation fraction between 0-1")
    args = parser.parse_args()

    df = pd.read_csv(TRAIN_CSV_FILE, index_col='id')
    train_indices, val_indices = split_indices(val_size=args.val_size)
    print(f"Train size: {len(train_indices)}")
    print(f"Val size: {len(val_indices)}")

    train_df = df.loc[train_indices]
    train_df.to_csv(TRAIN_CSV_FILE)
    val_df = df.loc[val_indices]
    val_df.to_csv(VAL_CSV_FILE)

    os.makedirs(VAL_IMAGES_FOLDER, exist_ok=True)

    images_files = os.listdir(TRAIN_IMAGES_FOLDER)
    for file in images_files:
        if int(file.split('.')[0]) in val_indices:
            shutil.move(TRAIN_IMAGES_FOLDER / file, VAL_IMAGES_FOLDER / file)
    
