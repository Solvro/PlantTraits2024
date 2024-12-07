# Plant Traits 2024
Uncovering the biosphere: Predicting 6 Vital Plant Traits from Plant Images for Ecosystem Health

## Uczestnicy
- 

## Challenge
https://www.kaggle.com/competitions/planttraits2024

## Dane

- [Dataset](https://www.kaggle.com/competitions/planttraits2024/data)

Stwórzcie plik .env z takimi zmiennymi jak są w pliku src/utils.py

## Instalacja środowiska

1. Create new virtual environment:
    
    ```
    conda create --name plant-traits python=3.10
    ```

2. Activate environment
    ```
    conda activate plant-traits
    ```

3. Update _pip_ version:
    ```
    python -m pip install --upgrade pip
    ```
4. Install packages:

    ```
    python -m pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124
    ```
5. Enable precommit hook:
    ```
    pre-commit install
    ```

## Struktura plikowa

    .
    |-- build
    |-- data
    |   |-- test_images
    |   `-- train_images
    |-- notebooks
    |-- src
    |   |-- __init__.py
    |   |-- planttraits
    |   |   |-- __init__.py
    |   |   |-- config.py
    |   |   |-- datasets
    |   |   |   `-- plant_traits_dataset.py
    |   |   |-- models
    |   |   |-- preprocessing
    |   |   |   |-- __init__.py
    |   |   |   |-- img_preprocessing.py
    |   |   |   |-- modis_vod_preprocessing.py
    |   |   |   |-- soli_preprocessing.py
    |   |   |   |-- std_preprocessing.py
    |   |   |   `-- worldclim_bio_preprocessing.py
    |   |   `-- utils.py
    `-- test
        |-- __init__.py
        `-- test_dataset.py

## Testowanie
```
    pytest test
```

