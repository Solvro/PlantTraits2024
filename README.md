# Plant Traits 2024
Uncovering the biosphere: Predicting 6 Vital Plant Traits from Plant Images for Ecosystem Health

## Uczestnicy
- Dominika
- Anna
- Szymon
- Jakub
- Grzegorz
- Julia

## Challenge
https://www.kaggle.com/competitions/planttraits2024

## Dane

- [Dataset](https://www.kaggle.com/competitions/planttraits2024/data)

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
    python -m pip install -e .[dev]
    ```

    If you have cuda (here 12.4, but you can adjust the version) then 

    ```
    python -m pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu124
    ```

5. Enable precommit hook:
    ```
    pre-commit install
    ```

## Zadania na 1 etap
1. Wykonajcie analizę danych (EDA) na przypisanym sobie podzbiorze danych - do tego celu możecie stworzyć notebook w folderze 'notebooks'
2. Na podstawie wykonanej analizy określcie, czy dane wymagają jakiegoś preprocessingu (np. uzupełnienie brakujących danych, obsługa odstających wartości, normalizacja) i uzupełnijcie odpowiednie pliki w module preprocessing. W najprostszym przypadku wymaga to tylko wybrania odpowiednich kolumn. Uważajcie też, żeby nie doszło do wycieku danych w trakcie tego preprocessingu - nie mieszajcie zbiorów testowych i treningowych. 
3. Odpalcie test w folderze test'y - to jest bardzo basic check, ale może będziemy w trakcie dodawać jakieś testy żeby badać poprawność chociażby rozmiarów danych.

Stwórzcie sobie oddzielnego brancha i na nim pracujcie narazie lokalnie, dopóki nie macie dostępu do github'a.

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
    |
    `-- .env

## Testowanie
```
pytest test
```

## Zmienne środowiskowe
Przykładowe uzupełnienie pliku .env
```
ROOT_DIR = "C:\Users\julia\VSCode\PlantTraits2024"
TEST_IMAGES_FOLDER="data/test_images"
TRAIN_IMAGES_FOLDER="data/train_images"
TEST_CSV_FILE="data/test.csv"
TRAIN_CSV_FILE="data/train.csv"
```
