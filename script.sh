#!/bin/bash
#SBATCH --job-name=trening_modelu
#SBATCH --output=wyniki_%j.out
#SBATCH --error=blad_%j.err
#SBATCH --time=00:01:00              # Maksymalny czas (HH:MM:SS)
#SBATCH --ntasks=1                   # Liczba zadań
#SBATCH --cpus-per-task=1           # Liczba CPU na zadanie
#SBATCH --mem=16G                   # RAM
#SBATCH --partition=plgrid-short             # Kolejka (np. gpu / cpu)

module load python/3.10             # Załaduj odpowiedni moduł
source ~/venv/bin/activate          # Aktywuj wirtualne środowisko

python train.py                     # Uruchomienie skryptu treningowego

srun -p plgrid-short -N 1 --ntasks-per-node=1 -n 1 --time=00:01:00 --pty /bin/bash -l