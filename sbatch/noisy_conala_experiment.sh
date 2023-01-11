#!/bin/bash

#SBATCH --mem 64000
#SBATCH --gres=gpu:1
#SBATCH --partition LARGE-G2
#SBATCH --output output/logs

export PYTHONUNBUFFERED=1

python train.py configs/train/conala_noisy_100.yaml
python train.py configs/train/conala_noisy_90.yaml
python train.py configs/train/conala_noisy_80.yaml
python train.py configs/train/conala_noisy_70.yaml
python train.py configs/train/conala_noisy_60.yaml
python train.py configs/train/conala_noisy_50.yaml
python train.py configs/train/conala_noisy_40.yaml
python train.py configs/train/conala_noisy_30.yaml
python train.py configs/train/conala_noisy_20.yaml
python train.py configs/train/conala_noisy_10.yaml
python train.py configs/train/conala_noisy_0.yaml
