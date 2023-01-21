#!/bin/bash

#SBATCH --mem 64000
#SBATCH --gres=gpu:1
#SBATCH --partition LARGE-G2
#SBATCH --output output/logs

export PYTHONUNBUFFERED=1

python predict.py configs/predict/conala_noisy_100.yaml
python predict.py configs/predict/conala_noisy_60.yaml
python predict.py configs/predict/conala_noisy_50.yaml
python predict.py configs/predict/conala_noisy_40.yaml
python predict.py configs/predict/conala_noisy_30.yaml
python predict.py configs/predict/conala_noisy_20.yaml
python predict.py configs/predict/conala_noisy_10.yaml