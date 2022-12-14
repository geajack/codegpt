#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G1
#SBATCH --output output/slurm-logs

PYTHONUNBUFFERED=1 python train.py $1
