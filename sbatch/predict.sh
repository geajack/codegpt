#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G2
#SBATCH --output output/slurm-latest

PYTHONUNBUFFERED=1 python predict.py $1