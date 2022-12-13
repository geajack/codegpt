#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G2

PYTHONUNBUFFERED=1 python train.py
