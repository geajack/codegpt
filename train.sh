#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G2

python train.py
