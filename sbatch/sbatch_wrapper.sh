#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G2
#SBATCH --output output/logs

PYTHONPATH=. PYTHONUNBUFFERED=1 ../pyenv/bin/python $@