#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition MEDIUM-G2
#SBATCH --output output/logs

ifconfig
PYTHONPATH=. PYTHONUNBUFFERED=1 python -m debugpy --listen 5000 --wait-for-client $@