#!/bin/bash

#SBATCH --mem 500
#SBATCH --partition LARGE-G2

python -m http.server 5000