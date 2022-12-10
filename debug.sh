#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G2

ssh -N -f -R 5000:localhost:5000
python -m http.server 5000