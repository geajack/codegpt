#!/bin/bash

export node=147.252.6.52
sbatch sbatch/sbatch_wrapper.sh $@
ssh -N -L 5678:localhost:5000 $node