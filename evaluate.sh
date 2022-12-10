#!/bin/bash

#SBATCH --mem 64000
#SBATCH --partition LARGE-G2

export CUDA_VISIBLE_DEVICES=0
LANG=java
DATADIR=../dataset/concode
OUTPUTDIR=../save/concode
PRETRAINDIR=microsoft/CodeGPT-small-java-adaptedGPT2
LOGFILE=text2code_concode_eval.log

python -u run.py \
        --data_dir=$DATADIR \
        --langs=$LANG \
        --output_dir=$OUTPUTDIR \
        --pretrain_dir=$PRETRAINDIR \
        --log_file=$LOGFILE \
        --model_type=gpt2 \
        --block_size=512 \
        --do_infer \
        --logging_steps=100 \
        --seed=42
