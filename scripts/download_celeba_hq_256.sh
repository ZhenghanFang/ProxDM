#!/bin/bash
set -e

# pip install kaggle
# put kaggle.json in .kaggle directory

export KAGGLE_CONFIG_DIR=".kaggle"

DATA_DIR="data/celeba_hq"
mkdir -p $DATA_DIR

kaggle datasets download badasstechie/celebahq-resized-256x256 -p $DATA_DIR
python -m zipfile -e $DATA_DIR/celebahq-resized-256x256.zip $DATA_DIR
