#!/bin/bash
rm -r ./configs/
python nilm-thresholding/generate_config_files.py
FILES=./configs/*
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  python nilm-thresholding/test_model.py --path-config $f --save-scores  --no-save-predictions
done
