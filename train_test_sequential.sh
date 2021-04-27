#!/bin/bash
rm -r ./configs
python nilm_thresholding/generate_config_files.py
FILES=./configs/*
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  if python nilm_thresholding/train_model.py --path-config $f; then
    if python nilm_thresholding/test_model.py --path-config $f; then
      echo "SUCCESS"
    else
      break
    fi
  else
    break
  fi
done
