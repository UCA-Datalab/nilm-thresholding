#!/bin/bash
python better_nilm/generate_config_files.py
FILES=./configs/*
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  python better_nilm/train_test_model.py --path-config $f
done