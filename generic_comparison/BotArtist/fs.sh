#!/bin/bash

# List of dataset names
datasets=("botometer-feedback-2019" "cresci-2017" "cresci-stock-2018" "midterm-2018" "cresci-2015" "cresci-rtbust-2019" "gilani-2017" "Twibot-20" "Twibot-22")

# Loop over each dataset name
for dataset in "${datasets[@]}"
do
    # Run the Python script with the current dataset name
    python3 feature_extractor.py --dataset $dataset
    mv $dataset* ../$dataset/
done
