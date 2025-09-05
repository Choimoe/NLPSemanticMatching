#!/bin/bash

# This script is the main entry point for generating predictions.
# It assumes the current working directory is 'code/'.

echo "Starting prediction..."

# Set the python path to include the project root, so imports work correctly
export PYTHONPATH=$(dirname "$PWD")

# Run the prediction script
python3 predict.py

echo "Prediction finished. Results are in prediction_result/result.tsv"
