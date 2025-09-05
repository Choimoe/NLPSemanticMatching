#!/bin/bash

# This script is used to train the model.
# It assumes the current working directory is 'code/'.

echo "Starting model training..."

# Set the python path to include the project root, so imports work correctly
export PYTHONPATH=$(dirname "$PWD")
code_path="$PWD/code/train.py"

python3 $code_path

echo "Training finished."
