# -*- coding: utf-8 -*-
import torch
import os

# --- Device Configuration ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- Path Configuration ---
DATA_DIR = "./tcdata/oppo_breeno_round1_data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.tsv")
TEST_PATH = os.path.join(DATA_DIR, "testB.tsv")

USER_DATA_DIR = "user_data"
MODEL_OUTPUT_PATH = os.path.join(USER_DATA_DIR, "model_data/bert_semantic_matching.bin")

PREDICTION_DIR = "prediction_result"
PREDICTION_PATH = os.path.join(PREDICTION_DIR, "result.tsv")

# --- Model & Tokenizer Configuration ---
PRE_TRAINED_MODEL_NAME = "bert-base-chinese"
MAX_LEN = 128 # Max sequence length for BERT

# --- Training Hyperparameters ---
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
