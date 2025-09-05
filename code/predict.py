# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

import config
from dataset import SemanticMatchingDataset
from model import SemanticMatchingModel
from utils import set_seed

def generate_predictions(model, data_loader, device):
    model.eval()
    final_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            
    return final_outputs

def run():
    set_seed(config.RANDOM_SEED)

    # Ensure output directories exist
    os.makedirs(config.PREDICTION_DIR, exist_ok=True)

    df_test = pd.read_csv(config.TEST_PATH, sep='\t', names=['q1', 'q2']).fillna("0")
    
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    
    test_dataset = SemanticMatchingDataset(
        texts_a=df_test.q1.values,
        texts_b=df_test.q2.values,
        labels=None,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )
    
    test_data_loader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=False)
    
    device = config.DEVICE
    model = SemanticMatchingModel(config.PRE_TRAINED_MODEL_NAME)
    model.load_state_dict(torch.load(config.MODEL_OUTPUT_PATH, map_location=device))
    model.to(device)
    
    predictions = generate_predictions(model, test_data_loader, device)
    
    # Flatten the predictions list
    predictions = [item for sublist in predictions for item in sublist]
    
    # Save predictions to the required format
    submission = pd.DataFrame(predictions)
    submission.to_csv(config.PREDICTION_PATH, header=False, index=False)
    print(f"Predictions saved to {config.PREDICTION_PATH}")

if __name__ == "__main__":
    run()
