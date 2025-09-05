# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

import config
from dataset import SemanticMatchingDataset
from model import SemanticMatchingModel
from utils import set_seed, setup_logger

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    
    for batch in tqdm(data_loader, total=len(data_loader), desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    final_targets = []
    final_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            targets = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            final_targets.extend(targets.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            
    return roc_auc_score(final_targets, final_outputs)


def run():
    set_seed(config.RANDOM_SEED)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(config.MODEL_OUTPUT_PATH), exist_ok=True)
    
    logger = setup_logger("training_logger", "training.log")

    df = pd.read_csv(config.TRAIN_PATH, sep='\t', names=['q1', 'q2', 'label']).fillna("0")
    
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

    # For demonstration, we will train on a single fold.
    # For a robust solution, consider training on all folds (Cross-Validation).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    train_idx, val_idx = next(iter(skf.split(df, df.label)))
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    train_dataset = SemanticMatchingDataset(
        texts_a=df_train.q1.values,
        texts_b=df_train.q2.values,
        labels=df_train.label.values,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    train_data_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    
    val_dataset = SemanticMatchingDataset(
        texts_a=df_val.q1.values,
        texts_b=df_val.q2.values,
        labels=df_val.label.values,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN
    )

    val_data_loader = DataLoader(val_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)

    device = config.DEVICE
    model = SemanticMatchingModel(config.PRE_TRAINED_MODEL_NAME)
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=config.LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    
    best_auc = 0
    for epoch in range(config.EPOCHS):
        logger.info(f"--- Epoch {epoch+1}/{config.EPOCHS} ---")
        train_one_epoch(model, train_data_loader, optimizer, scheduler, device)
        auc = evaluate(model, val_data_loader, device)
        logger.info(f"Validation AUC = {auc}")
        
        if auc > best_auc:
            torch.save(model.state_dict(), config.MODEL_OUTPUT_PATH)
            best_auc = auc
            logger.info(f"Best model saved with AUC: {best_auc}")

if __name__ == "__main__":
    run()

