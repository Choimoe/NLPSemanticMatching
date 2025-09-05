# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch

class SemanticMatchingDataset(Dataset):
    """
    Custom PyTorch Dataset for the semantic matching task.
    """
    def __init__(self, texts_a, texts_b, labels, tokenizer, max_len):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_a)

    def __getitem__(self, item):
        text_a = str(self.texts_a[item])
        text_b = str(self.texts_b[item])

        encoding = self.tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        if self.labels is not None:
            output['labels'] = torch.tensor(self.labels[item], dtype=torch.float)

        return output
