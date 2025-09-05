# -*- coding: utf-8 -*-
import torch.nn as nn
from transformers import BertModel

class SemanticMatchingModel(nn.Module):
    """
    BERT-based model for semantic matching.
    """
    def __init__(self, pre_trained_model_name):
        super(SemanticMatchingModel, self).__init__()
        self.bert = BertModel.from_pretrained(pre_trained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # The last_hidden_state is not used here, we use the pooler_output
        # which is the output of the [CLS] token after a Linear layer and Tanh activation.
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        
        output = self.drop(pooled_output)
        return self.out(output)
