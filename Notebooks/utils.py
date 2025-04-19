import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM


class CustomDataset(Dataset):
    def __init__(self, text,tokenizer,max_length,stride):
        self.tokenizer = tokenizer
        self.text = text
        self.max_length = max_length
        self.stride = stride
        
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_ids = token_ids[i:i + max_length]
            target_ids = token_ids[i + 1:i + max_length + 1]
            
            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.input_ids[idx], dtype=torch.long)
        y = torch.tensor(self.target_ids[idx], dtype=torch.long)
        return x, y


class SelfAttention(nn.Module):
    def __init__(self, d_in,d_out):
        super(SelfAttention, self).__init__()
        self.Query_w = nn.Linear(d_in, d_out, bias=False)
        self.Key_w = nn.Linear(d_in, d_out, bias=False)
        self.Value_w = nn.Linear(d_in, d_out, bias=False)
        self.d_out = d_out
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        query = x @ self.Query_w.weight
        key = x @ self.Key_w.weight
        value = x @ self.Value_w.weight
        
        attention_scores = query @ key.transpose(1, 2) / (self.d_out ** 0.5)
        attention_weights = self.softmax(attention_scores)
        return attention_weights @ value # context vectors


class CausalAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout=0.1):
        super(CausalAttention, self).__init__()
        self.query_w = nn.Linear(d_in, d_out, bias=False)
        self.key_w = nn.Linear(d_in, d_out, bias=False)
        self.value_w = nn.Linear(d_in, d_out, bias=False)
        self.d_out = d_out
        self.dropout = nn.Dropout(0.1)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    
    def forward(self, x):
        query = x @ self.query_w.weight
        key = x @ self.key_w.weight
        value = x @ self.value_w.weight

        attention_scores = query @ key.transpose(1, 2) / (self.d_out ** 0.5)
        masked_attention_scores = attention_scores.masked_fill(self.mask == 1, float("-inf"))
        attention_weights = torch.nn.functional.softmax(masked_attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ value # context vector
