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
        """
        d_in: input embedding dimension
        d_out: output embedding dimension - d_head
        """
        self.Query_w = nn.Linear(d_in, d_out, bias=False)
        self.Key_w = nn.Linear(d_in, d_out, bias=False)
        self.Value_w = nn.Linear(d_in, d_out, bias=False)
        self.d_out = d_out
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        query = self.Query_w(x) # = x @ Query_w.weight.T + Query_w.bias(none) 
        key = self.Key_w(x) # = x @ Key_w.weight.T + Key_w.bias(none)
        value = self.Value_w(x) # = x @ Value_w.weight.T + Value_w.bias(none)
        
        attention_scores = query @ key.transpose(1, 2) / (self.d_out ** 0.5)
        attention_weights = self.softmax(attention_scores)
        return attention_weights @ value # context vector

class CausalAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout=0.1):
        super(CausalAttention, self).__init__()
        """
        d_in : embedding dimension at the input
        d_out : embedding dimension at the output = head_dims 
        context_length : length of the input sequence
        
        Query_w : Linear layer for of shape (d_in, d_out) 
        
        query : batch_size x sequence_length x d_out
        """
        
        self.query_w = nn.Linear(d_in, d_out, bias=False)
        self.key_w = nn.Linear(d_in, d_out, bias=False)
        self.value_w = nn.Linear(d_in, d_out, bias=False)
        self.d_out = d_out
        self.dropout = nn.Dropout(0.1)
        self.mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    
    def forward(self, x):
        query = self.query_w(x) # same as x @ Query_w.weight.T + Query_w.bias (none)
        key = self.key_w(x) # same as x @ Key_w.weight.T + Key_w.bias (none)
        value = self.value_w(x) # same as x @ Value_w.weight.T + Value_w.bias (none)

        attention_scores = query @ key.transpose(1, 2) / (self.d_out ** 0.5)
        masked_attention_scores = attention_scores.masked_fill(self.mask == 1, float("-inf"))
        attention_weights = torch.nn.functional.softmax(masked_attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return attention_weights @ value # context vector
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.values_w = nn.Linear(d_in, d_out)
        self.keys_w = nn.Linear(d_in, d_out)
        self.queries_w = nn.Linear(d_in, d_out)
        self.fc_out = nn.Linear(d_out, d_out)

    def forward(self, x):
        batch_size, input_length, _ = x.shape

        values = self.values_w(x)
        keys = self.keys_w(x)
        query = self.queries_w(x)

        # (batch_size, sequence_length, d_out) -> (batch_size, sequence_length, num_heads, head_dim)
        keys = keys.view(batch_size, input_length, self.num_heads, self.head_dim)
        query = query.view(batch_size, input_length, self.num_heads, self.head_dim)
        values = values.view(batch_size, input_length, self.num_heads, self.head_dim)


        # (batch_size, sequence_length, num_heads, head_dim) -> (batch_size, num_heads, sequence_length, head_dim)
        keys = keys.transpose(1, 2)
        query = query.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = query @ keys.transpose(2,3)/(self.head_dim**0.5)
        mask = torch.triu(torch.ones(input_length, input_length), diagonal=1).bool()
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ values # batch_size x num_heads x sequence_length x head_dim
        attention_output = attention_output.transpose(1, 2) # batch_size x sequence_length x num_heads x head_dim
        
        context_vector = attention_output.contiguous().view(batch_size, input_length, self.num_heads * self.head_dim) # batch_size x sequence_length x d_out
        return self.fc_out(context_vector)
        
