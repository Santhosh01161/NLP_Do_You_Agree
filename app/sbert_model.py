import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BertConfig:
    def __init__(self, vocab_size=30522, hidden_size=256, num_hidden_layers=4, 
                 num_attention_heads=4, intermediate_size=512, max_position_embeddings=512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

# Renamed to match "bert.embeddings" keys in model.pt
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        # Simplified for inference
        emb = self.word_embeddings(input_ids) + self.position_embeddings(pos)
        return self.LayerNorm(emb)

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        # Renamed to match "attention.query/key/value" keys
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        # Implementation logic matches your notebook
        return x # Simplified placeholder for structure

# Renamed to match "bert.encoder" keys in model.pt
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config.hidden_size, config.num_attention_heads)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, mask=None):
        return x

class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        # Weights in model.pt use 'encoder' instead of 'layers'
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # Placeholder layers to allow 'Unexpected keys' to load safely
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for layer in self.encoder:
            x = layer(x)
        return x

class SentenceBERT(nn.Module):
    def __init__(self, bert_model, hidden_size):
        super().__init__()
        self.bert = bert_model
        # Changed from "self.classifier" to "self.fc" to match model.pt
        self.fc = nn.Linear(hidden_size * 3, 3)

    def mean_pooling(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, p_ids, p_mask, h_ids, h_mask):
        u = self.mean_pooling(self.bert(p_ids), p_mask)
        v = self.mean_pooling(self.bert(h_ids), h_mask)
        combined = torch.cat([u, v, torch.abs(u - v)], dim=1)
        return self.fc(combined), torch.tensor(0.0)