from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer
import json
import os

# Import architecture from your sbert_model.py
from sbert_model import SentenceBERT, BertModel, BertConfig

app = Flask(__name__)

# 1. Setup Device and Load Tokenizer
device = torch.device('cpu')
# Path to your tokenizer files
tokenizer = BertTokenizer.from_pretrained('./tokenizer')

# 2. Load Configuration from your config.json
with open('config.json', 'r') as f:
    cfg = json.load(f)

config = BertConfig(
    vocab_size=cfg['vocab_size'],
    hidden_size=cfg['hidden_size'],
    num_hidden_layers=cfg['num_hidden_layers'],
    num_attention_heads=cfg['num_attention_heads'],
    intermediate_size=cfg['intermediate_size'],
    max_position_embeddings=cfg['max_position_embeddings']
)

# 3. Initialize Model and Load Weights
bert_base = BertModel(config)
model = SentenceBERT(bert_base, hidden_size=config.hidden_size)
model.load_state_dict(torch.load('model.pt', map_location=device), strict=False)
model.eval()

# 4. Load Metrics (Task 3 results)
stats = {}
if os.path.exists('metrics.json'):
    with open('metrics.json', 'r') as f:
        stats = json.load(f)
    # Rename key so Jinja2 can read it using dot notation
    if 'macro avg' in stats:
        stats['macro_avg'] = stats['macro avg']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    premise = ""
    hypothesis = ""
    
    # Matching your notebook's mapping: 0=Neutral, 1=Entailment, 2=Contradiction
    labels = {0: 'Neutral', 1: 'Entailment', 2: 'Contradiction'}
    
    if request.method == 'POST':
        premise = request.form['premise']
        hypothesis = request.form['hypothesis']
        
        # --- MANUAL TENSOR CONVERSION TO BYPASS VERSION CHECK ---
        p_enc = tokenizer(premise, padding='max_length', truncation=True, max_length=cfg['max_len'])
        h_enc = tokenizer(hypothesis, padding='max_length', truncation=True, max_length=cfg['max_len'])
        
        p_ids = torch.tensor([p_enc['input_ids']])
        p_mask = torch.tensor([p_enc['attention_mask']])
        h_ids = torch.tensor([h_enc['input_ids']])
        h_mask = torch.tensor([h_enc['attention_mask']])
        
        with torch.no_grad():
            # Siamese forward pass
            logits, _ = model(p_ids, p_mask, h_ids, h_mask)
            
            prediction_idx = torch.argmax(logits, dim=1).item()
            conf = torch.softmax(logits, dim=1)[0][prediction_idx].item()
            
            result = {
                'label': labels.get(prediction_idx, "Unknown"),
                'confidence': f"{conf:.2%}"
            }
            
    return render_template('index.html', 
                           result=result, 
                           stats=stats, 
                           premise=premise, 
                           hypothesis=hypothesis)

if __name__ == '__main__':
    app.run(debug=True)