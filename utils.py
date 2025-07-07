from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from models_config import models
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


def get_model(checkpoint_type):
    model_path = models[checkpoint_type]
    model = SentenceTransformer(model_path)
    return model

def load_luar_model_tokenizer(device):
    luar_model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    luar_tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
    luar_model.to(torch.device(device))
    luar_model.eval()
    return luar_model, luar_tokenizer

def get_embedding(model_name, model, texts, tokenizer = None):    
    if model_name == 'luar':
        tokenized_data = tokenizer(
            texts, 
            max_length=512, 
            padding="max_length", 
            return_tensors="pt", 
            truncation=True, 
        )
        tokenized_data.to(model.device)
        input_ids = tokenized_data["input_ids"]
        attention_mask = tokenized_data["attention_mask"]
        input_ids = input_ids.unsqueeze(1).transpose(0,1)
        attention_mask = attention_mask.unsqueeze(1).transpose(0,1)
        embeddings = model(input_ids, attention_mask)
        assert len(embeddings) == 1
        vector = embeddings[0].cpu().detach().numpy()
    elif model_name in models:
        vector = np.mean(model.encode(texts), axis=0)
    else:
        raise ValueError("Invalid checkpoint type")
    
    return normalize(vector.reshape(1, -1))[0]