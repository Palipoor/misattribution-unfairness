import faiss
import numpy as np
import os
import pickle as pkl
from bidict import bidict
from numpy.linalg import norm
import tensorflow_datasets as tfds
from utils import get_embedding, get_model, load_luar_model_tokenizer
from models_config import models
from tqdm import tqdm


def normalize(v):
    norm_v = norm(v)
    if norm_v == 0:
        return v
    return v / norm_v

def get_center(idx, subset = None):
    first = idx.reconstruct(0)
    shape = first.shape
    sum = np.zeros(shape)
    if subset:
        for i in subset:
            vector = idx.reconstruct(i)
            sum += vector
        return normalize(sum / len(subset))
    else:
        num_vectors = idx.ntotal
        for i in range(num_vectors):
            vector = idx.reconstruct(i)
            sum += vector    
        return  normalize(sum / num_vectors)


def get_distances(idx, ref, subset = None):
    distances = []
    if subset:
        for i in subset:
            vector = idx.reconstruct(i)
            distances.append(1 - np.dot(ref, vector))
        return distances
    else:
        num_vectors = idx.ntotal
        for i in range(num_vectors):
            vector = idx.reconstruct(i)
            distances.append(1 - np.dot(ref, vector))
        return distances

def index_candidates_from_dataframe(model_name, device, candidates, dataset, prefix=''):

    name = f'{prefix}_{dataset}_{model_name}'
    if model_name == 'luar':
        model, tokenizer = load_luar_model_tokenizer(device)
        index = faiss.IndexFlatL2(512)
    elif model_name in models:
        model = get_model(model_name)
        tokenizer = None
        model.to(device)
        index = faiss.IndexFlatL2(768)
    else:
        raise ValueError("Invalid Model Name")
    
    candidate_embeddings = []
    candidate_ids = bidict()
    num_candidates = len(candidates)
    for i,row in tqdm(candidates.iterrows(), total=num_candidates, desc="Indexing candidates"):
        texts = row['text']
        embedding = get_embedding(model_name, model, texts, tokenizer)
        candidate_embeddings.append(embedding)
        candidate_ids[row['id']] = i

    candidate_embeddings = np.vstack(candidate_embeddings)
    index.add(candidate_embeddings)
    faiss.write_index(index, f'{name}.index')
    if not os.path.exists(f'{prefix}_{dataset}_candidate_ids.pkl'): # candidate ids do not depend on model
        pkl.dump(candidate_ids, open(f'{prefix}_{dataset}_candidate_ids.pkl', 'wb'))


def index_candidates_from_tfdataset(model_name, device, candidates_path, dataset, prefix=''): # only for reddit dataset
    if dataset != 'reddit':
        raise ValueError("This function is only for the reddit dataset for now.")
    
    candidate_dataset = tfds.load('reddit_user_id', split="test_target", shuffle_files=False, data_dir=candidates_path)
    num_candidates = len(candidate_dataset)
    candidate_rows = list(candidate_dataset.take(num_candidates))
    name = f'{prefix}_{dataset}_{model_name}'
    index = None
    index_name = f'{name}.index'
    candidate_ids = bidict()
    for i, row in enumerate(candidate_rows):
        uid = row['user_id'].numpy().decode('utf8')
        candidate_ids[uid] = i # two-way mapping
    

    if model_name == 'luar':
        model, tokenizer = load_luar_model_tokenizer(device)
        index = faiss.IndexFlatL2(512)
    elif model_name in models:
        model = get_model(model_name)
        tokenizer = None
        model.to(device)
        index = faiss.IndexFlatL2(768)
    else:
        raise ValueError("Invalid Model Name")

    if model is None and tokenizer is None:
        raise ValueError("Model and tokenizer must be provided for embedding generation.")

    candidate_embs = []
    for i, row in enumerate(candidate_rows):
        texts = [t.decode('utf8').strip() for t in row['body'].numpy().tolist()]
        candidate_embs.append(get_embedding(type, model, texts, tokenizer))

    index.add(np.vstack(candidate_embs))
    faiss.write_index(index, index_name)
    pkl.dump(candidate_ids, open(f'{prefix}_{dataset}_candidate_ids.pkl', 'wb'))

