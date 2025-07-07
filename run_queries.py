import numpy as np
import faiss
import pickle as pkl
import pandas as pd
import click
from tqdm import tqdm
from bidict import bidict
from utils import get_embedding, get_model, load_luar_model_tokenizer
from models_config import models
import tensorflow_datasets as tfds


def search_faiss_index(query_emb, faiss_idx, candidate_user_ids, query_user_id):
    _, I = faiss_idx.search(np.expand_dims(query_emb, 0), k=len(candidate_user_ids))
    I = I[0]
    user_ids_ranked = list(map(lambda idx: candidate_user_ids.inv[idx], I))
    try:
        query_user_rank = user_ids_ranked.index(query_user_id) + 1
    except:
        breakpoint()

    recall_1 = int(query_user_rank <= 1)
    recall_8 = int(query_user_rank <= 8)
    reciprocal_rank = 1.0/query_user_rank
    return {'recall_1': recall_1, 'recall_8': recall_8, 'reciprocal_rank': reciprocal_rank, 'user_ids_ranked': user_ids_ranked, 'query_user_rank': query_user_rank, 'ranks': I}



def run_queries_and_evaluate_from_dataframe(model_name, device, dataset, queries, prefix=''):
    candidates_index_name = f'{prefix}_{dataset}_{model_name}'
    index = faiss.read_index(f'{candidates_index_name}.index')
    candidate_ids = bidict(pkl.load(open(f'{prefix}_{dataset}_candidate_ids.pkl', 'rb')))
    if model_name == 'luar':
        model, tokenizer = load_luar_model_tokenizer(device)
    elif model_name in models:
        model = get_model(model_name)
        tokenizer = None
        model.to(device)
    else:
        raise ValueError("Invalid Model Name")

    recall_1 = {}
    recall_8 = {}
    mrrs = []
    query_ids = bidict()
    shape = (len(queries),len(candidate_ids))

    # /!\ warning: big file - we use memmap to read it from disk and not load it all into memory
    filename = f'{prefix}_{model_name}_ranks.dat' # This file is used to store the ranks of candidates for each query and later calculate misattribution unfairness and other metrics.
    dtype = np.int32
    cand_ranks = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    for i,row in tqdm(queries.iterrows(), total = len(queries)):
        if i % 100 == 0:
            cand_ranks.flush()
        texts = row['text']
        query_user_id = row['id']
        embedding = get_embedding(model_name, model, texts, tokenizer)
        res = search_faiss_index(embedding, index, candidate_ids, query_user_id)
        recall_1[query_user_id] = res['recall_1']
        recall_8[query_user_id] = res['recall_8']
        for r,user in enumerate(res['ranks']):
            cand_ranks[i][user] = r
        mrrs.append(res['reciprocal_rank'])
        query_ids[row['id']] = i
    
    cand_ranks.flush()
    del cand_ranks
    recall_1s = np.mean(list(recall_1.values()))
    recall_8s = np.mean(list(recall_8.values()))
    mrr = np.mean(mrrs)
    print(f"{prefix}_{dataset}_{model_name}: ")
    print(f"Recall@1: {np.mean(recall_1s)}")
    print(f"Recall@8: {np.mean(recall_8s)}")
    print(f"MRR: {mrr}")
    pkl.dump(query_ids, open(f'{prefix}_{dataset}_query_ids.pkl', 'wb')) # save for later - these ids match the rows and columns of the cand_ranks file


def run_queries_and_evaluate_from_tfdataset(model_name, device, dataset, queries_path, prefix=''):
    if dataset != 'reddit':
        raise ValueError("This function is only for the reddit dataset for now.")
    
    queries = tfds.load('reddit_user_id', split="test_query", shuffle_files=False, data_dir=queries_path)
    num_queries = len(queries)
    query_rows = list(queries.take(num_queries))
    candidates_index_name = f'{prefix}_{dataset}_{model_name}'
    index = faiss.read_index(f'{candidates_index_name}.index')
    candidate_ids = bidict(pkl.load(open(f'{prefix}_{dataset}_candidate_ids.pkl', 'rb')))
    if model_name == 'luar':
        model, tokenizer = load_luar_model_tokenizer(device)
    elif model_name in models:
        model = get_model(model_name)
        tokenizer = None
        model.to(device)
    else:
        raise ValueError("Invalid Model Name")

    recall_1 = {}
    recall_8 = {}
    mrrs = []
    query_ids = bidict()
    shape = (len(queries),len(candidate_ids))
    filename = f'{prefix}_{model_name}_ranks.dat' # This file is used to store the ranks of candidates for each query and later calculate misattribution unfairness and other metrics.
    dtype = np.int32
    cand_ranks = np.memmap(filename, dtype=dtype, mode='w+', shape=shape)
    for i, row in enumerate(query_rows):
        if i % 100 == 0:
            cand_ranks.flush()
        uid = row['user_id'].numpy().decode('utf8')
        query_user_id = uid
        texts = [t.decode('utf8').strip() for t in row['body'].numpy().tolist()]
        q_embedding = get_embedding(model_name, model, texts, tokenizer)
        res = search_faiss_index(q_embedding, index, candidate_ids, query_user_id)
        recall_1[query_user_id] = res['recall_1']
        recall_8[query_user_id] = res['recall_8']
        for r,user in enumerate(res['ranks']):
            cand_ranks[i][user] = r
        mrrs.append(res['reciprocal_rank'])
        query_ids[uid] = i
    
    cand_ranks.flush() # write to disk
    del cand_ranks # free memory


@click.command()
@click.option('-m', '--model_name', type=str, default='luar')
@click.option('-d', '--dataset', type=click.Choice(['reddit', 'blogs', 'fanfiction']), default='reddit')
@click.option('-q', '--queries_path', type=str, help="Path to queries file")
@click.option('-p', '--prefix', type=str, default='')
def main(model_name, dataset, queries_path, prefix):
    if 'jsonl' in queries_path:
        queries = pd.read_json(queries_path, lines=True)
        run_queries_and_evaluate_from_dataframe(model_name, 'cuda', dataset, queries, prefix)
    else:
        run_queries_and_evaluate_from_tfdataset(model_name, 'cuda', dataset, queries_path, prefix)
    print("Evaluation complete")

if __name__ == '__main__':
    main()