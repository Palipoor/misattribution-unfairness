from index_utils import *
from utils import *
import os
import click
import pandas as pd
from models_config import models


@click.command()
@click.option('--model_name', '-m', default='luar', type=click.Choice(list(models.keys()) + ['luar']), help="Embedding model name")
@click.option('--dataset', '-d', default='reddit', type=click.Choice(['reddit', 'blogs', 'fanfiction']), help="Dataset")
@click.option('--candidates_path', '-c', type=str, help="Path to candidates (haystack) file")
@click.option('--prefix', '-p', default='', type=str, help="Prefix (to distinguish between different runs)")
def main(model_name, dataset, candidates_path, prefix):
    index_path = f'{prefix}_{dataset}_{model_name}.index'
    if os.path.exists(index_path):
        print("Index already exists")
    else:
        print("Indexing candidates...")
        if 'jsonl' in candidates_path:
            print("Reading candidates from JSONL file")
            candidates = pd.read_json(candidates_path, lines=True)
            index_candidates_from_dataframe(model_name, 'cuda', candidates, dataset, prefix)
        else:
            print('Reading candidates from tensorflow dataset')
            index_candidates_from_tfdataset(model_name, 'cuda', candidates_path, dataset, prefix)
        print("Indexing complete, index saved to", index_path)

if __name__ == '__main__':
    main()
