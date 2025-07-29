# Quantifying Misattribution Unfairness in Authorship Attribution
Repository for the paper "Quantifying Misattribution Unfairness in Authorship Attribution" - ACL 2025 


- [paper on arxiv](https://arxiv.org/abs/2506.02321)
- [paper on ACL anthology](https://aclanthology.org/2025.acl-short.80/)


## Running the code

1. Clone the repository

2. Install the requirements from `requirements.txt`

3. Download datasets from 

4. For each dataset + model, first create the index. This index stores the embeddings of the candidates. 
   ```bash
   python create_index.py --dataset <dataset_name> --model_name <model_name> --candidates_path <path_to_candidates_file> --prefix <run-prefix>
   ```

5. To run the queries and evaluate (Recall@k and MRR), as well as to save the rankings for further analysis, run:
   ```bash
   python run_queries.py --dataset <dataset_name> --model_name <model_name> --prefix <run-prefix> --queries_path <path_to_queries_file>
   ```  




Please reach out if you have any questions or issues running the code. 
