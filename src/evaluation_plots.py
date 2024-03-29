## built-in modules
import os
import json
from pathlib import Path
## pip installed modules
import pandas as pd
import sqlite3
## self-made modules
from src.utils import compare_sql_queries_by_block, compare_sql_queries_by_results, barplot_queries_evaluation_by_score, compare_block_structures, barplot_queries_evaluation_by_llm


## 0. Load data
history_baseball_db_path = Path(r"C:\Users\thoma\Desktop\Tommy\Programmieren\nl_to_sql_challenge\data\history_of_baseball.sqlite")
path_data = Path(os.getcwd()) / 'data'  # when running in console manually
file_ground_truth = path_data / 'examples_queries_test.json'
file_generated_data = path_data / 'generated_queries.json'
with open(file_ground_truth, 'r', encoding='utf-8') as f:
    data_ground_truth = json.load(f)
with open(file_generated_data, 'r', encoding='utf-8') as f:
    data_generated = json.load(f)
path_results = Path(r"C:\Users\thoma\Desktop\Tommy\Programmieren\nl_to_sql_challenge\results")
# file_results = Path("results_prompt_template_1_no_info.json")
# file_results = Path("results_prompt_template_schema_links_only.json")
file_results = Path("results_prompt_template_schema.json")
with open(path_results / file_results, 'r', encoding='utf-8') as f:
    data_results = json.load(f)



## 1. Compare given generated queries with the ground truth
scores_by_llm_base = {}
scores_structure_comparison = []
scores_block_comparison = []
scores_result_comparison = []
for idx_item, (item_gt, item_gen) in enumerate(zip(data_ground_truth, data_generated)):
    ground_truth = item_gt.get('query')
    generated = item_gen.get('generated_query')
    ## compute scores
    score_structure = compare_block_structures(ground_truth, generated)
    score_blocks, blocks_info = compare_sql_queries_by_block(ground_truth, generated)
    score_results = compare_sql_queries_by_results(ground_truth, generated, history_baseball_db_path)
    ## save scores
    scores_block_comparison.append(score_blocks)
    scores_structure_comparison.append(score_structure)
    scores_result_comparison.append(score_results)
scores_by_llm_base['Generated vs. Ground Truth'] = {
    'structure': scores_structure_comparison,
    'blocks': scores_block_comparison,
    'results': scores_result_comparison
}
## Plot the results
barplot_queries_evaluation_by_llm(scores_by_llm_base)
barplot_queries_evaluation_by_score(scores_by_llm_base)



## 2. Evaluate the real results generated by current approach
# file_results = Path("results_prompt_template_1_no_info.json")
# file_results = Path("results_prompt_template_schema_links_only.json")
file_results = Path("results_prompt_template_schema.json")
with open(path_results / file_results, 'r', encoding='utf-8') as f:
    data_results = json.load(f)

scores_by_llm = {}
llm_selection = ["openai_turbo", "mistral_medium"]
# llm_selection = ["openai_turbo", "mistral_small", "mistral_medium", "sql_coder"]
for llm_type in llm_selection:
    ## SQL query comparison by blocks
    scores_structure_comparison = []
    scores_block_comparison = []
    scores_result_comparison = []
    for item_results in data_results:
        ground_truth = item_results.get('Ground truth', '')
        generated = item_results.get('Generated', {}).get(llm_type, '')
        ## compute scores
        score_structure = compare_block_structures(ground_truth, generated)
        score_blocks, blocks_info = compare_sql_queries_by_block(ground_truth, generated)
        score_results = compare_sql_queries_by_results(ground_truth, generated, history_baseball_db_path)
        ## save scores
        scores_structure_comparison.append(score_structure)
        scores_block_comparison.append(score_blocks)
        scores_result_comparison.append(score_results)
    scores_by_llm[llm_type] = {
        'structure': scores_structure_comparison,
        'blocks': scores_block_comparison,
        'results': scores_result_comparison
    }
scores_by_llm['Generated vs. Ground Truth'] = scores_by_llm_base['Generated vs. Ground Truth']
## Plot the results
barplot_queries_evaluation_by_llm(scores_by_llm)
barplot_queries_evaluation_by_score(scores_by_llm)
