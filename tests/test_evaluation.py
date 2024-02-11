## built-in modules
from pathlib import Path
import json
import os


# path_data = Path(__file__).parent / 'data'  # when running from the root directory
path_data = Path(os.getcwd()) / 'data'  # when running in console manually
file_ground_truth = path_data / 'examples_queries_test.json'
file_generated_data = path_data / 'generated_queries.json'
with open(file_ground_truth, 'r', encoding='utf-8') as f:
    data_ground_truth = json.load(f)
with open(file_generated_data, 'r', encoding='utf-8') as f:
    data_generated = json.load(f)


def test_compare_data():
    for key in data_ground_truth:
        assert data_ground_truth[key] == data_generated[key]
        
        
        
    # assert data_ground_truth == data_generated