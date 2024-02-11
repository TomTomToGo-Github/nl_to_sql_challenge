## built-in modules
from pathlib import Path
import json
import os
from pprint import PrettyPrinter
import re
import sqlparse
from src.utils import get_basic_db_info, get_basic_db_info, get_db_schema, get_foreign_keys, get_potential_foreign_keys


# path_data = Path(__file__).parent.parent / 'data'  # when running from the src directory
path_data = Path(os.getcwd()) / 'data'  # when running in console manually
file_ground_truth = path_data / 'examples_queries_test.json'
file_generated_data = path_data / 'generated_queries.json'
with open(file_ground_truth, 'r', encoding='utf-8') as f:
    data_ground_truth = json.load(f)
with open(file_generated_data, 'r', encoding='utf-8') as f:
    data_generated = json.load(f)


def data_exploration(data, data_type: str='generated'):
    info_json = dict()
    info_json['list_unique_questions'] = list(set([query.get('question') for query in data if query.get('question')]))
    info_json['num_items'] = len(data)  # 27 queries with information
    info_json['num_questions'] = len([query.get('question') for query in data if query.get('question')])  # all queries have a question that is not None
    if data_type == 'generated':
        info_json['list_unique_queries'] = list(set([query.get('generated_query') for query in data if query.get('generated_query')]))  # all queries seem unique
        info_json['num_queries'] = len([query.get('generated_query') for query in data if query.get('generated_query')])  # all queries have data
        info_json['num_unique_queries'] = len(info_json['list_unique_queries'])  # all queries seem unique
    else:
        info_json['list_unique_databases'] = list(set([query.get('db_id') for query in data if query.get('db_id')]))  # all queries seem to originate fro one database with db_id = 'TheHistoryofBaseball
        info_json['num_databases'] = len([query.get('db_id') for query in data if query.get('db_id')])  # all queries have a db_id that is not None
        info_json['num_unique_databases'] = list(set([query.get('db_id') for query in data if query.get('db_id')]))  # all queries seem to originate fro one database with db_id = 'TheHistoryofBaseball
        info_json['list_unique_queries'] = list(set([query.get('query') for query in data if query.get('db_id')]))  # all queries seem unique
        info_json['num_queries'] = len([query.get('query') for query in data if query.get('query')])  # all queries have a query that is not None
        info_json['num_unique_queries'] = len(info_json['list_unique_queries'])  # all queries have a query that is not None
    return info_json

data = data_ground_truth
data_type = 'ground_truth'
info_gt = data_exploration(data, data_type)
data = data_generated
data_type = 'generated'
info_gen = data_exploration(data, data_type)


pp = PrettyPrinter(indent=4)
pp.pprint(info_gt)
pp.pprint(info_gen)

def normalize_query_string(query_string: str):
    query_string = query_string.lower()
    query_string = re.sub(r'\s+', r' ', query_string)
    query_string = " ".join([re.sub(r"'", r'"', query_word) for query_word in query_string.split()])
    return query_string

assert len([q for q in info_gt['list_unique_questions'] if q in info_gen['list_unique_questions']]) == 27  # the questions are the same
assert len([q for q in info_gt['list_unique_queries'] if q in info_gen['list_unique_queries']]) == 0  # none of the queries is an exact match
clean_queries_gt = [normalize_query_string(q) for q in info_gt['list_unique_queries']]
clean_queries_gen = [normalize_query_string(q) for q in info_gen['list_unique_queries']]
assert len([q for q in clean_queries_gt if q in clean_queries_gen]) == 3  # more differences than just spacing or newlines

num_equal = 0
## get feeling for the data and what in particular is mismatching
for idx_item, (item_gt, item_gen) in enumerate(zip(data_ground_truth, data_generated)):
    num_equal += 1 if (item_gt.get('question') == item_gen.get('question')) else 0  # the items match in order
    print(f'\n\nQuestion {idx_item + 1}.: ', item_gt.get('question'))
    print('     Ground truth: ', re.sub(r'\s+', r' ', item_gt.get('query')))
    print('     LLMGenerated: ', re.sub(r'\s+', r' ', item_gen.get('generated_query')))



from sql_metadata import Parser
table_names = set()
column_names = set()
for item_gt in data_ground_truth:
    query = item_gt.get('query')
    parsed_query = Parser(query)
    # parsed_query.tokens
    # column_names.update(parsed_query.columns_dict)
    # parsed_query.columns_aliases_names
    # parsed_query.columns_aliases_dict
    table_names.update(set(parsed_query.tables))
    # parsed_query.values
    # parsed_query.limit_and_offset
print(table_names)  ## tables found in the queries
# print(column_names)  ## incorrect parser...


# Display all table names with column names
history_baseball_db_path = Path(r"C:\Users\thoma\Desktop\Tommy\Programmieren\nl_to_sql_challenge\data\history_of_baseball.sqlite")
table_info = get_basic_db_info(history_baseball_db_path)
print('All tables: ', table_info.keys())
for table_name, col_info in table_info.items():
    print(f"\n\nTable: {table_name}")
    print(f"    Columns: {col_info['column_names']}")

# Extract schema and foreign key information
history_baseball_db_path = Path(r"C:\Users\thoma\Desktop\Tommy\Programmieren\nl_to_sql_challenge\data\history_of_baseball.sqlite")
schema_info = get_db_schema(history_baseball_db_path)
# foreign_keys_info = get_foreign_keys(history_baseball_db_path)
# Printing the schema information
print("Schema Information:")
for table, schema in schema_info.items():
    print(f"Table: {table}")
    for column in schema:
        print(f"  Column: {column[1]}, Type: {column[2]}") 
# # Printing the foreign key information  ### KEY INFORMATION WAS NOT AVAILABLE
# print("\nForeign Key Information:")
# for table, fkeys in foreign_keys_info.items():
#     print(f"Table: {table}")
#     for fkey in fkeys:
#         print(f"  Column: {fkey[3]} links to {fkey[2]} on {fkey[0]}")s
potential_foreign_keys = get_potential_foreign_keys(history_baseball_db_path)

# Printing the potential foreign key information
print("Potential Foreign Key Information:")
for table, relations in potential_foreign_keys.items():
    print(f"Table: {table}")
    for column, target_table in relations:
        print(f"  Column: {column} might link to table: {target_table}")
        

## Print exactly matching results
import sqlite3
import pandas as pd
from pathlib import Path
history_baseball_db_path = Path(r"C:\Users\thoma\Desktop\Tommy\Programmieren\nl_to_sql_challenge\data\history_of_baseball.sqlite")
for idx_item, (item_gt, item_gen) in enumerate(zip(data_ground_truth, data_generated)):
    with sqlite3.connect(history_baseball_db_path) as conn:
        query_gt = item_gt.get('query')
        query_gen = item_gen.get('generated_query')
        data_gt = pd.read_sql_query(query_gt, conn)
        try:
            data_gen = pd.read_sql_query(query_gen, conn)
        except Exception as e:
            # print(e)
            data_gen = "Invalid query"
        # print(f'\n\nQuestion {idx_item + 1}.: ', item_gt.get('question'))
        # print('\n     Ground truth:\n', data_gt)
        # print('\n     LLMGenerated\n: ', data_gen)

        if data_gt.equals(data_gen):
            print(f'\nNr. {idx_item + 1}.: Matching results')



# visual inspection:
#     1. query correct, but "COUNT(*)" is extra
#     2. identical
#     3. seems to be correct -> but the extra join is useless I would assume that table player has player_id and can be connected difectly to award, no need for award vote table
#     4. 
#     5. Missing Block WHERE in the end
#     6. 
#     7. Quotes mismatch (single instead of double)
#     8. Quotes mismatch (single instead of double)
#     9. 
#     10. 
#     11. 
#     12. 
#     13. 
#     14. 
#     15. 
#     16. 
#     17. 
#     18. 
#     19. 
#     20. 
#     21. 
#     22. 
#     23. 
#     24. 
#     25. 
#     26. 
#     27. Quotes mismatch (single instead of double)