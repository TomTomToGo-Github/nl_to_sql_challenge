import re
# import random
## pip installed modules
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt


## Compare the query structure ("blocks") of the ground truth and generated queries
def normalize_query_string(query_string: str):
    query_string = preserve_string_case_conversion(query_string)  # Convert to lower case
    query_string = re.sub(r'\s+', r' ', query_string)  # Remove extra spaces, newlines, etc.
    query_string = re.sub(r'\\', r'', query_string)  # Remove backslashes (mistral output...)
    query_string = " ".join([re.sub(r"'", r'"', query_word) for query_word in query_string.split()])  # Replace single quotes with double quotes
    query_string = re.sub(r"^.*?(?=select)", r"", query_string)  # Delete explanations before the first SELECT keyword
    return query_string

# def parse_sql_blocks(query_string: str):
#     keywords = ['select', 'from', 'where', 'join', 'group by', 'order by', 'limit']  ## normalized to lower case
#     blocks = {}
#     current_keyword = None
#     for word in re.split(r'(\b(?:' + '|'.join(keywords) + r')\b)', query_string, flags=re.IGNORECASE):
#         if word.lower() in keywords:
#             current_keyword = word.lower()
#             blocks[current_keyword] = []
#         elif current_keyword:
#             blocks[current_keyword].append(word)
#     for key, values in blocks.items():
#         blocks[key] = ' '.join(values)
#     return blocks

def parse_sql_blocks(query_string):
    keywords = ['select', 'from', 'where', 'join', 'group by', 'order by', 'limit']  # normalized to lower case
    blocks = []
    current_keyword = None
    current_block = []

    for word in re.split(r'(\b(?:' + '|'.join(keywords) + r')\b)', query_string, flags=re.IGNORECASE):
        if word.lower() in keywords:
            # Save the previous block before starting a new one
            if current_keyword is not None and current_block:
                blocks.append({current_keyword: ' '.join(current_block).strip()})
                current_block = []  # Reset current block for the next keyword
            current_keyword = word.lower()
        elif current_keyword:
            current_block.append(word)

    # After the loop, save the last block if it's not empty
    if current_keyword is not None and current_block:
        blocks.append({current_keyword: ' '.join(current_block).strip()})

    return blocks


def compare_block_structures(query1: str, query2: str):
    # query1 = "SELECT player_id FROM salary WHERE year = \"2015\" ORDER BY salary DESC LIMIT 1"
    # query2 = "select player.name_first, player.name_last from player inner join salary on player.player_id = salary.player_id where salary.year = 2015 order by salary.salary desc limit 1;"
    blocks1 = parse_sql_blocks(query1)
    blocks2 = parse_sql_blocks(query2)
    block_order1 = [list(block.keys())[0] for block in blocks1]
    block_order2 = [list(block.keys())[0] for block in blocks2]
    
    # Count matching block types from the start
    # Count matching block types from the start
    matching_blocks_count = 0
    for idx, block in enumerate(block_order1):
        if idx < len(block_order2) and block == block_order2[idx]:
            matching_blocks_count += 1
        else:
            break  # Stop counting as soon as a non-matching block is encountered
    
    # Count matching block types from the end only if all blocks from the start do not match
    if matching_blocks_count < len(block_order1):
        if matching_blocks_count < len(block_order1):
            for idx, block in enumerate(reversed(block_order1)):
                if idx < len(block_order2) and block == block_order2[len(block_order2) - 1 - idx]:
                    matching_blocks_count += 1
                else:
                    break  # Stop counting as soon as a non-matching block is encountered
    
    # Calculate the similarity score
    total_blocks = max(len(block_order1), len(block_order2))
    if total_blocks == 0:  # Prevent division by zero
        similarity_score = 0
    else:
        similarity_score = matching_blocks_count / total_blocks
    
    return similarity_score

def preserve_string_case_conversion(query):
    # Pattern to match strings enclosed in quotes
    pattern = r'\"[^\"]*\"'
    
    # Find all matches and replace them with placeholders
    matches = re.findall(pattern, query)
    placeholders = [f"##PLACEHOLDER{index}##" for index, _ in enumerate(matches)]
    temp_query = query
    for placeholder, match in zip(placeholders, matches):
        temp_query = temp_query.replace(match, placeholder, 1)
    
    # Convert to lowercase
    lower_query = temp_query.lower()
    
    # Replace placeholders with original matches
    final_query = lower_query
    for placeholder, match in zip(placeholders, matches):
        final_query = final_query.replace(placeholder.lower(), match, 1)
    
    return final_query

def word_appearance_score(string1, string2):
    string1 = normalize_query_string(string1)
    string2 = normalize_query_string(string2)
    words1 = string1.split()
    words2 = string2.split()
    num_words1 = len(words1)
    num_words2 = len(words2)
    common_words = 0
    for word1 in words1:
        # print(word1, words2)
        if word1 in words2:
            common_words += 1
            words2.remove(word1)
    score = common_words / max(num_words1, num_words2)
    return score

def compare_sql_queries_by_block(ground_truth, generated):
    # Question 5.:  What are the salaries of players who have ever enter hall of fame?
    # ground_truth = """SELECT T2.salary FROM salary as T2 JOIN hall_of_fame as T1 ON T1.player_id = T2.player_id WHERE T1.inducted = "Y" """
    # generated =""" SELECT T2.salary FROM salary as T2 JOIN hall_of_fame as T1 ON T1.player_id = T2.player_id """


    # Question 6.:  What are the minimum votes needed to enter hall of fame for each year since 1871?
    # ground_truth = """SELECT min(votes), yearid FROM hall_of_fame WHERE inducted = "Y" AND yearid >= 1871 GROUP BY yearid"""
    # generated =""" SELECT yearid, MIN(needed) FROM hall_of_fame WHERE yearid >= 1871 GROUP BY yearid """


    # Question 7.:  What are the salaries in National League?
    #    ground_truth = """SELECT salary FROM salary WHERE league_id = "NL" """
    #    generated =""" SELECT salary FROM salary WHERE league_id = 'NL' """

    all_blocks_gt = parse_sql_blocks(normalize_query_string(ground_truth))
    all_blocks_gen = parse_sql_blocks(normalize_query_string(generated))
    comparison = {'identical': [], 'different': []}
    scores = []
    for bl_gt_idx, block_gt in enumerate(all_blocks_gt):
        for block_gt_key, block_gt_value in block_gt.items():
            if bl_gt_idx < len(all_blocks_gen):
                block_gen_value = all_blocks_gen[bl_gt_idx].get(block_gt_key, '')
                if block_gt_value == block_gen_value:
                    comparison['identical'].append(block_gt_key)
                    scores.append(1)
                else:
                    comparison['different'].append(block_gt_key)
                    scores.append(word_appearance_score(block_gt_value, block_gen_value))
            else:
                comparison['different'].append(block_gt_key)
                scores.append(0)
                
    average_score = np.mean(scores) if scores else 0  # Handle the case where there are no columns
    return average_score, comparison


## Compare results column by column an compute one score
def sequence_comparison_score(seq, target):
    """
    Check if a sequence is contained in a target sequence and calculate the score.
    This function is now generalized to handle both numeric and string sequences.
    """
    seq_len = len(seq)
    target_len = len(target)
    # Generalized comparison that works for both numeric and string sequences
    for start_index in range(target_len - seq_len + 1):
        if np.array_equal(target[start_index:start_index + seq_len], seq):
            return seq_len / target_len
    return 0

def compare_dataframe_columns(ground_truth_df, generated_df):
    scores = []
    for gt_col in ground_truth_df.columns:
        col_scores = []
        gt_data = ground_truth_df[gt_col].values
        for gen_col in generated_df.columns:
            gen_data = generated_df[gen_col].values
            # Check data types and proceed accordingly
            if gt_data.dtype.kind in 'iufc' and gen_data.dtype.kind in 'iufc':
                # Both columns are numeric
                col_score = sequence_comparison_score(gt_data, gen_data)
            elif gt_data.dtype.kind in 'O' and gen_data.dtype.kind in 'O':
                # Both columns are objects (typically strings in pandas)
                col_score = sequence_comparison_score(gt_data, gen_data)
            else:
                # Mixed data types, assign a score of 0
                col_score = 0
            col_scores.append(col_score)
        # Take the maximum score for the current ground truth column against all generated columns
        scores.append(max(col_scores) if col_scores else 0)  # Handle the case where there are no columns
    zero_count = scores.count(0)
    if generated_df.shape[1] > ground_truth_df.shape[1]:
        scores.extend([0] * (generated_df.shape[1] - ground_truth_df.shape[1]))
    elif generated_df.shape[1] < ground_truth_df.shape[1]:
        zero_count < ( generated_df.shape[1] - ground_truth_df.shape[1])
        scores.extend([0] * (max(0, ground_truth_df.shape[1] - generated_df.shape[1] - zero_count)))
    # Compute the average score if there's more than one column in ground_truth_df
    average_score = np.mean(scores) if scores else 0  # Handle the case where there are no columns
    return average_score

def compare_sql_queries_by_results(query_gt, query_gen, db_path):
    with sqlite3.connect(db_path) as conn:
        ground_truth_df = pd.read_sql_query(query_gt, conn)
        try:
            generated_df = pd.read_sql_query(query_gen, conn)
        except Exception as e:
            generated_df = pd.DataFrame()
        score_result = compare_dataframe_columns(ground_truth_df, generated_df)
    return score_result


## SQL generation / database exploration
def get_basic_db_info(database_path):

    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Get the table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()

    # Create a dictionary to store the table information
    table_info = {}

    # Iterate over the table names
    for table_name in table_names:
        table_name = table_name[0]  # Extract the table name from the tuple

        # Get the column names and types for the table
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        # Extract the column names and types
        column_names = [column[1] for column in columns]
        column_types = [column[2] for column in columns]

        # Store the column names and types in the table information dictionary
        table_info[table_name] = {
            "column_names": column_names,
            "column_types": column_types
        }

    # Close the database connection
    conn.close()

    return table_info

def get_db_schema(database_path, table_selection=None):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Store schema information
    schema_info = {}
    schema_text = {}
    
    # Fetch schema for each table
    for table in [table for table in tables if table[0] in table_selection] if table_selection else tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema_info[table_name] = cursor.fetchall()
    # Convert schema information into usable string
    for table, schema in schema_info.items():
        # print(f"Table: {table}")
        schema_string = f"CREATE TABLE {table} (\n"
        for column in schema:
            # print(f"  Column: {column[1]}, Type: {column[2]}") 
            schema_string += f"     {column[1]} {column[2]},\n"
        schema_text[table] = schema_string[:-2] + '\n);\n'

    return schema_text, schema_info

def get_foreign_keys(database_path,  table_selection=None):
    """ For when the foreign_key_list is available"""
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    # Dictionary to store foreign key information
    foreign_keys_info = {}
    
    # Fetch foreign key info for each table
    for table in [table for table in tables if table[0] in table_selection] if table_selection else tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA foreign_key_list({table_name});")
        foreign_keys = cursor.fetchall()
        if foreign_keys:
            foreign_keys_info[table_name] = foreign_keys
    
    return foreign_keys_info

def get_potential_foreign_keys(database_path, table_selection=None):
    """ For when the foreign_key_list is not available """
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    # Dictionary to store column information for each table
    table_columns = {}
    
    # Fetch column names for each table
    for table_name in tables:
        cursor.execute(f"PRAGMA table_info({table_name});")
        table_columns[table_name] = [row[1] for row in cursor.fetchall()]
    
    # Dictionary to store potential foreign key information
    potential_foreign_keys = {}
    
    # Find potential foreign key relationships
    table_columns = {table_name: columns for table_name, columns in table_columns.items() if table_name in table_selection} if table_selection else table_columns
    for table_name, columns in table_columns.items():
        matches = []
        for target_table, target_columns in table_columns.items():
            if table_name != target_table:  # Avoid self-referencing
                for column in columns:
                    if column in target_columns and column.endswith('_id'):
                        matches.append((column, target_table))
        if matches:
            potential_foreign_keys[table_name] = matches
    
    return potential_foreign_keys

def get_schema_and_links(history_baseball_db_path, limited_table_names):
        
    schema_text, schema_info = get_db_schema(history_baseball_db_path, limited_table_names)
    # schema_text, schema_info = get_db_schema(history_baseball_db_path)
    # foreign_keys_info = get_foreign_keys(history_baseball_db_path, limited_table_names)
    # Printing the schema information
    # print("Schema Information:")
    # for table, schema in schema_info.items():
    #     print(f"Table: {table}")
    #     for column in schema:
    #         print(f"  Column: {column[1]}, Type: {column[2]}") 


    # # Printing the foreign key information  ### KEY INFORMATION WAS NOT AVAILABLE
    # print("\nForeign Key Information:")
    # for table, fkeys in foreign_keys_info.items():
    #     print(f"Table: {table}")
    #     for fkey in fkeys:
    #         print(f"  Column: {fkey[3]} links to {fkey[2]} on {fkey[0]}")s
    potential_foreign_keys = get_potential_foreign_keys(history_baseball_db_path, limited_table_names)
    # potential_foreign_keys = get_potential_foreign_keys(history_baseball_db_path)
    schemata = ''
    table_joins = ''
    mentioned_tables = []
    for table, schema in schema_text.items():
        mentioned_tables.append(table)
        # print(schema)
        # print(potential_foreign_keys[table])
        schemata += schema + '\n'
        for join in potential_foreign_keys[table]:
            if join[1] not in mentioned_tables:
                table_joins += f'-- {table}.{join[0]} can be joined with {join[1]}.{join[0]}\n'
    return schemata, table_joins



# ## Plots
def barplot_queries_evaluation_by_llm(scores_by_llm):
    num_scores = len(scores_by_llm)
    num_queries = len(scores_by_llm[next(iter(scores_by_llm.keys()))]['structure'])  # Assuming each scenario has the same number of queries
    index = np.arange(num_queries)
    bar_width = 0.2

    # Colors and labels for the score types
    colors = ['red', 'green', 'blue']
    labels = ['structure', 'blocks', 'results']
    
    # Create a figure with subplots for each LLM
    fig, axs = plt.subplots(num_scores, figsize=(15, 8), sharey=True)
    max_score = 0  # Variable to track the maximum score

    for llm_idx, (llm_type, scores) in enumerate(scores_by_llm.items()):
        ax = axs[llm_idx] if num_scores > 1 else axs
        legend_info = []

        for i, score_label in enumerate(labels):
            values = scores[score_label]
            ax.bar(index + i * bar_width, values, bar_width, color=colors[i])
                        # Update max_score if needed
            max_score = max(max_score, max(values))

            # Calculate the full score count and average score for the label
            full_scores_count = sum(1 for score in values if score == 1)
            avg_score = np.mean(values)
            legend_info.append(f"{score_label} (Full: {full_scores_count}, Avg: {avg_score:.2f})")

        ax.set_ylabel('Scores')
        ax.set_title(f'LLM: {llm_type}')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels([f'Q{n+1}' for n in range(num_queries)], rotation=45)

        # Set legend with enhanced information
        ax.legend(legend_info, loc='upper right')

    padding = 0.5  # Adjust padding as needed
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for ax in axs:
        ax.set_ylim(0, max_score + padding)


    plt.tight_layout()
    plt.show()

def barplot_queries_evaluation_by_score(scores_by_llm):
    # Assuming scores_by_llm is a dictionary where keys are llm_types and values are dictionaries of scores
    bar_width = 4  # Adjust spacing for clarity
    num_llm_types = len(scores_by_llm)
    num_queries = len(next(iter(scores_by_llm.values()))['structure'])  # Get the number of queries from the first llm_type

    # Increase the spacing factor for more space between groups of bars
    spacing_factor = 1.5  # Adjust this value as needed for more space
    index = np.arange(num_queries) * (num_llm_types + 1) * bar_width * spacing_factor

    labels = ['structure', 'blocks', 'results']
    colors = ['blue', 'red', 'black','green']  # Colors for each llm_type
    full_score_threshold = 1.0  # Define what you consider a full score

    # Create a subplot for each score type
    fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True, sharey=True)

    max_score = 0  # Variable to track the maximum score

    for score_idx, score_label in enumerate(labels):
        ax = axs[score_idx]
        legend_labels = []

        for llm_idx, (llm_type, scores) in enumerate(scores_by_llm.items()):
            values = scores[score_label]
            bar_positions = [x + llm_idx * bar_width for x in index]
            ax.bar(bar_positions, values, bar_width, color=colors[llm_idx % len(colors)])
            
            # Update max_score if needed
            max_score = max(max_score, max(values))
            
            # Calculate statistics for legend
            full_scores = sum(1 for score in values if score >= full_score_threshold)
            avg_score = np.mean(values)
            legend_label = f'{llm_type} - Full: {full_scores}, Avg: {avg_score:.2f}'
            legend_labels.append(legend_label)

        ax.set_ylabel('Scores')
        ax.set_title(f'Scores for {score_label.capitalize()}')

        if score_idx == 2:  # Only set x-axis labels for the bottom subplot
            ax.set_xlabel('Query Number')
            ax.set_xticks(index + (bar_width * num_llm_types / 2) * spacing_factor)
            ax.set_xticklabels([f'Q{n+1}' for n in range(num_queries)], rotation=45)

        ax.legend(legend_labels, loc='upper right', title='LLM Type')

    # Increase the y-axis upper limit based on max_score plus some padding
    padding = 0.7  # Adjust padding as needed
    for ax in axs:
        ax.set_ylim(0, max_score + padding)

    plt.tight_layout()
    plt.show()
            