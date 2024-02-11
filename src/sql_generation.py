## built-in modules
import os
import json
import time
from pathlib import Path
## pip installed modules
import requests
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import GPT4All ## CALL IS DIFFERENT FROM STANDARD GPT4All
from dotenv import load_dotenv
## self-made modules
from src.utils import normalize_query_string, get_schema_and_links
import time

# MODEL_NAME = "sqlcoder2.Q5_0.gguf"  # just nonsense with quantized version? or because LM-Studio reported failed?
MODEL_NAME = "sqlcoder-7b.Q5_K_M"
MDOEL_PATH = Path(r"C:/Users/thoma/Desktop/Tommy/Programmieren/gpt4all/models")
MODELS = {'sql_coder': GPT4All(model=str(MDOEL_PATH / MODEL_NAME), temp=0)}

_ = load_dotenv()   
URL = "https://api.mistral.ai/v1/chat/completions"
TOKEN = os.getenv('MISTRAL_API_KEY')
# MODEL = "mistral-medium"  # mixtral 8x7B is mistral-small -> both open source to host by oneself
MISTRAL_HEADERS = {'Authorization': f'Bearer {TOKEN}'}


def prompt_mistral(model, prompt):
    content = ""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        # "top_p": 1,
        # "max_tokens": 16,
        # "stream": False,
        # "safe_prompt": False,
        # "random_seed": None
    }
    response = requests.post(URL, headers=MISTRAL_HEADERS, json=data)
    if response.status_code == 200:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        # print(content)
    else:
        print(f"Error: Received status code {response.status_code}")
        print(response.text)
    return content

def prompt_local(model, prompt):
    model = MODELS[model]
    return model.invoke(prompt)
    
def format_prompt(template_vars, prompt_template_file):
    with open(prompt_template_file, "r", encoding="utf-8") as file:
        prompt_template = file.read()

    prompt_template = PromptTemplate(
        template=prompt_template,
        input_variables=list(template_vars.keys()),
    )

    formatted_prompt = prompt_template.format_prompt(**template_vars)
    
    return formatted_prompt

def nl_to_sql(template_vars, prompt_template_file):
    formatted_prompt = format_prompt(template_vars, prompt_template_file)
    print(formatted_prompt.text)
    # Quick invoke
    llm_openai_turbo = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    # llm_openai_4 = OpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    # try:
    #     response_openai_4 = llm_openai_4.invoke(formatted_prompt.text)
    # except:
    #     response_openai_4 = "An error occurred while prompting gpt-4."
    print("Question: ", template_vars['question'])
    try:
        print("     Prompting gpt-3.5-turbo-instruct")
        response_openai_turbo = llm_openai_turbo.invoke(formatted_prompt.text)
        # response_openai_turbo = llm_openai_turbo.invoke(formatted_prompt.text[:-130])
    except:
        response_openai_turbo = "An error occurred while prompting gpt-3.5-turbo."
    try:
        print("     Prompting mistral-small")
        response_mistral_small = prompt_mistral("mistral-small", formatted_prompt.text) ## mixtral 8x7B
    except:
        response_mistral_small = "An error occurred while prompting mistral small."
    try:
        print("     Prompting mistral-medium")
        response_mistral_medium = prompt_mistral("mistral-medium", formatted_prompt.text)
    except:
        response_mistral_medium = "An error occurred while prompting mistral medium."
    try:
        print("     Prompting sql_coder locally")
        response_sql_coder = prompt_local("sql_coder", formatted_prompt.text)
        # response_sql_coder = 'Not prompted yet'
    except:
        response_sql_coder = "An error occurred while prompting local sql_coder."
    
    sql_queries = {
        'Question': template_vars['question'],
        'Generated': {
            "openai_turbo": normalize_query_string(response_openai_turbo),
            # "openai_4": normalize_query_string(response_openai_4),
            "mistral_small": normalize_query_string(response_mistral_small),
            "mistral_medium": normalize_query_string(response_mistral_medium),
            "sql_coder": normalize_query_string(response_sql_coder)
        }
    }
    return sql_queries


history_baseball_db_path = Path(r"C:\Users\thoma\Desktop\Tommy\Programmieren\nl_to_sql_challenge\data\history_of_baseball.sqlite")
# path_data = Path(__file__).parent.parent / 'data'  # when running from the src directory
path_data = Path(os.getcwd()) / 'data'  # when running in console manually
file_ground_truth = path_data / 'examples_queries_test.json'
file_generated_data = path_data / 'generated_queries.json'
with open(file_ground_truth, 'r', encoding='utf-8') as f:
    data_ground_truth = json.load(f)





## Step 1: Convert questions into sql with standard prompt template
template_vars = {}
question_generation_list = []
prompt_template_filename = Path("prompt_template_1.txt")
for idx_item, item_gt in enumerate(data_ground_truth):
    template_vars['question'] = item_gt.get('question')
    ground_truth = item_gt.get('query')
    prompt_template_file = Path(os.getcwd()) / Path('./src/prompting') / prompt_template_filename
    sql_queries = nl_to_sql(template_vars, prompt_template_file)
    sql_queries['Ground truth'] = item_gt['query']
    question_generation_list.append(sql_queries)
    print(f"Question {idx_item + 1}: {template_vars['question']}")
    print(f"      Ground truth: {ground_truth}")
    for key, query in sql_queries['Generated'].items():
        print(f"      {key:28}:    {normalize_query_string(query)}")
    print("\n")
## Save results with unix timestamp
path_results = Path("results")
name_results = f'results_{prompt_template_filename.stem}_{int(time.time())}.json'
with open(path_results / name_results, 'w', encoding='utf-8') as f:
    f.write(json.dumps(question_generation_list, indent=4, ensure_ascii=False))




## Step 2: Try to improve results using prompt schema - first by "cheating" with known limited table namess
## Idea to limit prompt schema -> use RAG to narrow down potential tables by finding columns and table names that are similar to question
limited_table_names = ['player_award_vote', 'salary', 'player', 'player_award', 'hall_of_fame']
# Extract schema and foreign key information
history_baseball_db_path = Path(r"C:\Users\thoma\Desktop\Tommy\Programmieren\nl_to_sql_challenge\data\history_of_baseball.sqlite")
schema_string, table_joins = get_schema_and_links(history_baseball_db_path, limited_table_names)
# len(schema_string)
# len(table_joins)

## Convert question into sql
template_vars = {
    # "schemata": '',
    "schemata": schema_string,
    "table_joins": table_joins
}
question_generation_list = []
prompt_template_filename = Path("prompt_template_schema.txt")
for idx_item, item_gt in enumerate(data_ground_truth[:]):
    template_vars['question'] = item_gt.get('question')
    prompt_template_file = Path(os.getcwd()) / Path('./src/prompting') / prompt_template_filename
    sql_queries = nl_to_sql(template_vars, prompt_template_file)
    sql_queries['Ground truth'] = item_gt['query']
    question_generation_list.append(sql_queries)
    print(f"Question {idx_item + 1}: {template_vars['question']}")
    print(f"      Ground truth: { sql_queries['Ground truth']}")
    for key, query in sql_queries['Generated'].items():
        print(f"      {key:28}:    {normalize_query_string(query)}")
    print("\n")
path_results = Path("results")
name_results = f'results_{prompt_template_filename.stem}_{int(time.time())}.json'
with open(path_results / name_results, 'w', encoding='utf-8') as f:
    f.write(json.dumps(question_generation_list, indent=4, ensure_ascii=False))


## Next and final steps
-> load all my queries into the evaluation plot
-> compute scrores -> how many yielded better results (1 if result ok. else -> maximum of .5 if block structure and elements is similar)
-> implement the template with limiting the tables and adding the schema

-> compare and see if improved

-> final step -> implement RAG to find these tables?

-> last -> select winning model and templating strategy and think about improvements:
        - model side with finetuning or prompt augmentation
        - evaluation side
        - CI/CD -> how can The results be used automatically?
        
        in theory it makes sense to flatten out the parsed sql and insert
        it into a fine tuning process.
        
        
        
