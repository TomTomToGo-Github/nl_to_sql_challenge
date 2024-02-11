# Natural language to sql
The aim is to develop a working prototype that converts natural language into usable SQL queries

## Step 1. Using this repo
-> Create virtual environment to match defaultInterpreterPath in settings.json (set up in python 3.12.0)
```py -m venv .venv```

## https://docs.pytest.org/en/7.1.x/explanation/goodpractices.html
```pip install e .``` (install in editable mode)

## Brainstorm
1. identify details of the task - how many different (parse with sqlparse, or alternatively with REGEX or ANTLR if not good enough)
    - SELECT (=column names)
    - FROM (table names)
    - WITH (filters -> columns equal or smaller than)
    - JOINS (INNER, OUTER, ...) + table aliases
    - AGGREGATION FUNCTIONS (functions like count, max, min, avg etc.)
    - HAVING (filters -> aggregated values)
    - NESTED (SELECT inside FROM statement)
    - ORDER BY (no priority -> might be easy addition in the end)
    - LIMIT (no priority -> might be an easy addition in the end)

2. Create the evaluation first
    - Different variations that lead to the same result (parser? and/or regex to deal with character cases, spaces, newlines, tabs, \t, \r)
    - Similarity -> Count the number of correct blocks normalized by the number of blocks in result
    - Similarity + Importance (prioritize correct data before Sorting and Limit adds)

3. Step by step approaches
    a. Simple problem (just SELECT and FROM correctly)
    b. INCLUDE joins with aliases
    c. INCLUDE filters and aggregations
    d. NESTED and sorting, + lim

4. Methods appraoch
    - Establish baseline accuracy -> question to LLM -> prompt and parse to get only SQL code for  (mistral 7B running local and openaI)
    - Enrich with database info
    - play with prompts
