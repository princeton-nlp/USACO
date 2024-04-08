# In-Memory Learning

# Setup

Run
```
git clone https://github.com/princeton-nlp/USACO
cd USACO
pip install -r requirements.txt
```

See download links and details about data in the Data section below.

## OpenAI API key

Run
`echo OPENAI_API_KEY=[YOUR_KEY] >> .env`.

In the code, we read this via
```
from dotenv import load_dotenv
load_dotenv()
```
and seems less buggy than `export OPENAI_API_KEY`.

## Judge sandbox directories

Local evaluation requires absolute paths — please update the global path constants at the top of the files `USACOBench/evaluation/judges/usaco_judge.py`. A good default
is to set them to `{parent of this repository}/USACO/{copy rest of path}`.

Note that as more solutions are generated, the directory may become large. Optionally to save space, you may also delete solutions or pickle files you no longer want at any time, as long as judging is not currently running.

# Data

USACO-related data (problems and problem dict) and the `cp_handbook` corpus can be downloaded [here](https://drive.google.com/drive/folders/1ZC0lVRCnlSaIo6eTjEc_vPtFJLZWbFAO?usp=sharing). For evaluation, please add `data/corpuses` and `data/datasets` directories under the repository root, and move any downloaded data to locations as described below:

* `data/corpuses/`: corpuses of competitive programming articles and book chapters (json)
    * `cpbook_v2`: [cp-algorithms book](https://cp-algorithms.com/), split into 157 articles. We currently just use `article` (raw text with problems section removed) for retrieval, but we also provide keys `title` and `full_article` (problems section included) in the corpus.
    * `cp_handbook`: [CP Handbook](https://github.com/pllk/cphb), split into 118 LaTeX sections. We currently use `section_content` for retrieval, but also provide keys `section_title`, `chapter_title`, `chapter_path` (path in the source repo).
* `data/datasets/`: datasets of competitive programming problems (HuggingFace datasets)
    * `usaco_v2`: 484 USACO problems (all available problems with test cases up to September 2023)
    * subsets of `usaco_v2` filtered by samples:
        * `usaco_v2_sample_pred_bronze`: 139 USACO bronze problems
        * `usaco_v2_sample_pred_bronze_sub`: 65 USACO bronze problems with explanations for all samples
        * `usaco_v2_sbs_38`: 38 USACO bronze problems with explanations for all samples (arbitrary subset of the above)
        * `usaco_v2_bronze_sbs_hard`: 19 USACO bronze problems with explanations for all samples (hard subset of `usaco_v2_sbs_38` where GPT4-turbo has a < 10% pass rate on 100 attempts)
        * `usaco_v2_explan_bronze`: alias of `usaco_v2_sample_pred_bronze_sub`
        * `usaco_v2_explan_bronze_solutions_english`: 63 USACO bronze problems with explanations for all samples and `solution_english` key
        * `usaco_v2_explan_bronze_solutions_full`: 42 USACO bronze problems with explanations for all samples, `solution_english` key, and `solution_python3` key
        * `usaco_v2_explan_silver`: 116 USACO silver problems with explanations for all samples
        * `usaco_v2_explan_gold`: 106 USACO gold problems with explanations for all samples
        * `usaco_v2_explan_platinum`: 116 USACO platinum problems with explanations for all samples
    * subsets of `usaco_v2` filtered by contest type (annual US Open contests have harder problems)
        * `usaco_v2_bronze_us_open` (38 problems)
        * `usaco_v2_bronze_non_us_open` (101 problems)
        * `usaco_v2_silver_us_open` (36 problems)
        * `usaco_v2_silver_non_us_open` (99 problems)
    * `cpbook_problems_v2`: the subset of 142 CodeForces problems from the `code_contests/` dataset that appear as exercises in the cp-algorithms book, contains keys `description`, `problem_id`, `cf_dataset_idx`, `cf_index`, `cf_rating`, some tests, and other metadata

We provide utility functions for loading and saving datasets, e.g.
```python
from iml.data_utils import load_problems, load_problem_dict, load_corpus
cpbook_problems = load_problems('cpbook_problems_v2', num_problems=50)
cpbook_problem_dict = load_problem_dict('cpbook_problems_v2')
cpbook_articles = load_corpus('cpbook_v2')
```

Then, creating an GPT4 agent and evaluating it on this dataset is as simple as:
```python
from iml.agents import Agent
from iml.evaluation import evaluate_agent, print_metrics
agent = Agent()
result_sets, solution_sets = evaluate_agent(agent,
               cpbook_problems.select(range(10)),
               cpbook_problem_dict,
               attempts=5)
print_metrics(result_sets)
```

# USACO eval

We evaluate Python locally using `exec`; the execution code is adapted from OpenAI's [HumanEval](https://github.com/openai/human-eval/blob/master/human_eval/execution.py) task -- note that it is important to **take care when running unverified model-generated code**.
