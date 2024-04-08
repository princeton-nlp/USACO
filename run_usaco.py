'''
Contains all code to duplicate experiments in "Can language models solve olympiad programming questions?"
To utilize open models, create your own callable model function in models.py, and import it as with GPTs/Claude.
'''

import argparse
from functools import partial
from rank_bm25 import BM25Okapi
from models import gpts, claude 
from utils import load_json, save_json, generate_episodic_retrieval_queries, generate_semantic_retrieval_queries, generate_episodic_semantic_retrieval_queries
from USACOBench.prompts import solve_prompt_fn, retrieval_prompt_fn, reflexion_prompt_fn, RetrievalType
from USACOBench.data_utils import load_corpus, load_problem_dict, load_problems
from evaluate import evaluate_model
from USACOBench.evaluation import print_metrics
from dotenv import load_dotenv
from collections import Counter
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', help='model endpoint: ie. gpt-4-1106-preview', default='gpt-3.5-turbo')
parser.add_argument('-e', '--episodic_retrieval', help='whether to use episodic retrieval', default=0)
parser.add_argument('-s', '--semantic_retrieval', help='whether to use semantic retrieval', default=0)
parser.add_argument('-r', '--reflexion', help='whether to use reflexion', default=0)
parser.add_argument('-a', '--attempts', help='number of attempts', default=1)
args = parser.parse_args()

model_name = args.model_name
if 'gpt' in model_name:
    model_fn = gpts
elif 'claude' in model_name:
    model_fn = claude
else: 
    raise Exception("Model name not one of gpt or claude. Please modify code to add model support.")

problem_dict = load_problem_dict('usaco307')
model_fn = partial(model_fn, model=model_name)

if not args.episodic_retrieval and not args.semantic_retrieval and not args.reflexion:
    queries = []
    for problem_id in problem_dict.keys():
        queries.append({'problem_id': problem_id, 'problem_description': problem_dict[problem_id]['description']})

    rdict, sdict, rs, ss = evaluate_model(model_fn, solve_prompt_fn, queries=queries, verbose=True, attempts=1, problem_ids=list(problem_dict.keys())[:1])
    save_json([rdict, sdict, rs, ss], f'results/results_solve_{args.attempts}attempts')
    
elif args.episodic_retrieval and not args.semantic_retrieval and not args.reflexion:
    queries = []
    for problem_id in problem_dict.keys():
        queries.append({'problem_id': problem_id, 'problem_description': problem_dict[problem_id]['description']})

    rdict, sdict, rs, ss = evaluate_model(model_fn, solve_prompt_fn, queries=queries, verbose=True, attempts=1, problem_ids=list(problem_dict.keys())[:1])
    save_json([rdict, sdict, rs, ss], f'results/results_solve_{args.attempts}attempts')

    num_problems_fetched = 2
    queries = generate_episodic_retrieval_queries(num_problems_fetched, problem_dict, ss)
    retrieval_prompt_fn = partial(retrieval_prompt_fn, retrieval_type=RetrievalType.EPISODIC)
    rdict, sdict, rs, ss = evaluate_model(model_fn, retrieval_prompt_fn, queries=queries, verbose=True, attempts=1, problem_ids=list(problem_dict.keys())[:1])
    save_json([rdict, sdict, rs, ss], f'results/results_episodic_retrieval_{args.attempts}attempts')

elif not args.episodic_retrieval and args.semantic_retrieval and not args.reflexion:
    queries = []
    for problem_id in problem_dict.keys():
        queries.append({'problem_id': problem_id, 'problem_description': problem_dict[problem_id]['description']})

    rdict, sdict, rs, ss = evaluate_model(model_fn, solve_prompt_fn, queries=queries, verbose=True, attempts=1, problem_ids=list(problem_dict.keys())[:1])
    save_json([rdict, sdict, rs, ss], f'results/results_solve_{args.attempts}attempts')

    queries = generate_semantic_retrieval_queries(problem_dict, ss)
    retrieval_prompt_fn = partial(retrieval_prompt_fn, retrieval_type=RetrievalType.SEMANTIC)
    rdict, sdict, rs, ss = evaluate_model(model_fn, retrieval_prompt_fn, queries=queries, verbose=True, attempts=1, problem_ids=list(problem_dict.keys())[:1])
    save_json([rdict, sdict, rs, ss], f'results/results_episodic_retrieval_{args.attempts}attempts')

elif args.episodic_retrieval and args.semantic_retrieval and not args.reflexion:
    queries = []
    for problem_id in problem_dict.keys():
        queries.append({'problem_id': problem_id, 'problem_description': problem_dict[problem_id]['description']})

    rdict, sdict, rs, ss = evaluate_model(model_fn, solve_prompt_fn, queries=queries, verbose=True, attempts=1, problem_ids=list(problem_dict.keys())[:1])
    save_json([rdict, sdict, rs, ss], f'results/results_solve_{args.attempts}attempts')

    queries = generate_episodic_semantic_retrieval_queries(problem_dict, ss)
    retrieval_prompt_fn = partial(retrieval_prompt_fn, retrieval_type=RetrievalType.EPISODIC_SEMANTIC)
    rdict, sdict, rs, ss = evaluate_model(model_fn, retrieval_prompt_fn, queries=queries, verbose=True, attempts=1, problem_ids=list(problem_dict.keys())[:1])
    save_json([rdict, sdict, rs, ss], f'results/results_episodic_semantic_retrieval_{args.attempts}attempts')

print_metrics(rs)
print('Result summary:')
result_types = [result['result_type'] for result_set in rs for result in result_set]
print(Counter(result_types))
print()



