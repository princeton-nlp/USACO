from typing import List, Dict, Any, Tuple, Union, Callable
import pickle
import json
from datetime import datetime
from typing import Dict, List
from USACOBench.evaluation.metrics import pass_at_k
from rank_bm25 import BM25Okapi

Problem = Dict[Any, Any]
Solution = Dict[str, Union[str, None]]
SolutionSet = List[Solution]
SolutionDict = Dict[str, SolutionSet]
Result = Dict[str, str]
ResultSet = List[Result]
ResultDict = Dict[str, ResultSet]
Query = Dict[str, str]

# NOTE: these are technically not thread-safe, although they are up to
# microsecond approximation due to timestamps

def save_json(obj, path, timestamp=True, verbose=True):
    if timestamp:
        timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
        fname = '{}_{}.json'.format(path, timestamp_str)
    else:
        fname = '{}.json'.format(path)
    print('Saved json at {}'.format(fname))
    with open(fname, 'w') as f:
        json.dump(obj, f)
    return fname

def load_json(fname, verbose=True):
    with open(fname + '.json', 'r') as f:
        return json.load(f)

def save_pickle(obj, path, timestamp=True, verbose=True):
    if timestamp:
        timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
        fname = '{}_{}.pickle'.format(path, timestamp_str)
    else:
        fname = '{}.pickle'.format(path)
    print('Saved pickle at {}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(fname, verbose=True):
    with open(fname + '.pickle', 'rb') as f:
        return pickle.load(f)

def get_rdict(rs, ss):
    '''
    Takes (index-matched) lists of result sets and solution sets and returns
        the relevant result dictionary, i.e. dictionary from problem_id to result set
        Assumes at least one attempt for each problem.
    '''
    rdict = {}
    for r, s in zip(rs, ss):
        rdict[s[0]['problem_id']] = r
    return rdict

def filter(result_sets, difficulties=['bronze'], problem_dict=None):
    output = []
    for result_set in result_sets:
        trues = False
        for difficulty in difficulties:
            trues = trues or problem_dict[result_set[0]['problem_id']]['problem_level'] == difficulty
        if trues:
            output.append(result_set)
    return output

def search(text, problem_descriptions, problem_ids):
    for i, problem in enumerate(problem_descriptions):
        if text == problem:
            return problem_ids[i]
    return None

def combine(result_sets1, result_sets2):
    combined = []
    for result_set in result_sets1:
        temp = result_set + find(result_set[0]['problem_id'], result_sets2)
        combined.append(temp)
    return combined

def find(problem_id, result_sets): 
    for result_set in result_sets:
        if result_set[0]['problem_id'] == problem_id:
            return result_set
    return None

def calculate_percentage_identical(queries, target_queries):
    target_queries_dict = {}
    score = 0
    for query in target_queries:
        target_queries_dict[query['problem_id']] = query
    for query in queries:
        if query['similar_problem_id'] == target_queries_dict[query['problem_id']]['similar_problem_id']:
            score += 1
    return score / len(queries) * 100

def filter_queries(in_queries, queries):
    in_set = set([query['problem_id'] for query in in_queries])
    result = []
    for query in queries:
        if query['problem_id'] in in_set:
            result.append(query)
    return result

def parse_execution_output(execution_output):
    result_list = execution_output['result_list']
    if not result_list:
        result_list_str = execution_output['status']
    else:
        result_list_str = '\n'.join([result['status'] for result in result_list])
    num_tests = execution_output['num_tests']
    num_passed = execution_output['num_passed']
    return f'Testing Output: \n {result_list_str} \n Number of tests: {num_tests} \n Number passed: {num_passed}'

def get_difficulty_performances(full_results, problem_dict, k=1):
    bronze = filter(full_results, ['bronze'], problem_dict=problem_dict)
    silver = filter(full_results, ['silver'], problem_dict=problem_dict)
    gold = filter(full_results, ['gold'], problem_dict=problem_dict)
    platinum = filter(full_results, ['platinum'], problem_dict=problem_dict)
    return pass_at_k(bronze, k=k)[0], pass_at_k(silver, k=k)[0], pass_at_k(gold, k=k)[0], pass_at_k(platinum, k=k)[0]

# p is the number of problems retrieved
def generate_episodic_retrieval_queries(p, problem_dict, solution_sets):
    # Generate 3-problem queries:
    corpus = [problem_dict[problem_id]['description'] + '\nSolution: \n' + problem_dict[problem_id]['solution_english'] + '\nSolution Code: \n' + problem_dict[problem_id]['solution_python3'] for problem_id in problem_dict.keys()]
    solutions = solution_sets # assuming [rdict, sdict, rs, ss] format
    query_texts = []
    problem_ids = []
    for solution in solutions:
        solution_text = solution[0]['solution']
        problem_id = solution[0]['problem_id']
        if problem_id in problem_dict.keys():
            query_texts.append(problem_dict[problem_id]['description'] + '\n' + solution_text)
            problem_ids.append(problem_id)


    sim_prob_queries_new_code = []
    for i, problem_id in enumerate(problem_ids):
        curr_description = problem_dict[problem_id]['description'] + '\nSolution: \n' + problem_dict[problem_id]['solution_english'] + '\nSolution Code: \n' + problem_dict[problem_id]['solution_python3']
        duplicated = corpus[:]
        duplicated.remove(curr_description)
        tokenized_corpus = [doc.split(' ') for doc in duplicated]
        bm25 = BM25Okapi(tokenized_corpus)

        curr_query = query_texts[i]
        tokenized_query = curr_query.split(" ")
        similar_problem_texts = bm25.get_top_n(tokenized_query, duplicated, n=p)
        similar_problem_text = ""
        words = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
        similar_problem_ids = []
        for i, text in enumerate(similar_problem_texts):
            similar_problem_text += f"\n\n {words[i]} problem and solution \n\n" + text 
            similar_problem_ids.append(search(text, corpus, problem_ids))

        for val in similar_problem_ids:
            if val == None:
                print(similar_problem_texts)
                assert 1 == 2
        sim_prob_queries_new_code.append({'problem_id': problem_id, 'retrieval_text': '[BEGIN SIMILAR PROBLEMS]\n' + similar_problem_text + '\n[END SIMILAR PROBLEMS]\n', 'retrieval_problem_ids': similar_problem_ids})
    save_json(sim_prob_queries_new_code, f'queries_firstsolve_{p}problem_episodic')
    return sim_prob_queries_new_code

def generate_semantic_retrieval_queries(problem_dict, solution_sets):
    # Fetching the corpus to pass in: The corpus being textbook chapters
    textbook = load_json('data/corpuses/cpbook_v2')
    textbook_corpus = [article['full_article'] for article in textbook]
    tokenized_corpus = [chapter.split(' ') for chapter in textbook_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query_texts = dict()

    solutions = solution_sets
    query_texts = []
    problem_ids = []
    for solution in solutions:
        solution_text = solution[0]['solution']
        problem_id = solution[0]['problem_id']
        if problem_id in problem_dict.keys():
            query_texts.append(problem_dict[problem_id]['description'] + '\n' + solution_text)
            problem_ids.append(problem_id)

    textbook_queries_new_code = []
    for i, problem_id in enumerate(problem_ids):
        curr_query = query_texts[i]
        tokenized_query = curr_query.split(" ")
        textbook_text = bm25.get_top_n(tokenized_query, textbook_corpus, n=1)
        textbook_queries_new_code.append({'problem_id': problem_id, 'retrieval_text': '[BEGIN TEXTBOOK EXCERPT]\n' + textbook_text[0]  + '\n[END TEXTBOOK EXCERPT]\n'})
    save_json(textbook_queries_new_code, 'queries_firstsolve_semantic')
    return textbook_queries_new_code

def generate_episodic_semantic_retrieval_queries(problem_dict, solution_sets):
    textbook_queries = generate_semantic_retrieval_queries(problem_dict, solution_sets)
    episodic_queries = generate_episodic_retrieval_queries(2, problem_dict, solution_sets)
    textbook_queries_dict = dict()
    episodic_queries_dict = dict()

    for query in textbook_queries:
        textbook_queries_dict[query['problem_id']] = query
    for query in episodic_queries:
        episodic_queries_dict[query['problem_id']] = query
    
    final_queries = []
    for query in textbook_queries:
        retrieval_text_textbook = query['retrieval_text']
        episodic_query = episodic_queries_dict[query['problem_id']]
        retrieval_text_episodic = episodic_query['retrieval_text'] 
        retrieval_text = + retrieval_text_episodic + '\n\n' + retrieval_text_textbook 
        final_queries.append({'retrieval_problem_ids': episodic_query['retrieval_problem_ids'], 'retrieval_text': retrieval_text, 'problem_id': query['problem_id']})

    save_json(final_queries, 'queries_firstsolve_episodic_semantic')
    return final_queries