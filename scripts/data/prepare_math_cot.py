from typing import Any
import hashlib

from numpy.random import f

from openrlhf.math.parser import parse_ground_truth, parse_question, extract_answer
from openrlhf.math.grader import math_equal

from tabulate import tabulate
from colorama import init, Fore, Style

import argparse

import datasets

from collections import defaultdict
import numpy as np
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
import re
from typing import List, Dict, Set, Tuple


from datasets import load_dataset, concatenate_datasets, Dataset


PROMPT = """Solve the following math problem efficiently and clearly:
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.
"""

def prepocess_dataset_row(row: dict[str, Any]) -> dict[str, Any]:

    question = row["question"]
    answer = row["answer"]

    # Create a deterministic UUID-like hash from the question
    question_hash = hashlib.md5(question.encode()).hexdigest()
    
    return {
        "question_hash": question_hash,
        "question": question,
        "answer": answer,
    }

def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": f"{row['question']}",
            },
        ]
    }

def clean_ds(datasets: list):

    filtered_ds = []
    for ds in datasets:
        ds = ds.select_columns(["question"])
        ds = ds.map(lambda x: {"question": x["question"]["problem"], "answer": x["question"]["ground_truth_answer"]})

        # drop duplicate questions
        df = ds.to_pandas()
        # if ignore_index=False, the output will contain, some other index field.
        df_dedup = df.drop_duplicates(subset=["question"], ignore_index=True)
        ds_dedup = Dataset.from_pandas(df_dedup)


        filtered_ds.append(ds_dedup)
    
    return concatenate_datasets(filtered_ds)

def load_datasets():

    ds_train_1 = load_dataset("tasksource/PRM800K", split="train", data_files=["phase1_train.jsonl"])
    ds_train_2 = load_dataset("tasksource/PRM800K", split="train", data_files=["phase2_train.jsonl"])
    ds_test_1 = load_dataset("tasksource/PRM800K", split="train",  data_files=["phase1_test.jsonl"])
    ds_test_2 = load_dataset("tasksource/PRM800K", split="train",  data_files=["phase2_test.jsonl"])

    ds_train = clean_ds([ds_train_1, ds_train_2])
    ds_test = clean_ds([ds_test_1, ds_test_2])

    return ds_train, ds_test



def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Create a MinHash object from text."""
    m = MinHash(num_perm=num_perm)
    # Create word-level shingles
    words = re.findall(r'\w+', text.lower())
    for word in words:
        m.update(word.encode('utf-8'))
    return m


def create_inverted_index(texts: List[str]) -> Dict[str, Set[int]]:
    """Create an inverted index for exact matching."""
    index = defaultdict(set)
    for idx, text in enumerate(texts):
        # Normalize and tokenize
        words = set(re.findall(r'\w+', text.lower()))
        for word in words:
            index[word].add(idx)
    return index


def get_candidate_matches(query_text: str, inverted_index: Dict[str, Set[int]]) -> Set[int]:
    """Get candidate matches using the inverted index."""
    words = set(re.findall(r'\w+', query_text.lower()))
    # Get documents that share at least one word
    candidates = set()
    for word in words:
        candidates.update(inverted_index.get(word, set()))
    return candidates


def check_dataset_contamination(train_ds, test_ds, similarity_threshold: float = 0.7):
    """
    Check for contamination between training and test datasets using scalable methods.
    
    Args:
        train_ds: Training dataset with 'question' feature
        test_ds: Test dataset with 'question' feature
        similarity_threshold: Threshold for LSH similarity (default: 0.7)
    
    Returns:
        dict: Dictionary containing contamination statistics
    """
    train_questions = train_ds['question']
    test_questions = test_ds['question']
    
    print("Building indexes...")
    
    # Create inverted index for train questions (for exact matching)
    train_inverted_index = create_inverted_index(train_questions)
    
    # Initialize LSH index for approximate similarity matching
    lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
    
    # Add training examples to LSH index with progress bar
    train_minhashes = []
    for idx, question in enumerate(tqdm(train_questions, desc="Creating MinHashes for train")):
        mh = create_minhash(question)
        train_minhashes.append(mh)
        lsh.insert(f"train_{idx}", mh)
    
    exact_matches = []
    similar_matches = []
    
    # Process test questions with progress bar
    for test_idx, test_q in enumerate(tqdm(test_questions, desc="Processing test questions")):
        # 1. Check for exact matches using inverted index
        candidates = get_candidate_matches(test_q, train_inverted_index)
        
        normalized_test_q = ' '.join(re.findall(r'\w+', test_q.lower()))
        for train_idx in candidates:
            normalized_train_q = ' '.join(re.findall(r'\w+', train_questions[train_idx].lower()))
            if normalized_test_q == normalized_train_q:
                exact_matches.append({
                    'test_idx': test_idx,
                    'train_idx': train_idx,
                    'text': test_q
                })
        
        # 2. Check for similar matches using LSH
        if not exact_matches or len(exact_matches) == 0:  # Only check similarity if no exact match
            test_minhash = create_minhash(test_q)
            similar_ids = lsh.query(test_minhash)
            
            for similar_id in similar_ids:
                train_idx = int(similar_id.split('_')[1])
                similar_matches.append({
                    'test_idx': test_idx,
                    'train_idx': train_idx,
                    'similarity': 'LSH match',  # Exact similarity score not computed
                    'test_text': test_q,
                    'train_text': train_questions[train_idx]
                })
    
    # Get unique contaminated test indices
    contaminated_test_indices = set(match['test_idx'] for match in exact_matches)
    similar_test_indices = set(match['test_idx'] for match in similar_matches)
    
    stats = {
        'total_train_examples': len(train_questions),
        'total_test_examples': len(test_questions),
        'exact_matches': len(exact_matches),
        'similar_matches': len(similar_matches),
        'unique_contaminated_examples': len(contaminated_test_indices),
        'unique_similar_examples': len(similar_test_indices),
        'contaminated_examples': exact_matches,
        'similar_examples': similar_matches
    }
    
    return stats


def print_contamination_report(contamination_stats):
    """Print a formatted report of contamination statistics and examples with colors and tables."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"📊 Contamination Analysis Summary")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    # Create summary table
    summary_data = [
        ["Total Training Examples", contamination_stats['total_train_examples']],
        ["Total Test Examples", contamination_stats['total_test_examples']],
        ["Exact Matches", f"{Fore.RED}{contamination_stats['exact_matches']}{Style.RESET_ALL}"],
        ["Unique Test Examples with Exact Matches", f"{Fore.RED}{contamination_stats['unique_contaminated_examples']}{Style.RESET_ALL}"],
        ["Similar Matches (LSH)", f"{Fore.YELLOW}{contamination_stats['similar_matches']}{Style.RESET_ALL}"],
        ["Unique Test Examples with Similar Matches", f"{Fore.YELLOW}{contamination_stats['unique_similar_examples']}{Style.RESET_ALL}"]
    ]
    
    print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    # Print exact matches if they exist
    if contamination_stats['contaminated_examples']:
        print(f"\n{Fore.RED}⚠️  Exact Question Matches (Sample){Style.RESET_ALL}")
        exact_matches_data = []
        for i, match in enumerate(contamination_stats['contaminated_examples'][:3]):
            exact_matches_data.append([
                i + 1,
                match['test_idx'],
                match['train_idx'],
                match['text']
            ])
        
        print(tabulate(exact_matches_data,
                      headers=["#", "Test Index", "Train Index", "Question"],
                      tablefmt="fancy_grid"))

    # Print similar matches if they exist
    if contamination_stats['similar_examples']:
        print(f"\n{Fore.YELLOW}📋 Similar Questions (Sample){Style.RESET_ALL}")
        similar_matches_data = []
        for i, match in enumerate(contamination_stats['similar_examples'][:3]):
            similar_matches_data.append([
                i + 1,
                match['test_idx'],
                match['train_idx'],
                match['test_text'][:100] + "..." if len(match['test_text']) > 100 else match['test_text'],
                match['train_text'][:100] + "..." if len(match['train_text']) > 100 else match['train_text']
            ])
        
        print(tabulate(similar_matches_data,
                      headers=["#", "Test Index", "Train Index", "Test Question", "Train Question"],
                      tablefmt="fancy_grid"))

    # Print summary percentages
    print(f"\n{Fore.CYAN}📈 Contamination Percentages:{Style.RESET_ALL}")
    exact_percent = (contamination_stats['unique_contaminated_examples'] / contamination_stats['total_test_examples']) * 100
    similar_percent = (contamination_stats['unique_similar_examples'] / contamination_stats['total_test_examples']) * 100
    
    percentage_data = [
        ["Exact Match Contamination", f"{Fore.RED}{exact_percent:.2f}%{Style.RESET_ALL}"],
        ["Similar Match Contamination", f"{Fore.YELLOW}{similar_percent:.2f}%{Style.RESET_ALL}"],
        ["Clean Examples", f"{Fore.GREEN}{100 - exact_percent - similar_percent:.2f}%{Style.RESET_ALL}"]
    ]
    
    print(tabulate(percentage_data, headers=["Type", "Percentage"], tablefmt="fancy_grid"))


def format_ds(hf_ds):

    hf_ds = hf_ds.map(prepocess_dataset_row)
    hf_ds = hf_ds.map(parse_row)

    return hf_ds

def main(pargs):

    train_ds, test_ds = load_datasets()

    if not pargs.skip_contamination:
        stats = check_dataset_contamination(train_ds, test_ds)
        print_contamination_report(stats)

    # train_ds.to_json(f"{pargs.dest}/train.json")
    # test_ds.to_json(f"{pargs.dest}/test.json")

    formated_train_ds = format_ds(train_ds)
    formated_test_ds = format_ds(test_ds)

    formated_train_ds.to_json(f"{pargs.dest}/train.json")
    formated_test_ds.to_json(f"{pargs.dest}/test.json")



def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest", 
        type=str,
        required=True,
        help="The output destination for the processed dataset"
    )

    parser.add_argument(
        "-s",
        "--skip-contamination",
        action="store_true",
        help="Whether to skip the contamination analysis"
    )

    return parser.parse_args()



if __name__ == "__main__":
    pargs = _parse_args()
    main(pargs)