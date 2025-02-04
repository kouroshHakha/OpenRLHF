
from datasets import load_dataset
import re
import os

data_source = 'openai/gsm8k'

dataset = load_dataset(data_source, 'main')


train_dataset = dataset['train']
test_dataset = dataset['test']

instruction_following = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."



def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

# add a row to each data item that represents a unique id
def make_map_fn(split):

    def process_fn(example, idx):
        question_raw = example.pop('question')


        answer_raw = example.pop('answer')
        solution = extract_solution(answer_raw)
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": instruction_following},
                {"role": "user", "content": question_raw}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'answer': answer_raw,
                "question": question_raw,
            }
        }
        return data

    return process_fn


train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

local_dir = "/home/ray/default/data/gsm8k_qwen_parquet"

train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

breakpoint()
