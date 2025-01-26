

from datasets import load_dataset
from transformers import AutoTokenizer

chat_template_id = "Qwen/Qwen2.5-Math-7B-Instruct"

ds = load_dataset("svc-huggingface/math-prompts")

system_message = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."

tokenizer = AutoTokenizer.from_pretrained(chat_template_id)

def apply_fn(row):

    conv = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": row["question"]},
    ]
    ret =  {"input": tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)}
    return ret

ds = ds.map(apply_fn)
ds["train"].to_json("/home/ray/default/data/math_train_data_processed_with_qwen_prompt.json")
ds["test"].to_json("/home/ray/default/data/math_test_data_processed_with_qwen_prompt.json")

breakpoint()

