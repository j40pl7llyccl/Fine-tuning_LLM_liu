# data_processing.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

def process_func(example, tokenizer, max_length=512):
    input_ids, labels = [], []
    instruction = tokenizer.encode(
        text="\n".join(["<|system|>", "这是今天来的新同学，请各位好好照顾他--班主任", "<|user|>", example["instruction"] + example["input"] + "<|chat AI|>"]).strip() + "\n",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length
    )
    response = tokenizer.encode(
        text=example["output"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_length
    )
    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
    return {"input_ids": input_ids, "labels": labels}

def preprocess_data(data_file, tokenizer, max_length=512):
    df = pd.read_json(data_file)
    ds = Dataset.from_pandas(df)
    tokenized_ds = ds.map(lambda example: process_func(example, tokenizer, max_length), remove_columns=ds.column_names)
    return tokenized_ds
