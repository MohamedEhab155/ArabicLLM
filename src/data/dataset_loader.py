import json
from os import path
import random

def load_dataset(data_path: str):
    row_data = []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            row_data.append(json.loads(line.strip()))
            random.shuffle(row_data)  # Shuffle the data to ensure randomness
    return row_data


if __name__ == "__main__":
    data_path = r"E:\NLP_Industry_Ready\ArabicLLM - test_kaggle\data\DataSet\news-sample.jsonl"
    dataset = load_dataset(data_path)
    print(dataset[0]['content'])
            

    