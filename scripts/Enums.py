from enum import Enum


class data(Enum):
    path = r"E:\NLP_Industry_Ready\ArabicLLM - test_kaggle\data\DataSet\news-sample.jsonl"
    save_path = r"E:\NLP_Industry_Ready\ArabicLLM - test_kaggle\data\DataSet\sft_details_extraction.jsonl"


class tasks(Enum):
    details_extraction = "details_extraction"
    translation = "translation"
    summarization = "summarization"