from enum import Enum


class data(Enum):
    path = r"E:\NLP_Industry_Ready\ArabicLLM - test_kaggle\data\DataSet\news-sample.jsonl"
    save_path = r"E:\NLP_Industry_Ready\ArabicLLM - test_kaggle\data\DataSet\sft_details_extraction.jsonl"


class tasks(Enum):
    DETAILS_EXTRACTION = "details_extraction"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

class ModelType(Enum):
    BASE_MODEL="base_model"
    OPENAI_MODEL="openai_model"
    FINETUNING_MODEL="finetuning_model"
    