BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNING_MODEL_ID= r"/content/drive/MyDrive/models"
DEVICE = "cuda" 
TORCH_DTYPE = None

DATA_DIR=r"E:\NLP_Industry_Ready\ArabicLLM - test_kaggle\data"
#FILE_PATH_SAMPLE_STORY = r"/kaggle/working/ArabicLLM---test_kaggle/data/row/stories_sample.jsonl" 
FILE_PATH_SAMPLE_STORY = r"data/row/stories_sample.jsonl"
OUTPUT_DIR = r"/content/ArabicLLM---test_kaggle/outputs"
STORY_ID_TO_EVALUATE:int=1
TARGET_LANGUAGE="ar"

TEMPERATURE=0.3
MAX_TOKEN=1000