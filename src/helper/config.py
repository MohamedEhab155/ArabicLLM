BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = "/gdrive/MyDrive/youtube-resources/llm-finetuning"
DEVICE = "cuda" 
TORCH_DTYPE = None

#FILE_PATH_SAMPLE_STORY = r"/kaggle/working/ArabicLLM---test_kaggle/data/row/stories_sample.jsonl" 
FILE_PATH_SAMPLE_STORY = r"data/row/stories_sample.jsonl"
OUTPUT_DIR = r"/kaggle/working/ArabicLLM---test_kaggle/outputs"
STORY_ID_TO_EVALUATE:int=3
TARGET_LANGUAGE="ar"

TASK_MAPPING = [
    "details_extraction",
    "translation",
    "summarization"
]