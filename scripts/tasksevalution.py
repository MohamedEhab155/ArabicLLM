import json
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from src.models.base_model import base_model
from src.models.openai_model import OpenAIModel
from src.models.finetunnig_model import finetuning_model
from src.tasks.news_details_extraction_task import DetailsExtractionTask
from src.tasks.translation_task import TranslationTask
from src.helper import config
from .Enums import ModelType ,tasks

load_dotenv("src/.env")


def load_story_by_id(file_path: str, story_id: int) -> Optional[Dict[str, Any]]:
    """Load a story from JSONL file by its ID."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            if data.get("id") == story_id:
                return data
    return None


def get_task_class(task_name: str):
    """Map task name to corresponding task class."""
    task_map = {
        "details_extraction": DetailsExtractionTask,
        "translation": TranslationTask
    }
    return task_map.get(task_name)


def initialize_model(model_type: str, device: str):
    """Initialize the appropriate model based on type."""
    if model_type ==  ModelType.BASE_MODEL :
        return base_model(model_id=config.BASE_MODEL_ID, device=device)
    elif model_type == ModelType.OPENAI_MODEL.value:
        return OpenAIModel(
            endpoint=os.getenv("OPENAI_ENDPOINT"),
            model_name=os.getenv("OPENAI_MODEL_NAME"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_type==ModelType.FINETUNING_MODEL.value:
        return finetuning_model(config.BASE_MODEL_ID,config.FINETUNING_MODEL_ID,device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_output(story_id: int, task_name: str, model_type: str, output: str) -> None:
    """Save task output to JSON file."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(
        config.OUTPUT_DIR, 
        f"{story_id}_{task_name}_{model_type}_output.json"
    )
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "story_id": story_id,
            "task": task_name,
            "model": model_type,
            "output": output
        }, f, ensure_ascii=False, indent=4)
    
    print(f"Saved output to: {output_file}")


def run_task(model, task_class, story_content: str, task_name: str, target_language: str = None) -> str:
    """Execute the specified task on the story content."""
    if task_name == "translation":
        print(f"Target Language: {target_language}")
        return task_class.run(model, story_content, target_language)
    else:
        return task_class.run(model, story_content)


def main():

    task_name = tasks.TRANSLATION.value
    model_type= ModelType.FINETUNING_MODEL.value
    device = config.DEVICE
    target_language = 'en'

    print(f"Initializing model: {model_type}")
    model = initialize_model(model_type, device)
    
    # Get task class
    task_class = get_task_class(task_name)
    if task_class is None:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Load story
    story_data = load_story_by_id(
        config.FILE_PATH_SAMPLE_STORY, 
        config.STORY_ID_TO_EVALUATE
    )
    if story_data is None:
        raise ValueError(f"Story ID {config.STORY_ID_TO_EVALUATE} not found")
    
    story_id = story_data.get("id", "N/A")
    story_content = story_data["content"]
    
    # Display story preview
    print("=" * 80)
    print(f"Story ID: {story_id}")
    print(f"Task: {task_name}")
    print(f"Model: {model_type}")
    print("=" * 80)
    print("Story Preview:")
    print(story_content[:500] + "...")
    print("=" * 80)
    
    # Run task
    output = run_task(model, task_class, story_content, task_name, target_language)
    
    # Display output
    print(f"\n{task_name.replace('_', ' ').title()} Output:")
    print(output)
    print("=" * 80)
    
    # Save results
    save_output(story_id, task_name, model_type, output)


if __name__ == "__main__":
    main()