import json
import os
from src.models.base_model import base_model
from src.tasks.news_details_extraction_task import DetailsExtractionTask
from src.tasks.translation_task import TranslationTask
from src.helper import config


def load_story_by_id(file_path, story_id):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            if data.get("id") == story_id:
                return data
    return None


def get_task_class(task_name):
    task_map = {
        "details_extraction": DetailsExtractionTask,
        "translation": TranslationTask
    }
    return task_map.get(task_name)


def main():
    model_id = config.BASE_MODEL_ID
    device = config.DEVICE
    task_to_run = "translation"
    
    print(f"Evaluating base model: {model_id} on device: {device}")
    
    TaskClass = get_task_class(task_to_run)
    if TaskClass is None:
        raise ValueError(f"Unknown task: {task_to_run}")
    
    story_data = load_story_by_id(config.FILE_PATH_SAMPLE_STORY, config.STORY_ID_TO_EVALUATE)
    if story_data is None:
        raise ValueError(f"Story ID {config.STORY_ID_TO_EVALUATE} not found")
    
    story_id = story_data.get("id", "N/A")
    story_content = story_data["content"]
    
    print("=" * 80)
    print(f"Story ID: {story_id}")
    print("Original Story:")
    print(story_content[:500] + "...")
    print("=" * 80)
    
    model = base_model(model_id, device)
    
    if task_to_run == "translation":
        target_language = config.TARGET_LANGUAGE
        output = TaskClass.run(model, story_content, target_language)
        print(f"Target Language: {target_language}")
    else:
        output = TaskClass.run(model, story_content)
    
    print(f"{task_to_run.replace('_', ' ').title()} Output:")
    print(output)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(config.OUTPUT_DIR, f"{story_id}_{task_to_run}_output.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "story_id": story_id,
            "task": task_to_run,
            "output": output
        }, f, ensure_ascii=False, indent=4)
    
    print(f"Saved output to: {output_file}")


if __name__ == "__main__":
    main()