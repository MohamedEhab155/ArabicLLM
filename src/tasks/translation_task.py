from typing import List, Dict
import json

from src.schemas.translate_schema import TranslatedStory

def build_translation_prompt(story: str, target_language: str = "ar") -> List[Dict]:
    source_language = "english"
    if target_language in ["en", "english"]:    
        target_language = "English"
    elif target_language in ["ar", "arabic"]:
        target_language = "Arabic"

    if story is None or story.strip() == "":    
        raise ValueError("Story is empty or None")
    
    if target_language is None or target_language.strip() == "":
        raise ValueError("Target language is empty or None")
    
    target_language = target_language.strip().lower()
   
    messages =[
    {
        "role": "system",
        "content": "\n".join([
            "You are a professional translator.",
            "You will be provided by an {} text.",
            "You have to translate the text into the `Targeted Language`.",
            "Follow the provided Scheme to generate a JSON",
            "Do not generate any introduction or conclusion."
        ]).format(source_language)
    },
    {
        "role": "user",
        "content":  "\n".join([
            "## Story:",
            story.strip(),
            "",

            "## Pydantic Details:",
            json.dumps( TranslatedStory.model_json_schema(), ensure_ascii=False ),
            "",

            "## Targeted Language:",
            target_language,
            "",

            "## Translated Story:",
            "```json"

        ])
    }
]
    return messages


class TranslationTask:
    name = "translation"

    @staticmethod
    def run(model, story: str, target_language: str = "ar") -> str:
        prompt_builder = lambda s: build_translation_prompt(s, target_language)
        return model.run_task(TranslationTask.name, story, prompt_builder)
