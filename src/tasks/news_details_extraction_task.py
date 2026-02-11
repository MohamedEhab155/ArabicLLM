import json
from typing import List, Dict
from src.schemas.news_schema import NewsDetails
def build_details_extraction_prompt(story: str) -> List[Dict]:
  
    messages =[
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data paraser.",
                "You will be provided by an Arabic text associated with a Pydantic scheme.",
                "Generate the ouptut in the same story language.",
                "You have to extract JSON details from text according the Pydantic details.",
                "Extract details as mentioned in text.",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## Story:",
                story['content'].strip(),
                "",

                "## Pydantic Details:",
                json.dumps(
                    NewsDetails.model_json_schema(), ensure_ascii=False
                ),
                "",

                "## Story Details:",
                "```json"
            ])
        }
    ]
    return messages


# Optional: wrap as a Task object for clean interface
class DetailsExtractionTask:
    name = "details_extraction"

    @staticmethod
    def run(model, story: str):
        from src.models.base_model import base_model  # your class
        return model.run_task(DetailsExtractionTask.name, story, build_details_extraction_prompt)
