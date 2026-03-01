from openai import OpenAI
from typing import Callable, List, Dict


class OpenAIModel:
    def __init__(self, endpoint: str, model_name: str, api_key: str):
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
        self.model_name = model_name

    def run_task(self, task_type: str, story: str, prompt_builder_task: Callable[[str], List[Dict]]) -> str:
        messages = prompt_builder_task(story)
        if messages is None:
            raise RuntimeError("Prompt builder returned None")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            
        )
        
        return response