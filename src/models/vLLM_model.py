import requests
from typing import Callable, List, Dict
from transformers import AutoTokenizer
from src.helper import config


class VLLMModel:
    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        model_id: str = "arbic_llm",
        base_model_id: str = config.BASE_MODEL_ID,
        temperature: float = config.TEMPERATURE,
        max_tokens: int = config.MAX_TOKEN
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Tokenizer is only used locally for chat-template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # ------------------------------------------------------------------ #
    def _build_prompt(self, messages: List[Dict]) -> str:
        """Apply HuggingFace chat template to produce a raw string prompt."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def run_task(
        self,
        task_type: str,
        story: str,
        prompt_builder_task: Callable[[str], List[Dict]],
    ) -> dict:
        """
        Execute a task via the vLLM /v1/completions endpoint.

        Returns the raw response dict (same shape as OpenAI completions).
        """
        messages = prompt_builder_task(story)
        if messages is None:
            raise RuntimeError("Prompt builder returned None")

        prompt = self._build_prompt(messages)

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(
            f"{self.endpoint}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()