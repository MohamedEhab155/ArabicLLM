import os
from dotenv import load_dotenv

from .base_model import base_model
from .openai_model import OpenAIModel
from .vLLM_model import VLLMModel          # NEW
from .finetunnig_model import finetuning_model
from src.helper import config

load_dotenv()


def initialize_model(model_type: str, device: str = "cuda"):
   
    if model_type == "base_model":
        return base_model(model_id=config.BASE_MODEL_ID, device=device)

    elif model_type == "openai_model":
        return OpenAIModel(
            endpoint=os.getenv("OPENAI_ENDPOINT"),
            model_name=os.getenv("OPENAI_MODEL_NAME"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    elif model_type == "finetuning_model":
        return finetuning_model(
            config.BASE_MODEL_ID,
            config.FINETUNING_MODEL_ID,
            device=device,
        )

    elif model_type == "vllm_model":             
        return VLLMModel(
            endpoint=os.getenv("VLLM_ENDPOINT", "http://localhost:8000"),
            model_id=os.getenv("VLLM_MODEL_ID", "news-lora"),
            base_model_id=config.BASE_MODEL_ID,
            temperature=float(os.getenv("VLLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("VLLM_MAX_TOKENS", "1000")),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")