from .base_model import base_model
from .openai_model import OpenAIModel
from src.helper import config
import os
from dotenv import load_dotenv

load_dotenv()

def initialize_model(model_type: str, device: str):
    """Initialize the appropriate model based on type."""
    if model_type == "base_model":
        return base_model(model_id=config.BASE_MODEL_ID, device=device)
    elif model_type == "openai_model":
        return OpenAIModel(
            endpoint=os.getenv("OPENAI_ENDPOINT"),
            model_name=os.getenv("OPENAI_MODEL_NAME"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")