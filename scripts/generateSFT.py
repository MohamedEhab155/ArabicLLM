from src.utils.data_utils.BuildSTF import BUILD_STF
from src.models.initialize_model import initialize_model  
from .Enums import data, tasks

if __name__ == "__main__":  
    model = initialize_model(model_type="openai_model", device="cuda")
    builder = BUILD_STF(
        path=data.path.value,
        save_path=data.save_path.value,
        task_name=tasks.details_extraction.value,
        model=model
    )
    builder.build_sft_dataset()