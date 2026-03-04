from typing import Callable, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

class finetuning_model:
    from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class finetuning_model:
    def __init__(self, base_model_id, finetuning_model_id, device="cuda"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto",
            device_map="auto"
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            finetuning_model_id
        )

        self.model.to(device)

    def run_task(self, task_type, story: str, prompt_builder_task: Callable[[str], List[Dict]]) -> str:
        messages = prompt_builder_task(story)
        if messages is None:
            raise RuntimeError("Prompt builder returned None")

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs_model = self.tokenizer([text], return_tensors='pt').to(self.device)

        outputs_model = self.model.generate(
            inputs_model.input_ids,
            do_sample=False,
            max_new_tokens=1024
        )

        output = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs_model.input_ids, outputs_model)
        ]

        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return response

