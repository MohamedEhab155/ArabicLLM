from src.tasks.news_details_extraction_task import build_details_extraction_prompt
from src.tasks.translation_task import build_translation_prompt
from src.utils.data_utils.dataset_loader import load_dataset
from src.models.initialize_model import initialize_model
from tqdm import tqdm
import json
import logging
from json_repair  import json_repair
from src.schemas.news_schema import NewsDetails
class BUILD_STF:
    def __init__(self,path,save_path: str, task_name, model):
        self.save_path = save_path
        self.task_name = task_name
        self.model = model
        self.logger=logging.getLogger(__name__)
        if self.task_name == "details_extraction":
            self.build_prompt_fn = build_details_extraction_prompt
            self.data_raw = load_dataset(path)
            self.outputScheama = NewsDetails
        elif self.task_name == "translation":
             #self.build_prompt_fn = build_translation_prompt
             #self.data_raw = load_dataset(r"path_to_translation_data")
             pass
    
    def build_sft_dataset(self):

        price_per_1m_input_tokens = 0.150
        price_per_1m_output_tokens = 0.600

        prompt_tokens = 0
        completion_tokens = 0

        for data in tqdm(self.data_raw):

            messages = self.build_prompt_fn(data["content"].strip())
    
            response = self.model.run_task(self.task_name, data["content"].strip(), self.build_prompt_fn)

            choice = response.choices[0]

            if choice.finish_reason != "stop":
                prompt_tokens += response.usage.prompt_tokens
                continue

            llm_response = choice.message.content
            llm_resp_dict = self.parse_json(llm_response)

            if not llm_resp_dict:
                continue

            record = {
                "id": data["id"],
                "story": data["content"].strip(),
                "task": self.task_name,
                "output_scheme": json.dumps(
                    self.outputScheama.model_json_schema(),
                    ensure_ascii=False
                ),
                "response": llm_resp_dict
            }

            with open(self.save_path, "a", encoding="utf8") as dest:
                dest.write(
                    json.dumps(record, ensure_ascii=False, default=str) + "\n"
                )

            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            if data["id"] % 3 == 0:
                cost_input = (prompt_tokens / 1_000_000) * price_per_1m_input_tokens
                cost_output = (completion_tokens / 1_000_000) * price_per_1m_output_tokens
                total_cost = cost_input + cost_output

                self.logger.info(f"Iteration {data['id']}: Total Cost = ${total_cost:.4f}")
            
    def parse_json(self, text):
        try:
            return json_repair.loads(text)
        except:
            return None
       