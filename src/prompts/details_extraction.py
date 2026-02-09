from Schemas.news import  NewsDetails
import json
def build_prompt(story:str):
    details_extraction_messages = [
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
            story.strip(),
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
    


