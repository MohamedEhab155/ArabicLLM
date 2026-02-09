from transformers import AutoModelForCausalLM , AutoTokenizer
from helper import config
from prompts.details_extraction import build_prompt
def evaluate_base_details_extraction(story: str):
    model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_ID)
    tokenzier=AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
    device=config.DEVICE

    messages=build_prompt(story)
    text=tokenzier.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs_model=tokenzier([text],return_tensors='pt').to(device)

    outputs_model=model.generate(inputs_model.input_ids,
                                   do_sample=False, top_k=None, temperature=None, top_p=None,
                                    max_new_tokens=1024) # output_model= input_model + generated_ids 

    
    output=[
        output_ids[len(input_ids):]
        for input_ids,output_ids in zip(inputs_model.input_ids,outputs_model)     
    ]

    response=tokenzier.batch_decode(output,skip_special_tokens=True)[0]
    return response