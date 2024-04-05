import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          BitsAndBytesConfig, AutoTokenizer)

def LLM(model_path, config):
    llm = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,local_files_only=True,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return llm, tokenizer