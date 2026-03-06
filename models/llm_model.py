import torch
import os
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline

n4f_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

def get_hf_llm(model_name : str = "microsoft/Phi-3.5-mini-instruct",max_new_token = 1024,**kwargs):


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = n4f_config,
        low_cpu_mem_usage = True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipelines = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens = max_new_token,
        pad_token_id = tokenizer.eos_token_id,
        device_map="cuda:1",
        repetition_penalty = 1.15 #Tránh lặp từ
)
    
    llm = HuggingFacePipeline(
        pipeline=model_pipelines,
        model_kwargs=kwargs
    )

    return llm
