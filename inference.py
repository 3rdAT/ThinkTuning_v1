# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import json
import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset
from transformers import default_data_collator
from tqdm import tqdm

from modeling_llama import LlamaForCausalLM
import copy
import json

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.nn import functional as F

def main(
    model_name: str="meta-llama/Llama-2-7b-hf",
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 512, #The maximum numbers of tokens to generate
    output_file: str="/data/data/arrv/Think/eval/it2/l2-sft(1).json",
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token='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', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # eval_data = pd.read_csv('./science_prompts_precaution.csv')

    # dataset = load_from_disk("/data/data/arrv/data/gsm8k_mod")
    # dataset = dataset["test"]
    #eval_data = json.load(open("./output_da.json"))

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # model = load_model(model_name, quantization)
    model = LlamaForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-hf',
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        token='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',
    )

    model.eval()

    #sysmsg = """You will be provided with a grade-school math word problem that requires multiple steps to solve. Solve the problem step-by-step and provide the final answer. Your response should include the detailed step-by-step reasoning followed by '####' and the final numeric answer. For example, '<step-by-step reasoning> #### <final_numeric_answer>'. Do not add any other unnecessary content in your response."""
    #user_prompt = f"Context: {data['bing_srch_results']}\n\nGenerate a single factual statement using the given keywords: {data['keywords']}.\nStatement:"
    # user_prompt = f"[INST] <<SYS>>\n{sysmsg}\n<</SYS>>\n\n{q} [/INST]"
    user_prompt = '''I went to a Party'''
    batch = tokenizer(user_prompt, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    # start = time.perf_counter()
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **batch,
    #         # max_new_tokens=max_new_tokens,
    #         do_sample=do_sample,
    #         top_p=top_p,
    #         temperature=temperature,
    #         min_length=min_length,
    #         use_cache=use_cache,
    #         top_k=top_k,
    #         repetition_penalty=repetition_penalty,
    #         length_penalty=length_penalty,
    #         output_hidden_states= True, return_dict_in_generate=True,
    #         num_return_sequences = 5,
    #         **kwargs
    #     )

    outputs = model(**batch)
    
    print(outputs.keys())


if __name__ == "__main__":
    fire.Fire(main)