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

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import copy
import json

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.nn import functional as F

def main(
    model_name: str="meta-llama/Llama-2-7b-chat-hf",
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 4096, #The maximum numbers of tokens to generate
    output_file: str="/data/data/arrv/Think/eval/it2/l2-sft(1).json",
    seed: int=42, #seed value for reproducibility
    do_sample: bool=False, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=0.8, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.chat_template = ("{% if messages[0]['role'] == 'system' %}"
    #                         "{% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}"
    #                         "{% set messages = messages[1:] %}"
    #                     "{% else %}"
    #                         "{% set system_message = '' %}"
    #                     "{% endif %}"

    #                     "{% for message in messages %}"
    #                         "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    #                             "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    #                         "{% endif %}"

    #                         "{% if loop.index0 == 0 %}"
    #                             "{% set content = system_message + message['content'] %}"
    #                         "{% else %}"
    #                             "{% set content = message['content'] %}"
    #                         "{% endif %}"

    #                         "{% if message['role'] == 'user' %}"
    #                             "{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}"
    #                         "{% elif message['role'] == 'assistant' %}"
    #                             "{{ ' ' + content | trim + ' ' + eos_token }}"
    #                         "{% endif %}"
    #                     "{% endfor %}")
    
    tokenizer.chat_template = ("{% if messages[0]['role'] == 'system' %}"
                        "{% set offset = 1 %}"
                    "{% else %}"
                        "{% set offset = 0 %}"
                    "{% endif %}"

                    "{{ bos_token }}"
                    "{% for message in messages %}"
                        "{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}"
                            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
                        "{% endif %}"

                        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}"
                    "{% endfor %}"

                    "{% if add_generation_prompt %}"
                        "{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}"
                    "{% endif %}"
                    )

    # eval_data = pd.read_csv('./science_prompts_precaution.csv')

    # dataset = load_from_disk("/data/data/arrv/data/gsm8k_mod")
    # dataset = dataset["test"]
    #eval_data = json.load(open("./output_da.json"))

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    # model = load_model(model_name, quantization)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        token='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',
    )

    model.eval()
    # tokenizer = LlamaTokenizer.from_pretrained(model_name, token='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')
    # tokenizer.pad_token = tokenizer.eos_token
    data_with_prediction = {}

    #sysmsg = """You will be provided with a grade-school math word problem that requires multiple steps to solve. Solve the problem step-by-step and provide the final answer. Your response should include the detailed step-by-step reasoning followed by '####' and the final numeric answer. For example, '<step-by-step reasoning> #### <final_numeric_answer>'. Do not add any other unnecessary content in your response."""
    #user_prompt = f"Context: {data['bing_srch_results']}\n\nGenerate a single factual statement using the given keywords: {data['keywords']}.\nStatement:"
    # user_prompt = f"[INST] <<SYS>>\n{sysmsg}\n<</SYS>>\n\n{q} [/INST]"
    # user_prompt = ''''''
    # user_prompt = '''[Prefix]
    # Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    # A: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working
    # [Suffix] 
    # 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. So the answer is 10
    # [Thought] 
    # I need to multiply the amount earned per minute by the number of minutes worked

    # [Prefix] 
    # Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    # A:Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold
    # [Suffix] 
    # 48+24 = <<48+24=72>>72 clips altogether in April and May. The final answer is 72.
    # [Thought] 
    # I need to add the clips sold in April and May
    
    
    # [Prefix]
    # Q: A rectangle was created by arranging 6 100-won coins horizontally and 4 vertically. How much amount of money is when you sum up the coins placed around the circumference?
    # A: To find the amount of money around the circumference of the rectangle, we need to count the coins along the edges without counting the corners twice.\n\nThere are 6 coins along the horizontal edge, but we only count the 4 in between the corners for each of the two horizontal edges, which gives us 4 + 4 = 8 coins.
    # [Suffix]
    # There are 4 coins along the vertical edge, but we only count the 2 in between the corners for each of the two vertical edges, which gives us 2 + 2 = 4 coins. Now we add the corners. There are 4 corners, with one coin at each corner, so we have 4 coins. Adding these up, we have 8 (horizontal in-between) + 4 (vertical in-between) + 4 (corners) = 16 coins. Since each coin is worth 100 won, the total amount of money around the circumference is 16 coins * 100 won/coin = 1600 won.
    # [Thought]'''

    user_prompt = "[prefix]\nQ: Seunghwa took an express bus and a general bus to travel to Chuncheon, and arrived in Chuncheon, 120 kilometers (km) away, in 1 hour and 50 minutes. At first, she used the express bus, and later took the general bus for 1 hour and 10 minutes. This general bus used 6 liters of gasoline to travel 40.8 kilometers, and if Seunghwa used 14 liters of gasoline while taking the general bus, how many kilometers (km) Seunghwa moved by express bus for 1 minute? (The time and distance to transfer from the express bus to the bus are not considered.)\nA: First, let's calculate the total distance Seunghwa traveled on the general bus using the given information about gasoline consumption. The general bus used 6 liters of gasoline to travel 40.8 kilometers. Seunghwa used 14 liters of gasoline on the general bus. To find out how far she traveled on the general bus, we can set up a proportion: 6 liters / 40.8 km = 14 liters / x km Now, solve for x: x = (14 liters * 40.8 km) / 6 liters x = (571.2 km·liters) / 6 liters x = 95.2 km So, Seunghwa traveled 95.2 kilometers on the general bus. Now, let's find out the distance she traveled on the express bus. The total distance to Chuncheon is 120 kilometers. If she traveled 95.2 kilometers on the general bus, then the remaining distance must have been covered by the express bus: Distance by express bus = Total distance - Distance by general bus Distance by express bus = 120 km - 95.2 km Distance by express bus = 24.8 km Now, we need to\n[suffix]\n find out how many kilometers Seunghwa moved by express bus for 1 minute. We know the total time of the trip was 1 hour and 50 minutes, and she spent 1 hour and 10 minutes on the general bus. So, the time spent on the express bus is: Total time - Time on general bus = Time on express bus 1 hour and 50 minutes - 1 hour and 10 minutes = 40 minutes Now, we can calculate the distance traveled per minute on the express bus: Distance per minute on express bus = Distance by express bus / Time on express bus (in minutes) Distance per minute on express bus = 24.8 km / 40 minutes Distance per minute on express bus = 0.62 km/minute Therefore, Seunghwa moved 0.62 kilometers per minute on the express bus."
    
    messages = [
        {
            "role":"system",
            "content": "Generate a thought one would think in between the [prefix] and [suffix]"
        },
        {
            "role":"user",
            "content":f"{user_prompt}.\nFormat your answer between [thought] and [/thought]"
        },
    ]

    batch = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    batch = tokenizer(batch, return_tensors="pt", add_special_tokens=False)
    prompt_len = len(batch['input_ids'][0])
    print(prompt_len)
    batch = {k: v.to("cuda") for k, v in batch.items()}

    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=4096, #restricts the number new tokens to be generated. I have kept it to the len(target_prompt_tokens)-1 (1 coz of having a <s> token)
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            min_length=None,
            use_cache=True,
            top_k=50,
            repetition_penalty=1.0,
            length_penalty=1,
            output_hidden_states= True, return_dict_in_generate=True,
            output_scores=True,
        )
    output_tokens = outputs.sequences[0][prompt_len:]

    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

    print(output_text)

if __name__ == "__main__":
    fire.Fire(main)