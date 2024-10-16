from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset
import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
from datetime import datetime
import contextlib
from itertools import islice
import dataclasses
import gc


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM
import json
import pickle

from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure

import fire
import sys
import random
from datasets import load_dataset, load_from_disk
import pandas as pd
from transformers import default_data_collator

from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from transformers.data import DataCollatorForSeq2Seq

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    ModelOutput,
    replace_return_docstrings,
)

from dataclasses import dataclass
from torch.utils.data.dataloader import default_collate

import copy
import datasets
import os
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from collections.abc import Mapping
import wandb
from src.think_tuner import think_tuner_step

import ipdb
import torch


@dataclass
class train_configy:
    model_name: str="meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer_name: str=None
    run_validation: bool=False
    batch_size_training: int=2
    batching_strategy: str="padding" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int= 1
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=1
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=0
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "/scratch/ssaeidi1/aswin/data/train.json"
    output_dir: str = "."
    save_model: bool = True
    use_wandb: bool = False # Enable wandb for experient tracking
    save_metrics: bool = True # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "/scratch/ssaeidi1/aswin/model" # will be used if using profiler
    use_reward_decay: bool = False
    use_best_reward_signal: bool = False
    reward_based_optimization: bool = False
    sample_thought_by_prompt: bool = True

#=========================================================================================================================
###Function to get the preprocessed dataset in trainable format
def get_preprocessed_dataset(tokenizer, data_path):
    # Load dataset from Hugging Face
    # dataset = datasets.load_dataset("kaist-ai/CoT-Collection", split="train[:26000]")
    # dataset = load_from_disk('/scratch/aravik13/Think/data/CoT.parquet')

    dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train[:10000]")

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()  # Split 20% for validation

    def tokenize_add_label(sample):
        
        prompt = [
                    {
                        "role":"user",
                        "content":f"{sample['question']}"
                    },
                    {
                        "role":"assistant",
                        "content":f"{sample['answer']}"
                    },

                ]
        
        # pos_util_assistant = [
        #     {
        #     "role":"assistant",
        #     "content": ""
        #     },
        #    ]
        
        # pos_util_user = [  
        #     {
        #     "role":"user",
        #     "content": ""
        #     },
        #    ]
        
        prompt_tokens = tokenizer.apply_chat_template(prompt, tokenize=True)
        # ipdb.set_trace()

        question_tokens = tokenizer.encode(sample["question"], add_special_tokens=False)
        question_token_length = len(question_tokens)

        answer_tokens = tokenizer.encode(sample["answer"], add_special_tokens=False)
        answer_token_length = len(answer_tokens)

        start_idx_q = 0
        end_idx_q = 0
        for idx in range(len(prompt_tokens) - question_token_length):
    # Compare if a slice of the prompt matches the question tokens
            if prompt_tokens[idx:idx + question_token_length] == question_tokens:
                start_idx_q = idx
                end_idx_q = start_idx_q + question_token_length - 1
                break
        
        start_answer_idx = 0
        end_answer_idx = 0
        for idx in range(len(prompt_tokens) - answer_token_length):
    # Compare if a slice of the prompt matches the answer tokens
            if prompt_tokens[idx:idx + answer_token_length] == answer_tokens:
                start_answer_idx = idx
                end_answer_idx = start_answer_idx + answer_token_length - 1
                break
        
        # ipdb.set_trace()
        sample = {
            "input_ids": prompt_tokens,
            "attention_mask": [1] * (len(prompt_tokens)),
            "labels": prompt_tokens,
            "start_idx_q": start_idx_q,
            "end_idx_q": end_idx_q,
            "start_answer_idx": start_answer_idx,
            "end_answer_idx": end_answer_idx,
        }

        return sample

    # Apply the tokenize_add_label function to each dataset
    train_dataset = train_dataset.map(tokenize_add_label, remove_columns=list(train_dataset.features))
    val_dataset = val_dataset.map(tokenize_add_label, remove_columns=list(val_dataset.features))

    # Filter the datasets
    # train_dataset = train_dataset.filter(filter_large_samples)
    # val_dataset = val_dataset.filter(filter_large_samples)

    return train_dataset, val_dataset

#===================================================================================================================================
###DATA COLLATORS

InputDataClass = NewType("InputDataClass", Any)

def default_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    if return_tensors == "pt":
        return torch_default_data_collator(features)
    # elif return_tensors == "tf":
    #     return tf_default_data_collator(features)
    # elif return_tensors == "np":
    #     return numpy_default_data_collator(features)
    


###Torch Default Data Collator
def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch

    # print(f"The datatype of features is {type(features)}")

    # print(f"The shape of features is:{len(features)}")

    if not isinstance(features[0], Mapping):
        # print("Not a mapping")
        features = [vars(f) for f in features]
    first = features[0]

    # print(f"The first variable is:{first.keys()}")
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        # print("Inside if")
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        # print("Inside elif")

        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)
    # elif "labels_m" in first and first["labels_m"] in 

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            # print("key is not label as well as labels")
            if isinstance(v, torch.Tensor):
                # print("Inside if")
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                # print("Inside elif")
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                # print("Inside else")
                batch[k] = torch.tensor([f[k] for f in features])

    return batch

###===============================================================================================================================
#WANDB MANAGEMENT
@dataclass
class WANDB_CONFIG:
    project: str = 'Thinking' # wandb project name
    entity: Optional[str] = "aravik13" # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
    mode: Optional[str] = None

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_configy):
                print(f"Warning: unknown parameter {k}")


def setup_wandb(train_config, **kwargs):
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    return run

##===================================================================================================================

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None

#======================================================================================================
### TRAIN FUNCTION

def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    train_config.run_validation = False
    local_rank=None
    rank=None

    sampels=[]
    autocast = nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop

    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached

        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_thought_loss = 0.0
            total_reinforce_loss= 0.0
            total_gate_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    # print(f"The keys in the batch are:{batch.keys()}")
                    for key in batch.keys():
                        # if is_xpu_available():
                        #     batch[key] = batch[key].to('xpu:0')
                        # else:
                          batch[key] = batch[key].to(model.device)
                    with autocast():
                        outputs = think_tuner_step(batch, model=model, tokenizer=tokenizer, train_config=train_config)
                        loss = outputs.loss
                        thought_loss = outputs.nll_thought
                        reinforce_loss = outputs.reinforce_loss
                        gate_loss = outputs.gate_loss
                        
                    sampels.append(outputs.sampled_thought)
                    with open(f'sampels1.json', 'w') as f:
                        json.dump(sampels, f)

                    # loss = loss / gradient_accumulation_steps
                    # thought_loss = thought_loss / gradient_accumulation_steps
                    # reinforce_loss = reinforce_loss / gradient_accumulation_steps
                    # gate_loss = gate_loss / gradient_accumulation_steps

                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))

                    
                    # Remember the reinforce loss and thought loss has been already backwarded in the think_tuner_step for efficiency purposes.
                    model_loss = loss
                    
                    # gate_optimizer.zero_grad()
                    # gate_loss.backward() # increasing here
                    # gate_optimizer.step()

                    optimizer.zero_grad()
                    model_loss.backward()
                    optimizer.step()

                    # ipdb.set_trace()
                    total_loss += loss.cpu().detach().item() if isinstance(loss, torch.Tensor) else loss
                    total_thought_loss += thought_loss.cpu().detach().item() if isinstance(thought_loss, torch.Tensor) else thought_loss
                    total_reinforce_loss += reinforce_loss.cpu().detach().item() if isinstance(reinforce_loss, torch.Tensor) else reinforce_loss
                    total_gate_loss += gate_loss.cpu().detach().item() if isinstance(gate_loss, torch.Tensor) else gate_loss
                                        
                    # if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    #     if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                    #         torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                    #     optimizer.step()
                    #     optimizer.zero_grad()

                    pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float() if isinstance(loss, torch.Tensor) else loss,
                            'train/thought_loss': thought_loss.detach().float() if isinstance(thought_loss, torch.Tensor) else thought_loss,
                            'train/gate_loss': gate_loss.detach().float() if isinstance(gate_loss, torch.Tensor) else gate_loss,
                            'train/reinfroce_loss': reinforce_loss.detach().float() if isinstance(reinforce_loss, torch.Tensor) else reinforce_loss,
                        })
                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if step % 100 == 0 and step != 0:
                        if train_config.run_validation:
                            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                            if train_config.save_metrics:
                                val_step_loss.extend(temp_val_loss)
                                val_step_perplexity.extend(temp_step_perplexity)

                            checkpoint_start_time = time.perf_counter()

                        if train_config.save_model :
                            epoch_dir = os.path.join(train_config.output_dir, f"epoch{epoch}", f"step{step}")
                            os.makedirs(epoch_dir, exist_ok=True)
                            # Save the model in the new directory
                            model.save_pretrained(epoch_dir)
                            print(f"Model is saved in {epoch_dir} directory")
                            
                        model.train()

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                    # if step>=50:
                    #     return results
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        memtrace.print_stats()
        lr_scheduler.step()

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()

            if train_config.save_model :
                epoch_dir = os.path.join(train_config.output_dir, f"epoch{epoch}",f"step{step}")
                os.makedirs(epoch_dir, exist_ok=True)
                # Save the model in the new directory
                model.save_pretrained(epoch_dir)
                print(f"Model is saved in {epoch_dir} directory")

            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    return results

##===================================================================================================================================================================

###Evaluation

def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_loss_f = 0.0
    eval_loss_m = 0.0
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                break
            for key in batch.keys():
                # if is_xpu_available():
                #     batch[key] = batch[key].to('xpu:0')
                # else:
                batch[key] = batch[key].to(model.device)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )
# Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
# Print evaluation metrics
    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)
    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

###===============================================================================================================================================================
 ##Training Params saving

def save_train_params(train_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    # fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    # train_config.dist_checkpoint_root_folder
    # + "/"
    # + train_config.dist_checkpoint_folder
    # + "-"
    train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)

###=================================================================================================================================================================
## ConcatDataset:

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "start_idx_q": [],
            "end_idx_q": [],
            "start_answer_idx": [],
            "end_answer_idx": [],
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
    
###==============================================================================================================================================

##Samplers

def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
        kwargs = {}
        batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
        if train_config.batching_strategy == "padding":
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
            kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        elif train_config.batching_strategy == "packing":
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = default_data_collator
        else:
            raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

        return kwargs

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)
        
class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
            )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas
    
###=========================================================================================================================================

def main(**kwargs):

    train_config = train_configy()
    update_config((train_config), **kwargs)

    wandn_run = None

    torch.cuda.manual_seed(0)
    torch.manual_seed(0)
    

    # wandb_run = setup_wandb(train_config, **kwargs)

    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        device_map='auto',
        use_cache=None,
        attn_implementation=None,
        token='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',
    )
    
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, token = "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")
    tokenizer.pad_token_id = tokenizer.eos_token_id

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

    # If there is a mismatch between tokenizer vocab size and embedding matrix, 
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    # if is_xpu_available():
    #     model.to("xpu:0")
    # elif torch.cuda.is_available():
    #     model.to("cuda:0")

     # Load and preprocess the dataset for training and validation

    dataset_train, dataset_val = get_preprocessed_dataset(
        tokenizer,train_config.dataset
    )

    print(type(dataset_train))

    print(f"--> Training Set Length = {len(dataset_train)}")
    print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # # #Initialize the gate if it is not present.
    # if not hasattr(model, 'gate'):
    # # Initialize model.gate as a learnable parameter
    #     print("The gate was not present, so initializing it!")
    #     model.gate = nn.Sequential(nn.Linear(model.config.hidden_size, 1), nn.Sigmoid())
    #     model.reward_decay = 0.9

    model.reward_decay = 0.9

    # # All model parameters
    # all_params = list(model.parameters())

    # # Gate parameters
    # gate_params = list(model.gate.parameters())

    # # Subtract gate parameters from all model parameters
    # # Subtract gate parameters from all model parameters using their ids
    # main_params = [param for param in all_params if id(param) not in {id(gate_param) for gate_param in gate_params}]

    # Define the optimizer for the main parameters (excluding self.gate)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=train_config.lr, 
        weight_decay=train_config.weight_decay
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # gate_optimizer = optim.AdamW(
    #         model.gate.parameters(),
    #         lr=train_config.lr,
    #         weight_decay=train_config.weight_decay,
    # )
    # gate_scheduler = StepLR(gate_optimizer, step_size=1, gamma=train_config.gamma)


    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        # wandb_run
    )
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
    # for k,v in results.items():
    #     wandb_run.summary[k] = v

if __name__ == "__main__":
    fire.Fire(main)