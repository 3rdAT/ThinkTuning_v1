from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataclasses import dataclass
from transformers.utils import ModelOutput

from transformers.utils import (
    is_torchdynamo_compiling
)

from pack_input_ids import get_packed_inputs

import ipdb


@dataclass
class ThinkTunerOutputs(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    nll_thought: Optional[torch.FloatTensor] = None
    reinforce_loss: Optional[torch.FloatTensor] = None
    gate_loss: Optional[torch.FloatTensor] = None
    sampled_thought:Optional[List] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def calculate_unreduced_loss(logits, labels, vocab_size):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    unreduced_loss = loss_fct(shift_logits, shift_labels)
    # ipdb.set_trace()
    total_nll_loss = unreduced_loss.mean() #THis is for the batch_size entirely [3, n] [3]/[1]
    unreduced_loss = unreduced_loss.view(logits.shape[0], -1)
    
    return unreduced_loss, total_nll_loss

def think_loss(zz, idx, new_hidden_states, labels, packed_reasoning_path, model, gate_values, packed, unreduced_loss):
    
    gate_loss = 0.0
    reinforce_loss = 0.0
    nll_loss_thought = 0.0
    reward_signals = []
    nll_signals = []

    num_logits_to_keepy = 0
    if model.config.pretraining_tp > 1:
        lm_head_slices = model.lm_head.weight.split(model.vocab_size // model.config.pretraining_tp, dim=0)
        new_logits = [F.linear(new_hidden_states, lm_head_slices[i]) for i in range(model.config.pretraining_tp)]
        new_logits = torch.cat(new_logits, dim=-1)
    else:
        if labels is None and not is_torchdynamo_compiling():
            pass
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        new_logits = model.lm_head(new_hidden_states[:, -num_logits_to_keepy:, :]).float()

    packed_loss = None
    packed_labels = packed_reasoning_path
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    packed_logits = new_logits.float()
    # Shift so that tokens < n predict n
    shift_logits = packed_logits[..., :-1, :].contiguous()
    shift_labels = packed_labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none")
    shift_logits = shift_logits.view(-1, model.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    packed_loss = loss_fct(shift_logits, shift_labels)

    packed_loss = packed_loss.view(packed_logits.shape[0], -1)

        #Note: If len(input_ids) = n, the length of unreduced loss is n-1 (Reason, you don't want to learn to predict anything from the last token).
        # input_ids = [<s>, Hi, I, am, doing, TODAY, I, WENT, TO, A, MOVIE, really, well, </s>, <s>, Hi, I, am, doing, TODAY, I, HAD, AN, ACCIDENT, really, bad, </s>]'
                                            # |(<SoT>_index)         |<EoT>_index         |</s_index>                                     |<EoT>_index           |</s_index>

    # ipdb.set_trace()
    for index, batch in enumerate(new_hidden_states):
        configs = packed[f'{index}'] #{'0': [{'thought_no': 0, 'start_index': 0, 'thought_start_index': 97, 'thought_end_index': 130, 'end_index': 141}, {'thought_no': 1, 'start_index': 141, 'thought_start_index': 238, 'thought_end_index': 339, 'end_index': 350}], '1': [{'thought_no': 2, 'start_index': 0, 'thought_start_index': 97, 'thought_end_index': 198, 'end_index': 209}]}
        for indi, thought in enumerate(configs):  #configs = [{'thought_no': 0, 'start_index': 0, 'thought_start_index': 97, 'thought_end_index': 130, 'end_index': 141}, {'thought_no': 1, 'start_index': 141, 'thought_start_index': 238, 'thought_end_index': 339, 'end_index': 350}]
            # print(indi)
            nll_signal = torch.tensor([(model.reward_decay ** i+1) * loss for i, loss in enumerate(packed_loss[index][thought['thought_end_index']:thought['end_index']])]).to(device=unreduced_loss.device)
            reward_signal = (unreduced_loss[zz, idx:] - nll_signal).detach()
            # reward_signal = (unreduced_loss[idx:] - packed_loss[index][thought['thought_end_index']:thought['end_index']]).detach()
            nll_signal = torch.mean(nll_signal)
            reward_signal = torch.mean(reward_signal)
            reward_signals.append(reward_signal)
            nll_signals.append(nll_signal)


    # based on the difference between the reward_signal and average reward from the bunch of sampled rationales (to reduce variance)
    # We basically Sub
    reward_signals_tensor = torch.tensor(reward_signals)
    mean_reward_signals = torch.mean(reward_signals_tensor)

    gate_loss += -mean_reward_signals*gate_values[zz][idx]#Connect the computational graph from subsampling to the graph with gating mechanism

    if mean_reward_signals > 0:
        r_ind = 0
        for index, batch in enumerate(new_hidden_states):
            configs = packed[f'{index}']
            for indi, thought in enumerate(configs):
                var_difference = torch.clamp((reward_signals[r_ind] - mean_reward_signals), min=0)
                t_likelihood_loss = torch.mean(packed_loss[:thought['thought_end_index']])
                reinforce_loss += var_difference * t_likelihood_loss
                nll_loss_thought += var_difference * nll_signals[index]
                r_ind+=1
        nll_loss_thought = nll_loss_thought / len(new_hidden_states)

    return gate_loss, reinforce_loss, nll_loss_thought


def start_thinking(input_ids, hidden_states, logits, labels, unreduced_loss, model, tokenizer):
    new_sequence = input_ids.clone()
    total_gate_loss = 0.0
    total_nll_thought = 0.0
    total_reinforce_loss = 0.0
    logy = []
    reinforce_loss = 0.0
    gate_loss = 0.0
    nll_loss_thought = 0.0
    model.gate = model.gate.to(hidden_states.device)
    gate_values = model.gate(hidden_states)
    gate_values = gate_values.squeeze(-1)
    
    selected_indices = []
    for row in gate_values:
        indices = torch.nonzero(row > 0.5).squeeze()
        print("The indices are:",indices)
        try:
            if len(indices) >= 3:
                selected_indices.append(indices[torch.randperm(len(indices))[:3]].tolist())
            else:
                selected_indices.append(indices.tolist())
        except:
            selected_indices.append(indices.tolist())

    print("The selected-indices are:",selected_indices)
    for zz in range(len(input_ids)):
        temp = {}
        ## gate_values [b, n]
        temp[f"batch-{zz}"] = []
        for idx, aK in enumerate(selected_indices[zz]):
            value = gate_values[zz][aK]
            temp1 = {}
            if idx == len(gate_values[zz])-1:
                break
            if value <= 0.5:
                temp1[f"{idx}"] = ["Nothing"]
                temp[f"batch-{zz}"].append(temp1)
                pass
            else:
                temp1[f"{idx}"] = []
                
                #logic for thought generation and computing thought influenced logits
                logits_of_token = logits[zz][idx]
                probabilities = F.softmax(logits_of_token, dim=-1)  # Softmax across the last dimension
                
                # Get the top-k values and their corresponding indices
                top_k = 10  # Example: Get the top 5 tokens
                topk_indices = torch.topk(probabilities, top_k, dim=-1)
        
                reasoning_path = []
                thought_index = []
                for i in range(top_k):
                    if i < 7:
                        continue
                    # TODO: Append the <SoT> token to the start
                    new_sequence_id = torch.cat([new_sequence[zz][:idx+1], torch.Tensor([]).to(device=new_sequence.device)], dim=0)
                    # Append the sampled top-k token
                    sampled_token = torch.tensor(topk_indices.indices[i].unsqueeze(0)).to(device=new_sequence.device)  # Add the sampled token
                    new_sequence_id = torch.cat([new_sequence_id, sampled_token], dim=0)

                    new_attention_mask = torch.ones(len(new_sequence_id)).unsqueeze(0).to(dtype=torch.long, device=new_sequence_id.device)

                    batch = {'input_ids': new_sequence_id.unsqueeze(0).to(dtype=torch.long), 'attention_mask': new_attention_mask}
                    new_greedy_sequence_decoding = model.generate(**batch, max_new_tokens=10, do_sample=True, temperature=1.0 ,use_cache= True, top_p=1.0)

                    thought_index.append({
                        'thought_start': idx+1,
                        'thought_end': idx+1 + len(new_greedy_sequence_decoding[0][idx+1:])
                    })
                    reasoning_path.append(torch.cat([input_ids[zz][: idx+1], new_greedy_sequence_decoding[0][idx+1:], input_ids[zz][idx+1:]], dim=-1))
                    sampled_text_generate = {'seq':tokenizer.decode(reasoning_path[-1]), 't_len':len(reasoning_path[-1])}
                    temp1[f"{idx}"].append(sampled_text_generate)
                # reasoning_path shape: [topk_idx, batch_size, seq_len]
                temp[f"batch-{zz}"].append(temp1)


                # Single sample in a batch
                packed_reasoning_path, packed_attention_mask, packed_reasoning_path_casual_mask, packed = get_packed_inputs(reasoning_path, max_length=4090, pad_token_id=model.config.eos_token_id, thought_index=thought_index)
                new_hidden_states = model.model(
                    input_ids=packed_reasoning_path,
                    attention_mask=packed_reasoning_path_casual_mask
                )

                packed_reasoning_path = packed_reasoning_path.to(device='cpu')
                packed_attention_mask = packed_attention_mask.to(device='cpu')
                packed_reasoning_path_casual_mask = packed_reasoning_path_casual_mask.to(device='cpu')

                #Assuming I have access to the indices for the corresponding inputs in the packed batch and let it be stored in packed = {'batch_no':,'</s>':,'<SoT>':,'<EoT>':}
                # I need a config of the following kind:
                # packed = [{'configs': [{'rationale_no': r_no, '<s>_index': st_no, '<SoT>_index': sot_idx, '<EoT>_index': eot_idx, '</s>_index':ed_no},{...}]}]

                new_hidden_states = new_hidden_states[0]  # Feed this to a gating mechanism

                gate_loss_i, reinforce_loss_i, nll_loss_thought_i = think_loss(zz, idx, new_hidden_states, labels, packed_reasoning_path, model, gate_values, packed, unreduced_loss)
                gate_loss += gate_loss_i
                reinforce_loss += reinforce_loss_i
                nll_loss_thought += nll_loss_thought_i

            # ipdb.set_trace()
        #Sample-wise normalization
        # count = (gate_values[zz] > 0.5).sum().item()
        count = 3
        total_gate_loss += gate_loss / count
        total_nll_thought += nll_loss_thought / count
        total_reinforce_loss += reinforce_loss / count

        logy.append(temp)

    #Batch-wise normalization
    total_gate_loss = total_gate_loss / len(input_ids)
    total_reinforce_loss = total_reinforce_loss / len(input_ids)
    total_nll_thought = total_nll_thought / len(input_ids)

    return total_gate_loss, total_reinforce_loss, total_nll_thought, logy

def think_tuner_step(batch, model: AutoModelForCausalLM, tokenizer):
    
    # Initial forward pass
    outputs = model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
    logits = model.lm_head(outputs.last_hidden_state).float()

    # Get unreduced loss
    unreduced_loss, total_nll_loss = calculate_unreduced_loss(logits, batch["labels"], model.config.vocab_size)

    # Start Thinking and get all the losses.
    total_gate_loss, total_reinforce_loss, total_nll_thought, logy = start_thinking(batch["input_ids"], outputs.last_hidden_state, logits,  batch["labels"], unreduced_loss, model, tokenizer)


    return ThinkTunerOutputs(
        loss=total_nll_loss,
        nll_thought=total_nll_thought, 
        reinforce_loss=total_reinforce_loss,
        gate_loss=total_gate_loss,
        sampled_thought=logy,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )