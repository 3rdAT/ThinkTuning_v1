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


def gen_thought_by_prompt(prefix_ids, suffix_ids, model, tokenizer):

    prefix_text, suffix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True), tokenizer.decode(suffix_ids, skip_special_tokens=True)

    user_prompt = f"[prefix]\n{prefix_text}\n[suffix]\n{suffix_text}"
    
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
    batch = {k: v.to(model.device) for k, v in batch.items()}



    with torch.no_grad():
        outputs = model.generate(**batch, max_new_tokens=4096, do_sample=True, temperature=0.8, use_cache=True, top_p=1.0)

    output_tokens = outputs[0][prompt_len:]
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=True)

    try:
        thought_prompt = " --- "+output_text.split("[thought]")[1].split("[/thought]")[0].strip()+" --- "
    except:
        thought_prompt = " --- "+output_text+" --- "
    thought_ids = tokenizer(thought_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    # ipdb.set_trace()

    return thought_ids


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

def think_loss(zz, idx, new_hidden_states, labels, packed_reasoning_path, model, gate_values, packed, unreduced_loss, use_reward_decay=True, use_best_reward_signal=True, reward_based_optimization=False):
    
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
    unreduced_loss_detach = unreduced_loss.detach()
    # ipdb.set_trace()  
    for index, batch in enumerate(new_hidden_states):
        configs = packed[f'{index}'] #{'0': [{'thought_no': 0, 'start_index': 0, 'thought_start_index': 97, 'thought_end_index': 130, 'end_index': 141}, {'thought_no': 1, 'start_index': 141, 'thought_start_index': 238, 'thought_end_index': 339, 'end_index': 350}], '1': [{'thought_no': 2, 'start_index': 0, 'thought_start_index': 97, 'thought_end_index': 198, 'end_index': 209}]}
        for indi, thought in enumerate(configs):  #configs = [{'thought_no': 0, 'start_index': 0, 'thought_start_index': 97, 'thought_end_index': 130, 'end_index': 141}, {'thought_no': 1, 'start_index': 141, 'thought_start_index': 238, 'thought_end_index': 339, 'end_index': 350}]
            # print(indi)
            if reward_based_optimization:
                nll_signal = packed_loss[index][thought['thought_end_index']:thought['end_index']]
                reward_signal = (unreduced_loss_detach[zz, idx:] - packed_loss[index][thought['thought_end_index']:thought['end_index']])
                # ipdb.set_trace()
            else:
                if use_reward_decay:
                    decay_factors = model.reward_decay ** torch.arange(1, len(packed_loss[index][thought['thought_end_index']:thought['end_index']]) + 1, device=unreduced_loss.device)
                    nll_signal = decay_factors * packed_loss[index][thought['thought_end_index']:thought['end_index']]
                    #nll_signal = torch.tensor([(model.reward_decay ** i+1) * loss for i, loss in enumerate(packed_loss[index][thought['thought_end_index']:thought['end_index']])]).to(device=unreduced_loss.device)
                    # ipdb.set_trace()
                    reward_signal = (unreduced_loss_detach[zz, idx:] - nll_signal).detach()
                else:
                    nll_signal = packed_loss[index][thought['thought_end_index']:thought['end_index']]
                    reward_signal = (unreduced_loss_detach[zz, idx:] - nll_signal).detach()
            # reward_signal = (unreduced_loss[idx:] - packed_loss[index][thought['thought_end_index']:thought['end_index']]).detach()
            nll_signal = torch.mean(nll_signal)
            reward_signal = torch.mean(reward_signal)
            reward_signals.append(reward_signal)
            nll_signals.append(nll_signal)

            # ipdb.set_trace()


    # based on the difference between the reward_signal and average reward from the bunch of sampled rationales (to reduce variance)
    # We basically Sub
    reward_signals_tensor = torch.stack(reward_signals)
    mean_reward_signals = torch.mean(reward_signals_tensor)

    # ipdb.set_trace()

    gate_loss += -mean_reward_signals*gate_values[zz][idx]#Connect the computational graph from subsampling to the graph with gating mechanism

    #Flag for taking only the best reward signal

    # ipdb.set_trace()
    if not reward_based_optimization:
        if use_best_reward_signal:
            r_ind = 0
            max_reward_signal = None
            max_reward_index = -1

            # Find the index of the maximum reward signal
            for index, batch in enumerate(new_hidden_states):
                configs = packed[f'{index}']
                for indi, thought in enumerate(configs):
                    if max_reward_signal is None or reward_signals[r_ind] > max_reward_signal:
                        max_reward_signal = reward_signals[r_ind]
                        max_reward_index = (index, indi)  # Save the index of the max reward
                    r_ind += 1

            # Now compute the t_likelihood_loss only for the max reward signal
            r_ind = 0
            for index, batch in enumerate(new_hidden_states):
                configs = packed[f'{index}']
                for indi, thought in enumerate(configs):
                    if (index, indi) == max_reward_index:
                        t_likelihood_loss = torch.mean(packed_loss[:thought['thought_end_index']])
                        reinforce_loss += t_likelihood_loss
                        nll_loss_thought += nll_signals[index]
                    r_ind += 1
        else:
            if mean_reward_signals > 0:
                r_ind = 0
                for index, batch in enumerate(new_hidden_states):
                    configs = packed[f'{index}']
                    for indi, thought in enumerate(configs):
                        var_difference = torch.clamp((reward_signals[r_ind] - mean_reward_signals), min=0)
                        t_likelihood_loss = torch.mean(packed_loss[:thought['thought_end_index']])
                        reinforce_loss += var_difference * t_likelihood_loss
                        nll_loss_thought += var_difference * nll_signals[index] #P(yc|x, t)  
                        r_ind+=1
                nll_loss_thought = nll_loss_thought / 5 #Should have been topk? keeping it as 5, for the now
            
            else:
                r_ind = 0
                for index, batch in enumerate(new_hidden_states):
                    configs = packed[f'{index}']
                    for indi, thought in enumerate(configs):
                        t_likelihood_loss = torch.mean(packed_loss[:thought['thought_end_index']])
                        reinforce_loss += 0 * t_likelihood_loss
                        nll_loss_thought += nll_signals[index]
                        r_ind+=1
                nll_loss_thought = 0*nll_loss_thought 
    
    # ipdb.set_trace()

    return gate_loss, reinforce_loss, nll_loss_thought, mean_reward_signals


def start_thinking(input_ids, q_s, q_e, a_s, a_e, hidden_states, logits, labels, unreduced_loss, model, tokenizer, use_reward_decay=True, use_best_reward_signal=True, reward_based_optimization=True, sample_thought_by_prompt=True):
    new_sequence = input_ids.clone().detach()
    hidden_states = hidden_states.clone().detach()
    logits = logits.clone().detach()

    total_gate_loss = 0.0
    total_nll_thought = 0.0
    total_reinforce_loss = 0.0
    logy = []
    reinforce_loss = 0.0
    gate_loss = 0.0
    nll_loss_thought = 0.0

    gate_values = torch.zeros_like(input_ids).to(hidden_states.device)

    #TODO: 1) Gating Mechanism
    #         Pros: 1) Beneficial if the gate's optima is for sele
    # model.gate = model.gate.to(hidden_states.device)
    # gate_values = model.gate(hidden_states)

    # Calculate entropy for each token in the sequence

    entropy_based_sampling = False
    if entropy_based_sampling:
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probabilities * log_probabilities, dim=-1)

        # Select indices with high entropy
        selected_indices_ent = []
        for i in range(len(entropy)):
            # Get the top 3 indices with the highest entropy
            _, top_indices = torch.topk(entropy[i, a_s[i]:a_e[i]], 10)
            top_indices += a_e[i].item()  # Adjust indices to account for the prompt length
            selected_indices_ent.append(top_indices.tolist())

        # Calculate perplexity for each token in the sequence
        perplexity = torch.exp(unreduced_loss.view(logits.shape[0], -1))
        selected_indices_perp = []
        # Select indices with high perplexity
        for i in range(len(perplexity)):
            # Get the top 3 indices with the highest perplexity
            _, top_indices = torch.topk(perplexity[i, a_s[i]:a_e[i]], 3)
            top_indices += a_e[i].item()  # Adjust indices to account for the prompt length
            selected_indices_perp.append(top_indices.tolist())
    else:
        selected_indices = []
        for i in range(len(gate_values)):

            p1 = int((a_s[i] + (a_e[i]-a_s[i])/3).detach().item())
            p2 = int((a_s[i] + 2*((a_e[i]-a_s[i])/3)).detach().item())

            random_indicies = []
            random_indicies.append(torch.randint(a_s[i].item(), p1, (1,))[0])
            random_indicies.append(torch.randint(p1, p2, (1,))[0])
            random_indicies.append(torch.randint(p2, a_e[i].item(), (1,))[0])
            temp = []
            for j in random_indicies:
                gate_values[i][j] = 1.0
                temp.append(j)
            selected_indices.append(temp)

    # ipdb.set_trace()

    # selected_indices = selected_indices_ent
    for i, indices in enumerate(selected_indices):
        for idx in indices:
            gate_values[i][idx] = 1.0

    #gatevalues should be in the completion of the given input_ids
    # random_gate_values = torch.rand(prompt_index)
    # gate_values = gate_values.squeeze(-1)
    
    # selected_indices = []
    # for row in gate_values:
    #     indices = torch.nonzero(row > 0.5).squeeze()
    #     # print("The indices are:",indices)
    #     try:
    #         if len(indices) >= 3:
    #             selected_indices.append(indices[torch.randperm(len(indices))[:3]].tolist())
    #         else:
    #             selected_indices.append(indices.tolist())
    #     except:
    #         selected_indices.append(indices.tolist())


    # print("The selected-indices are:",selected_indices)

    # ipdb.set_trace()
    for zz in range(len(input_ids)):
        temp = {}
        ## gate_values [b, n]
        temp[f"batch-{zz}"] = []
        for idxy, idx in enumerate(selected_indices[zz]):
            value = gate_values[zz][idx]
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
                top_k = 1  # Example: Get the top 5 tokens
                topk_indices = torch.topk(probabilities, top_k, dim=-1)
                
                reasoning_path = []
                thought_index = []

                if sample_thought_by_prompt:
                    for i in range(top_k):
                        # if i < 5:
                        #     continue
                        # prefix_ids = new_sequence[zz][:idx+1]
                        # suffix_ids = new_sequence[zz][idx+1:]
                        # ipdb.set_trace()
                        prefix_ids = new_sequence[zz][q_s[zz]:idx+1]
                        suffix_ids = new_sequence[zz][idx+1:a_e[zz]]

                        thought_ids = gen_thought_by_prompt(prefix_ids, suffix_ids, model, tokenizer).squeeze(0)

                        new_sequence_id = torch.cat([new_sequence[zz][:idx+1], thought_ids, new_sequence[zz][idx+1:]], dim=-1)

                        thought_index.append({
                            'thought_start': idx+1,
                            'thought_end': idx+1 + len(thought_ids)
                        })

                        reasoning_path.append(new_sequence_id)
                        # ipdb.set_trace()
                        sampled_text_generate = {'seq':tokenizer.decode(reasoning_path[-1]), 't_len':len(reasoning_path[-1])}
                        temp1[f"{idx}"].append(sampled_text_generate)

                
                else:
                    for i in range(top_k):
                        # TODO: Append the <SoT> token to the start
                        if i < 5:
                            continue
                        new_sequence_id = new_sequence[zz][:idx+1]# torch.cat([new_sequence[zz][:idx+1], torch.Tensor([]).to(device=new_sequence.device)], dim=0)
                        # Append the sampled top-k token
                        sampled_token = torch.tensor(topk_indices.indices[i].unsqueeze(0)).to(device=new_sequence.device)  # Add the sampled token
                        new_sequence_id = torch.cat([new_sequence_id, sampled_token], dim=0)

                        # sampled_token = sampled_token.cpu()
                        # del sampled_token

                        new_attention_mask = torch.ones(len(new_sequence_id)).unsqueeze(0).to(dtype=torch.long, device=new_sequence_id.device)

                        batch = {'input_ids': new_sequence_id.unsqueeze(0).to(dtype=torch.long), 'attention_mask': new_attention_mask}
                        with torch.no_grad():
                            new_greedy_sequence_decoding = model.generate(**batch, max_new_tokens=200, do_sample=False, temperature=0, use_cache=True, top_p=1.0)
                        # new_greedy_sequence_decoding = torch.ones([1, 100]).to(device=new_sequence.device)
                        thought_index.append({
                            'thought_start': idx+1,
                            'thought_end': idx+1 + len(new_greedy_sequence_decoding[0][idx+1:])
                        })
                        reasoning_path.append(torch.cat([input_ids[zz][: idx+1], new_greedy_sequence_decoding[0][idx+1:], input_ids[zz][idx+1:]], dim=-1))
                        sampled_text_generate = {'seq':tokenizer.decode(reasoning_path[-1]), 't_len':len(reasoning_path[-1])}
                        temp1[f"{idx}"].append(sampled_text_generate)

                        # batch = {k: v.to("cpu") for k, v in batch.items()}
                        # del batch
                        # for reasons in reasoning_path:
                        #     reasons = reasons.to("cpu")
                        #     del reasons

                        new_greedy_sequence_decoding = new_greedy_sequence_decoding.to("cpu")
                        del new_greedy_sequence_decoding
                # reasoning_path shape: [topk_idx, batch_size, seq_len]
                temp[f"batch-{zz}"].append(temp1)

                # ipdb.set_trace()

                packed_reasoning_path, packed_attention_mask, packed_reasoning_path_casual_mask, packed = get_packed_inputs(reasoning_path, max_length=4090, pad_token_id=model.config.eos_token_id, thought_index=thought_index)
                packed_reasoning_path = packed_reasoning_path.to(device=hidden_states.device)
                packed_reasoning_path_casual_mask = packed_reasoning_path_casual_mask.to(device=hidden_states.device)
                # mini_batch_size = 3
                # new_hidden_states = []
                # for i in range(0, len(packed_reasoning_path), mini_batch_size):
                #     mini_packed_reasoning_path = packed_reasoning_path[i:i+mini_batch_size].to(device=hidden_states.device)
                #     mini_packed_reasoning_path_casual_mask = packed_reasoning_path_casual_mask[i:i+mini_batch_size].to(device=hidden_states.device)
                #     print(mini_packed_reasoning_path.shape)
                #     print(mini_packed_attention_mask.shape)
                #     print(mini_packed_reasoning_path_casual_mask.shape)
                #     mini_new_hidden_states = model.model(
                #         input_ids=mini_packed_reasoning_path,
                #         attention_mask=mini_packed_reasoning_path_casual_mask
                #     )
                #     new_hidden_states.append(mini_new_hidden_states[0])

                #     mini_packed_attention_mask = mini_packed_attention_mask.to(device='cpu')
                #     mini_packed_reasoning_path_casual_mask = mini_packed_reasoning_path_casual_mask.to(device='cpu')
                
                # new_hidden_states = torch.cat(new_hidden_states, dim=0)
                new_hidden_states = model.model(
                    input_ids=packed_reasoning_path,
                    attention_mask=packed_reasoning_path_casual_mask
                )
                new_hidden_states = new_hidden_states[0]

                # packed_reasoning_path = packed_reasoning_path.to(device='cpu')
                #Assuming I have access to the indices for the corresponding inputs in the packed batch and let it be stored in packed = {'batch_no':,'</s>':,'<SoT>':,'<EoT>':}
                # I need a config of the following kind:
                # packed = [{'configs': [{'rationale_no': r_no, '<s>_index': st_no, '<SoT>_index': sot_idx, '<EoT>_index': eot_idx, '</s>_index':ed_no},{...}]}]

                gate_loss_i, reinforce_loss_i, nll_loss_thought_i, mean_reward_singals_i = think_loss(zz, idx, new_hidden_states, labels, packed_reasoning_path, model, gate_values, packed, unreduced_loss, use_reward_decay, use_best_reward_signal, reward_based_optimization)
                
                #Normalizations (Latent-position wise and batch-wise)
                gate_loss_i = gate_loss_i/3*len(input_ids)
                reinforce_loss_i = reinforce_loss_i/3*len(input_ids)
                nll_loss_thought_i = nll_loss_thought_i/3*len(input_ids)
                mean_reward_singals_i = mean_reward_singals_i/3*len(input_ids)

                #Let's do a backward here itself to destroy the computational-graph then and there.

                # ipdb.set_trace()
                if reward_based_optimization:
                    thinking_related_loss = -(mean_reward_singals_i)
                else:
                    thinking_related_loss = reinforce_loss_i + nll_loss_thought_i

                thinking_related_loss.backward()

                # total_gate_loss += gate_loss_i if gate_loss_i > 0 else gate_loss_i.detach()
                # total_reinforce_loss += reinforce_loss_i if reinforce_loss_i > 0 else reinforce_loss_i.detach()
                # total_nll_thought += nll_loss_thought_i if nll_loss_thought_i > 0 else nll_loss_thought_i.detach()

                total_gate_loss += gate_loss_i.detach() if isinstance(gate_loss_i, torch.Tensor) else gate_loss_i
                total_reinforce_loss += reinforce_loss_i.detach() if isinstance(reinforce_loss_i, torch.Tensor) else reinforce_loss_i
                total_nll_thought += nll_loss_thought_i.detach() if isinstance(nll_loss_thought_i, torch.Tensor) else nll_loss_thought_i

                

                # new_hidden_states = new_hidden_states.to(device='cpu')
                # packed_reasoning_path = packed_reasoning_path.to(device='cpu')
                # packed_reasoning_path_casual_mask = packed_reasoning_path_casual_mask.to(device='cpu')
                # del new_hidden_states
                # del packed_reasoning_path
                # del packed_reasoning_path_casual_mask

        logy.append(temp)

    #For efficiency purposes, lets accumulate the gradients here itself by doing a backward, so that the computational_graph gets destroyed

    return total_gate_loss, total_reinforce_loss, total_nll_thought, logy

def think_tuner_step(batch, model: AutoModelForCausalLM, tokenizer, train_config):
    # Initial forward pass
    outputs = model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])
    logits = model.lm_head(outputs.last_hidden_state).float()

    # Get unreduced loss
    unreduced_loss, total_nll_loss = calculate_unreduced_loss(logits, batch["labels"], model.config.vocab_size)

    # Start Thinking and get all the losses.
    total_gate_loss, total_reinforce_loss, total_nll_thought, logy = start_thinking(batch["input_ids"], batch["start_idx_q"], batch["end_idx_q"], batch['start_answer_idx'], batch['end_answer_idx'], outputs.last_hidden_state, logits,  batch["labels"], unreduced_loss, model, tokenizer, use_reward_decay = train_config.use_reward_decay, use_best_reward_signal=train_config.use_best_reward_signal,reward_based_optimization=train_config.reward_based_optimization, sample_thought_by_prompt=train_config.sample_thought_by_prompt)
    
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