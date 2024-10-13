import torch

def pack_input_ids(input_ids, pad_token_id, max_length=32, thought_index=[]):
    """
    Pack input_ids from a list of tensors of shape [sequence_length] into batches of size max_length.
    Does not split individual sequences but packs multiple sequences together until max_length is reached.
    """
    packed_inputs = []
    attention_masks = []
    current_batch = []
    current_attention_mask = []
    current_length = 0
    current_attention_no = 1

    # Iterate through the list of sequence tensors
    pad_token_id = torch.tensor([pad_token_id]).to(input_ids[0].device)
    batchy = 0
    batch_config = {}

    if batchy not in batch_config:
        batch_config[f'{batchy}'] = []

    for index, sequence in enumerate(input_ids):
        # print(f"The batchy value is {batchy}")
        thought_index_config = thought_index[index]
        sequence_length = len(sequence)
        

        # Check if adding the current sequence exceeds the max_length
        if current_length + sequence_length > max_length:
            # If it exceeds, append the current batch and attention mask, and start a new batch

            packed_inputs.append(torch.cat(current_batch, dim=0))
            attention_masks.append(torch.tensor(current_attention_mask))

            # Start a new batch with the current sequence
            batchy = batchy+1
            # print(f"Batchy inside if: {batchy}")
            if batchy not in batch_config:
                batch_config[f'{batchy}'] = []

            start_index = 0
            current_batch = [sequence]
            current_attention_no = 1
            current_attention_mask = [current_attention_no] * sequence_length
            current_length = sequence_length

            temp = {'thought_no':index, 'start_index':start_index, 'thought_start_index':start_index+thought_index_config['thought_start'], 'thought_end_index':start_index+thought_index_config['thought_end'], 'end_index':current_length}
            batch_config[f'{batchy}'].append(temp)

            
        else:
            # Add the current sequence to the current batch
            start_index = current_length
            current_batch.append(torch.cat([sequence, pad_token_id]))
            current_attention_mask += [current_attention_no] * sequence_length
            current_attention_mask += [0]
            current_length += sequence_length
            end_index = current_length

            # print(f"Batchy inside else: {batchy}")
            
            temp = {'thought_no':index, 'start_index':start_index, 'thought_start_index':start_index+thought_index_config['thought_start'], 'thought_end_index':start_index+thought_index_config['thought_end'], 'end_index':end_index}
            batch_config[f'{batchy}'].append(temp)

        current_attention_no += 1

    # Add the last batch if it's not empty
    if current_batch:
        packed_inputs.append(torch.cat(current_batch, dim=0))
        attention_masks.append(torch.tensor(current_attention_mask))

    return packed_inputs, attention_masks, batch_config

def build_batched_causal_mask_from_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32):
    """
    Creates a batched 4D causal mask from attention_mask where elements with the same values in the attention_mask
    can attend to each other (set to 0 in the mask), and elements with value 0 or different values are ignored
    (set to min_dtype in the mask). The causal property ensures no future tokens are attended.

    Args:
        attention_mask (torch.Tensor): A 2D tensor of shape (batch_size, sequence_length) with different values representing groups.
        dtype (torch.dtype): Data type for the mask (default: torch.float32).

    Returns:
        causal_mask (torch.Tensor): A batched 4D causal mask of shape (batch_size, 1, sequence_length, sequence_length)
                                    where elements with the same values in attention_mask can attend to each other,
                                    and elements with value 0 or different values are ignored (set to min_dtype).
    """
    bz, seq_len = attention_mask.size()

    min_dtype = torch.finfo(torch.float32).min

    causal_mask = torch.triu(torch.full((seq_len, seq_len), min_dtype, dtype=dtype, device=attention_mask.device), diagonal=1).to(attention_mask.device)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bz, -1, -1, -1).clone()

    zero_positions = (attention_mask == 0)  # Shape: [bz, seq_len]

    k_indices = torch.arange(seq_len).view(1, 1, seq_len, 1).to(attention_mask.device)  # Shape: [1, 1, seq_len, 1]
    l_indices = torch.arange(seq_len).view(1, 1, 1, seq_len).to(attention_mask.device)  # Shape: [1, 1, 1, seq_len]
    j_indices = torch.arange(seq_len).view(1, seq_len, 1, 1).to(attention_mask.device)  # Shape: [1, seq_len, 1, 1]

    zero_positions = zero_positions.view(bz, seq_len, 1, 1)  # Shape: [bz, seq_len, 1, 1]

    masks_per_j = (k_indices >= j_indices) & (l_indices <= j_indices)  # Shape: [1, seq_len, seq_len, seq_len]

    masks = zero_positions & masks_per_j  # Shape: [bz, seq_len, seq_len, seq_len]

    mask = masks.any(dim=1)  # Shape: [bz, seq_len, seq_len]

    mask = mask.unsqueeze(1)  # Shape: [bz, 1, seq_len, seq_len]

    causal_mask[mask] = min_dtype

    return causal_mask

def get_packed_inputs(input_ids, max_length, pad_token_id, thought_index):
    """
    Pack sequences into batches of max_length and pad them, returning the padded sequences and attention masks.
    """
    packed_prompts, attention_mask, packed = pack_input_ids(input_ids, pad_token_id=pad_token_id,  max_length=max_length, thought_index=thought_index)
    
    # Pad the packed sequences and attention masks to the longest batch
    packed_prompts = torch.nn.utils.rnn.pad_sequence(packed_prompts, batch_first=True, padding_value=pad_token_id).to(input_ids[0].device)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    casual_mask = build_batched_causal_mask_from_attention_mask(attention_mask, torch.float).to(input_ids[0].device)

    return packed_prompts, attention_mask, casual_mask, packed