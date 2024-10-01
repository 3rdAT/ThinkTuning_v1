import torch

def pack_input_ids(input_ids, max_length=32):
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
    for sequence in input_ids:
        sequence_length = len(sequence)

        # Check if adding the current sequence exceeds the max_length
        if current_length + sequence_length > max_length:
            # If it exceeds, append the current batch and attention mask, and start a new batch
            packed_inputs.append(torch.cat(current_batch, dim=0))
            attention_masks.append(torch.tensor(current_attention_mask))

            # Start a new batch with the current sequence
            current_batch = [sequence]
            current_attention_no = 1
            current_attention_mask = [current_attention_no] * sequence_length
            current_length = sequence_length
            
        else:
            # Add the current sequence to the current batch
            current_batch.append(sequence)
            current_attention_mask += [current_attention_no] * sequence_length
            current_length += sequence_length
        current_attention_no += 1

    # Add the last batch if it's not empty
    if current_batch:
        packed_inputs.append(torch.cat(current_batch, dim=0))
        attention_masks.append(torch.tensor(current_attention_mask))

    return packed_inputs, attention_masks

def build_batched_causal_mask_from_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype = torch.float32):
    """
    Creates a batched 4D causal mask from attention_mask where only elements with the same values in the attention_mask
    can attend to each other, and elements with value 0 in the attention_mask are ignored.

    Args:
        attention_mask (torch.Tensor): A 2D tensor of shape (batch_size, sequence_length) with different values representing groups.
        dtype (torch.dtype): Data type for the mask (default: torch.float32).

    Returns:
        causal_mask (torch.Tensor): A batched 4D causal mask of shape (batch_size, 1, sequence_length, sequence_length)
                                    where elements with the same values in attention_mask can attend to each other,
                                    and elements with value 0 are ignored.
    """
    batch_size, seq_len = attention_mask.size()

    # Create a base causal mask (lower triangular matrix), ensuring tokens can't attend to future tokens
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=dtype)).unsqueeze(0).expand(batch_size, -1, -1).to(attention_mask.device)

    # Create a mask where elements with the same attention_mask values can attend to each other (for each batch)
    same_value_mask = (attention_mask.unsqueeze(1) == attention_mask.unsqueeze(2)).to(dtype)

    # Create a mask to ignore elements with value 0 in the attention_mask (for each batch)
    non_zero_mask = (attention_mask != 0).unsqueeze(1).to(dtype)  # Mask where 1 indicates non-zero elements

    # Combine the causal mask with the same-value mask (element-wise multiplication)
    combined_mask = causal_mask * same_value_mask

    # Apply the non-zero mask to ignore tokens with attention_mask value 0
    combined_mask = combined_mask * non_zero_mask * non_zero_mask.transpose(1, 2)

    # Expand the mask to 4D with the second dimension as 1
    combined_mask = combined_mask.unsqueeze(1)

    return combined_mask

def get_packed_inputs(input_ids, max_length, pad_token_id):
    """
    Pack sequences into batches of max_length and pad them, returning the padded sequences and attention masks.
    """
    packed_prompts, attention_mask = pack_input_ids(input_ids, max_length=max_length)
    
    # Pad the packed sequences and attention masks to the longest batch
    packed_prompts = torch.nn.utils.rnn.pad_sequence(packed_prompts, batch_first=True, padding_value=pad_token_id).to(input_ids[0].device)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).to(input_ids[0].device)

    casual_mask = build_batched_causal_mask_from_attention_mask(attention_mask, torch.float).to(input_ids[0].device)

    return packed_prompts, attention_mask, casual_mask