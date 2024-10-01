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

def get_packed_inputs(input_ids, max_length, pad_token_id):
    """
    Pack sequences into batches of max_length and pad them, returning the padded sequences and attention masks.
    """
    packed_prompts, attention_mask = pack_input_ids(input_ids, max_length=max_length)
    
    # Pad the packed sequences and attention masks to the longest batch
    packed_prompts = torch.nn.utils.rnn.pad_sequence(packed_prompts, batch_first=True, padding_value=pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return packed_prompts, attention_mask