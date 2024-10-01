import torch


def pack_input_ids(input_ids, max_length=32):
    """
    Pack input_ids of shape [batch_size, index, seq_len] into batches of size max_length.
    Handles tensor inputs with batch dimension and index.
    """
    bz, idx, seq_len = input_ids.shape
    packed_inputs = []
    attention_masks = []

    for i in range(bz):
        for j in range(idx):
            current_sequence = input_ids[i, j]  # Get the i-th batch and j-th index (shape [seq_len])
            current_length = 0
            current_batch = []
            current_attention_mask = []

            while current_length < seq_len:
                remaining_length = seq_len - current_length
                space_in_batch = max_length - len(current_batch)

                if remaining_length > space_in_batch:
                    # We can only take a part of the current sequence
                    part_to_add = current_sequence[current_length:current_length + space_in_batch]
                    current_batch.append(part_to_add)
                    current_attention_mask += [1] * space_in_batch
                    packed_inputs.append(torch.cat(current_batch, dim=0))
                    attention_masks.append(torch.tensor(current_attention_mask))

                    # Start a new batch
                    current_batch = []
                    current_attention_mask = []
                    current_length += space_in_batch
                else:
                    # We can add the remaining sequence to the current batch
                    part_to_add = current_sequence[current_length:]
                    current_batch.append(part_to_add)
                    current_attention_mask += [1] * remaining_length
                    packed_inputs.append(torch.cat(current_batch, dim=0))
                    attention_masks.append(torch.tensor(current_attention_mask))

                    # End the sequence processing for this batch
                    break

    # Convert packed_inputs and attention_masks to a tensor if needed
    packed_inputs = torch.stack(packed_inputs)
    attention_masks = torch.stack(attention_masks)

    return packed_inputs, attention_masks

def get_packed_inputs(input_ids, max_length, pad_token_id):
    packed_prompts, attention_mask = pack_input_ids(input_ids, max_length=max_length)
    packed_prompts = torch.nn.utils.rnn.pad_sequence(packed_prompts, batch_first=True, padding_value=pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return packed_prompts, attention_mask