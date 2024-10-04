import torch

attention_mask = [[1, 1, 1, 0, 2, 2, 2], [1, 0, 2, 2,0,0,0]]
attention_mask = torch.tensor(attention_mask)

min_dtype = torch.finfo(torch.float32).min
causal_mask = torch.full((attention_mask.size(1), attention_mask.size(1)), min_dtype)
causal_mask = torch.triu(causal_mask, diagonal=1)
causal_mask = causal_mask[None, None, :, :].expand(attention_mask.size(0), -1, -1, -1)
causal_mask = causal_mask.clone()
bz, seq_len = attention_mask.size()
for i in range(bz):
    for j in range(seq_len):
        if attention_mask[i, j] == 0:
            causal_mask[i, :, j:, :j+1] = min_dtype