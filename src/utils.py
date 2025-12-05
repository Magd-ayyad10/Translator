import torch
from typing import List


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2




def collate_fn(batch):
src_batch, tgt_batch = zip(*batch)
src_lens = [len(x) for x in src_batch]
tgt_lens = [len(x) for x in tgt_batch]
max_src = max(src_lens)
max_tgt = max(tgt_lens)


src_padded = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
tgt_padded = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)


for i, s in enumerate(src_batch):
src_padded[i, : len(s)] = s
for i, t in enumerate(tgt_batch):
tgt_padded[i, : len(t)] = t


return src_padded, tgt_padded




def make_src_mask(src):
# src: [batch, src_len]
mask = (src == PAD_ID)
return mask




def make_tgt_mask(tgt):
# tgt: [batch, tgt_len]
tgt_len = tgt.size(1)
subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), dtype=torch.bool), diagonal=1)
pad_mask = (tgt == PAD_ID)
return subsequent_mask, pad_mask