from torch.utils.data import Dataset
import torch


class ParallelDataset(Dataset):
def __init__(self, src_sentences, tgt_sentences):
assert len(src_sentences) == len(tgt_sentences)
self.src = src_sentences
self.tgt = tgt_sentences


def __len__(self):
return len(self.src)


def __getitem__(self, idx):
return torch.tensor(self.src[idx], dtype=torch.long), torch.tensor(self.tgt[idx], dtype=torch.long)