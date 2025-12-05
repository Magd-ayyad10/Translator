import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
def __init__(self, vocab_size, emb_size):
super().__init__()
self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
self.emb_size = emb_size


def forward(self, tokens):
return self.embedding(tokens) * (self.emb_size ** 0.5)




class PositionalEncoding(nn.Module):
def __init__(self, emb_size: int, maxlen: int = 5000):
super().__init__()
pe = torch.zeros(maxlen, emb_size)
position = torch.arange(0, maxlen).unsqueeze(1)
div_term = torch.exp(torch.arange(0, emb_size, 2) * (-torch.log(torch.tensor(10000.0)) / emb_size))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0)
self.register_buffer('pe', pe)


def forward(self, x):
# x: [batch, seq_len, emb]
x = x + self.pe[:, : x.size(1), :]
return x




class TransformerModel(nn.Module):
def __init__(self, src_vocab, tgt_vocab, emb_size=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
super().__init__()
self.src_tok_emb = TokenEmbedding(src_vocab, emb_size)
self.tgt_tok_emb = TokenEmbedding(tgt_vocab, emb_size)
self.positional_encoding = PositionalEncoding(emb_size)
self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead,
num_encoder_layers=num_encoder_layers,
num_decoder_layers=num_decoder_layers,
dim_feedforward=dim_feedforward,
dropout=dropout)
self.generator = nn.Linear(emb_size, tgt_vocab)


def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
# src: [batch, src_len]
# tgt: [batch, tgt_len]
src_emb = self.positional_encoding(self.src_tok_emb(src)) # [batch, src_len, emb]
tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt)) # [batch, tgt_len, emb]


# transformer expects [seq_len, batch, emb]
src_emb = src_emb.transpose(0, 1)
tgt_emb = tgt_emb.transpose(0, 1)


output = self.transformer(src_emb, tgt_emb,
src_mask=src_mask,
tgt_mask=tgt_mask,
src_key_padding_mask=src_key_padding_mask,
tgt_key_padding_mask=tgt_key_padding_mask,
memory_key_padding_mask=memory_key_padding_mask)
out = self.generator(output) # [tgt_len, batch, vocab]
return out.transpose(0, 1)