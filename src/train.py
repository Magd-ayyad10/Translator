import argparse


def load_and_encode(spm_model, path, bos_id=1, eos_id=2):
sp = spm.SentencePieceProcessor()
sp.Load(spm_model)
out = []
with open(path, 'r', encoding='utf-8') as f:
for line in f:
line = line.strip()
if not line:
continue
ids = sp.EncodeAsIds(line)
seq = [bos_id] + ids + [eos_id]
out.append(seq)
return out




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--spm_en', default='spm_en.model')
parser.add_argument('--spm_fr', default='spm_fr.model')
parser.add_argument('--checkpoints_dir', default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()


data_dir = Path(args.data_dir)
checkpoints_dir = Path(args.checkpoints_dir)
checkpoints_dir.mkdir(parents=True, exist_ok=True)


print('Loading and encoding data...')
train_src = load_and_encode(args.spm_en, data_dir / 'train.en')
train_tgt = load_and_encode(args.spm_fr, data_dir / 'train.fr')
valid_src = load_and_encode(args.spm_en, data_dir / 'valid.en')
valid_tgt = load_and_encode(args.spm_fr, data_dir / 'valid.fr')


# Build vocab sizes from sentencepiece vocab files
sp_en = spm.SentencePieceProcessor(); sp_en.Load(args.spm_en)
sp_fr = spm.SentencePieceProcessor(); sp_fr.Load(args.spm_fr)
src_vocab = sp_en.GetPieceSize() + 3 # account for BOS/EOS/PAD
tgt_vocab = sp_fr.GetPieceSize() + 3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset = ParallelDataset(train_src, train_tgt)
valid_dataset = ParallelDataset(valid_src, valid_tgt)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


model = TransformerModel(src_vocab, tgt_vocab).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


best_valid_loss = float('inf')


for epoch in range(1, args.epochs + 1):
model.train()
t