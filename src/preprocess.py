import argparse
import sentencepiece as spm
from pathlib import Path




def train_spm(input_file: str, model_prefix: str, vocab_size: int = 16000):
spm.SentencePieceTrainer.Train(
input=input_file,
model_prefix=model_prefix,
vocab_size=vocab_size,
character_coverage=1.0,
model_type='unigram')




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--vocab_size', type=int, default=16000)
args = parser.parse_args()


data_dir = Path(args.data_dir)
# Expect train.en and train.fr
train_en = data_dir / 'train.en'
train_fr = data_dir / 'train.fr'


assert train_en.exists() and train_fr.exists(), 'train.en and train.fr must exist in data_dir'


print('Training SentencePiece for English...')
train_spm(str(train_en), 'spm_en', args.vocab_size)
print('Training SentencePiece for French...')
train_spm(str(train_fr), 'spm_fr', args.vocab_size)


print('Done. Generated spm_en.model / spm_fr.model')