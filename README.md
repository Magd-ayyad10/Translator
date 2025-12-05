# NMT English → French (Transformer, PyTorch)

This repository trains a Transformer-based Neural Machine Translation model (English → French) using PyTorch and SentencePiece.

### Requirements
- Python 3.8+
- Run: pip install -r requirements.txt

### Steps
1. Train tokenizer:
   python src/preprocess.py --data_dir data --vocab_size 16000

2. Train model:
   python src/train.py --data_dir data --checkpoints_dir checkpoints --epochs 20

3. Translate sentence:
   python src/translate.py --checkpoint checkpoints/best.pt --spm_en spm_en.model --spm_fr spm_fr.model --sentence "hello"
