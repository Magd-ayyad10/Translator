import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import re

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 20  # Increase for better accuracy (e.g., 50-100)
LATENT_DIM = 256
NUM_SAMPLES = 10000  # Number of samples to train on (increase if you have compute power)
DATA_PATH = 'eng_fr.txt'

# --- Data Cleaning ---
def clean_text(text):
    # Remove the tags if present
    text = re.sub(r'\\s*', '', text)
    return text.strip()

# --- Load and Preprocess Data ---
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[:min(NUM_SAMPLES, len(lines) - 1)]:
    line = clean_text(line)
    if '\t' not in line:
        continue
    input_text, target_text = line.split('\t')[:2]
    
    # We use "tab" as the "start sequence" character
    # and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    
    input_texts.append(input_text)
    target_texts.append(target_text)

# Tokenization
input_tokenizer = Tokenizer(char_level=True)
input_tokenizer.fit_on_texts(input_texts)
input_data = input_tokenizer.texts_to_sequences(input_texts)

target_tokenizer = Tokenizer(char_level=True)
target_tokenizer.fit_on_texts(target_texts)
target_data = target_tokenizer.texts_to_sequences(target_texts)

# Vocabulary sizes
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# Max sequence length
max_encoder_seq_length = max([len(txt) for txt in input_data])
max_decoder_seq_length = max([len(txt) for txt in target_data])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)

# Padding
encoder_input_data = pad_sequences(input_data, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(target_data, maxlen=max_decoder_seq_length, padding='post')

# Prepare decoder target data (one timestep ahead)
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, target_seq in enumerate(target_data):
    for t, char_index in enumerate(target_seq):
        if t > 0:
            # decoder_target_data is ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, char_index] = 1.0

# --- Build Model ---
# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, LATENT_DIM)(encoder_inputs)
encoder_lstm = LSTM(LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, LATENT_DIM)(decoder_inputs)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Compile
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
print("Starting training...")
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2
)

# --- Save Artifacts ---
model.save('nmt_model.h5')

with open('tokenizer_input.pkl', 'wb') as f:
    pickle.dump(input_tokenizer, f)
    
with open('tokenizer_target.pkl', 'wb') as f:
    pickle.dump(target_tokenizer, f)

# Save config for inference
config = {
    'max_encoder_seq_length': max_encoder_seq_length,
    'max_decoder_seq_length': max_decoder_seq_length,
    'num_encoder_tokens': num_encoder_tokens,
    'num_decoder_tokens': num_decoder_tokens,
    'latent_dim': LATENT_DIM
}
with open('model_config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("Model and tokenizers saved.")