import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
import pickle

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    model = load_model('nmt_model.h5')
    
    with open('tokenizer_input.pkl', 'rb') as f:
        input_tokenizer = pickle.load(f)
        
    with open('tokenizer_target.pkl', 'rb') as f:
        target_tokenizer = pickle.load(f)
        
    with open('model_config.pkl', 'rb') as f:
        config = pickle.load(f)
        
    return model, input_tokenizer, target_tokenizer, config

try:
    model, input_tokenizer, target_tokenizer, config = load_artifacts()
except Exception as e:
    st.error("Error loading model files. Please run 'train_model.py' first.")
    st.stop()

# --- Reconstruct Inference Models ---
# We need to break the trained model into encoder and decoder parts for inference
latent_dim = config['latent_dim']

# Encoder
encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output # lstm_1 output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder
decoder_inputs = model.input[1] # input_2
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_embedding_layer = model.layers[3] # embedding_1
decoder_lstm = model.layers[5] # lstm_2
decoder_dense = model.layers[6] # dense

decoder_inputs_x = decoder_embedding_layer(decoder_inputs)
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs_x, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to something readable.
reverse_target_char_index = dict((i, char) for char, i in target_tokenizer.word_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    # In our training, we used '\t' as start char, its index is target_tokenizer.word_index['\t']
    target_seq[0, 0] = target_tokenizer.word_index['\t']

    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index.get(sampled_token_index, '')

        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > config['max_decoder_seq_length']):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# --- Streamlit UI ---
st.title("🇫🇷 English to French Translator")
st.markdown("Neural Machine Translation using LSTM (Seq2Seq)")

text_input = st.text_input("Enter English text:", "Go.")

if st.button("Translate"):
    if text_input:
        # Preprocess input
        seq = input_tokenizer.texts_to_sequences([text_input])
        padded_seq = pad_sequences(seq, maxlen=config['max_encoder_seq_length'], padding='post')
        
        # Predict
        translation = decode_sequence(padded_seq)
        
        st.success("French Translation:")
        st.write(f"**{translation.strip()}**")
    else:
        st.warning("Please enter some text.")

st.sidebar.info("Note: This is a character-level model trained on a small subset of data for demonstration.")