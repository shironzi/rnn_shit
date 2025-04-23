import librosa
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from keras.src.callbacks import EarlyStopping
from keras.src.utils import pad_sequences
from tensorflow.python.keras.callbacks import ModelCheckpoint

from test16 import process_data, phoneme2idx, max_len
import re

dataset = load_dataset("MichaelR207/wiktionary_pronunciations-final")

audio_dataset = dataset['train']['GPT4o_pronunciation'][35].get('array')
sampling_rate = dataset['train']['GPT4o_pronunciation'][35].get('sampling_rate')
word_dataset = dataset['train']['word'][35]

def extract_mfcc(waveform, sampling_rate):
    if isinstance(waveform, np.ndarray):
        audio = waveform
        sr = sampling_rate
    else:
        audio, sr = librosa.load(waveform, sr=sampling_rate)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T  # Transpose so that time steps are along the rows
    return mfcc

# Prepare training data
X_train = []
Y_train = []

for word in word_dataset:
    try:
        # Extract MFCC for the word
        mfcc = extract_mfcc(audio_dataset, sampling_rate)
        X_train.append(mfcc)

        # Process phoneme data
        ref_padded, ref_phonemes = process_data(word)
        Y_train.append(ref_padded)
    except Exception as e:
        print(f"Error processing word '{word}': {e}")
        continue

# Convert to NumPy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Ensure X_train and Y_train have the same number of samples
min_samples = min(len(X_train), len(Y_train))
X_train = X_train[:min_samples]
Y_train = Y_train[:min_samples]

# Reshape data if necessary
X_train = np.reshape(X_train, (X_train.shape[0], -1))
Y_train = np.reshape(Y_train, (Y_train.shape[0], -1))

# Build model
vocab_size = len(phoneme2idx) + 1

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, batch_size=1, epochs=100, validation_split=0.2)