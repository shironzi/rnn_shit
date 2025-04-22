import json
import numpy as np
import torch
from transformers import pipeline
from phonemes import g2p
import tensorflow as tf

try:
    model_name = "models/pronunciation_model_v3.h5"
    model = tf.keras.models.load_model(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

try:
    with open('phoneme_vocab.json', 'r') as f:
        vocab_data = json.load(f)

    phoneme2idx = vocab_data['phoneme2idx']
    max_len = vocab_data['max_len']
    vocab_size = vocab_data['vocab_size']
    print("Vocabulary loaded successfully!")
except FileNotFoundError:
    print("Error: phoneme_vocab.json not found!")

def text_to_phonemes(words):
    words = words.lower()
    words = words.replace("|", " ").split()
    phonemes = []

    for word in words:
        try:
            phoneme = g2p(word)
            phonemes.append(phoneme)
        except Exception as e:
            phonemes.append(" ")

    return phonemes

def process_data(text):
    text = text.lower()
    phonemes = text_to_phonemes(text)

    phonemes_flat = []

    for i, words in enumerate(phonemes):
        for word in words:
            phonemes_flat.append(word)
        if i < len(phonemes) - 1:
            phonemes_flat.append("")
    indices = []

    for ph in phonemes_flat:
        idx = phoneme2idx.get(ph, 0)
        indices.append(idx)

    padded = np.zeros(max_len, dtype=int)
    padded[:min(len(indices), max_len)] = indices[:max_len]

    padded = padded.reshape(1, -1)

    return padded, phonemes_flat

def speech_to_text_wav2vec2(raw_audio):
    device = 0 if torch.cuda.is_available() else -1
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="facebook/wav2vec2-base-960h",
        device=device,
        framework="pt"
    )

    result = asr_pipeline(raw_audio)
    transcript = result["text"]

    return transcript

# calculate to pronunciation accuracy
def calculate_pronunciation_accuracy(student_phonemes, reference_phonemes):
    correct = 0
    total = len(reference_phonemes)

    for i in range(min(len(student_phonemes), len(reference_phonemes))):
        if student_phonemes[i] == reference_phonemes[i]:
            correct += 1

    accuracy = correct / total * 100  # Return as a percentage
    return accuracy

# get the missing phonemes
def calculate_missing_phonemes(student_phonemes, reference_phonemes):
    missing = []
    for i in range(len(reference_phonemes)):
        if i >= len(student_phonemes) or student_phonemes[i] != reference_phonemes[i]:
            missing.append(reference_phonemes[i])

    return missing

def evaluate_pronunciation(student_speech, reference_speech):
    student = speech_to_text_wav2vec2(student_speech)
    student = student.lower()
    reference = reference_speech.lower()

    stu_padded, stu_phonemes = process_data(student)
    ref_padded, ref_phonemes = process_data(reference)

    predictions = model.predict(stu_padded)

    predicted_phonemes = np.argmax(predictions, axis=-1)
    predicted_phonemes = [phoneme2idx.get(idx, "") for idx in predicted_phonemes[0]]

    # Calculate accuracy
    accuracy = calculate_pronunciation_accuracy(stu_phonemes, ref_phonemes)

    # Calculate missing phonemes
    missing_phonemes = calculate_missing_phonemes(stu_phonemes, ref_phonemes)

    # Output the results
    print(f"Pronunciation Accuracy: {accuracy}%")
    print(f"Missing Phonemes: {missing_phonemes}")

student_speech = "audio/delight.mp3"
reference_speech = "delight"
evaluate_pronunciation(student_speech, reference_speech)

