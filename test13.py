import editdistance
import torch
from transformers import pipeline
import difflib

from phonemes import text_to_phonemes

def speech_to_text_whisper(raw_audio):
    device = 0 if torch.cuda.is_available() else -1
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        framework="pt",
    )

    result = asr_pipeline(raw_audio)
    transcript = result["text"]

    return transcript

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


def pronunciation(raw_audio, reference):
    transcript = speech_to_text_wav2vec2(raw_audio)

    student_phonemes = text_to_phonemes(transcript)
    reference_phonemes = text_to_phonemes(reference)

    student_phonemes_flat = []
    reference_phonemes_flat = []

    for word_phonemes in student_phonemes:
        student_phonemes_flat.extend(word_phonemes)

    for word_phonemes in reference_phonemes:
        reference_phonemes_flat.extend(word_phonemes)

    # Montreal Forced Aligner
    distance = editdistance.eval(reference_phonemes_flat, student_phonemes_flat)

    max_len = max(len(reference_phonemes_flat), len(student_phonemes_flat))

    similarity = (1 - (distance / max_len)) * 100 if max_len > 0 else 0.0

    student_str = " ".join(student_phonemes_flat)
    reference_str = " ".join(reference_phonemes_flat)

    incorrect_pronunciation = []
    min_len = min(len(reference_phonemes_flat), len(student_phonemes_flat))

    for i in range(min_len):
        if reference_phonemes_flat[i] != student_phonemes_flat[i]:
            incorrect_pronunciation.append([reference_phonemes_flat[i], student_phonemes_flat[i]])

    if len(student_phonemes_flat) > len(reference_phonemes_flat):
        for i in range(min_len, len(student_phonemes_flat)):
            incorrect_pronunciation.append(["", student_phonemes_flat[i]])

    print("Student Speech:", transcript)
    print("Correct Speech:", reference)
    print("Student Phonemes:", student_str)
    print("Correct Phonemes:", reference_str)
    print("Incorrect Phonemes:", incorrect_pronunciation)
    print(f"Similarity: {similarity:.2f}%")



def word_error_rate(raw_audio_path, reference):
    # Get the transcription
    transcript = speech_to_text_whisper(raw_audio_path)

    # Normalize and tokenize both transcript and reference
    student_words = transcript.lower().strip().split()
    reference_words = reference.lower().strip().split()

    # Ensure both lists are non-empty
    if not student_words or not reference_words:
        print("Error: Transcript or reference is empty.")
        return None

    # Calculate edit distance (insertions, deletions, substitutions)
    distance = editdistance.eval(reference_words, student_words)

    # WER = (S + D + I) / N, where N is the number of words in the reference
    wer = distance / max(len(reference_words), 1)
    wer_percentage = wer * 100.0

    # Use difflib to find missing and replaced words
    matcher = difflib.SequenceMatcher(None, reference_words, student_words)
    missing_words = []
    replaced_words = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            missing_words.extend(reference_words[i1:i2])
        elif tag == 'replace':
            replaced_words.extend(reference_words[i1:i2])

    print("Student Speech:", " ".join(student_words))
    print("Correct Speech:", " ".join(reference_words))
    print("Missing Words:", missing_words)
    print("Replaced Words:", replaced_words)
    print(f"Word Error Rate: {wer_percentage:.2f}%")

    return wer

word_error_rate("audio/country_philippines.mp3", "I lived at philippines")
word_error_rate("audio/i_pizza.mp3", "I love pizza")