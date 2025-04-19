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
    transcript = speech_to_text_whisper(raw_audio_path)

    students_words = transcript.lower().split(" ")
    reference_words = reference.lower().split(" ")

    students_words.pop(0)

    distance = editdistance.eval(reference_words, students_words)

    # WER = (Substitutions + Insertions + Deletions) / Number of words in reference
    wer = distance / max(len(reference_words), 1)
    wer_percentage = wer * 100.0

    matcher = difflib.SequenceMatcher(None, reference_words, students_words)
    missing_words = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('delete', 'replace'):
            # Words in reference but not in transcript (or replaced)
            missing_words.extend(reference_words[i1:i2])
    
    print("Student Speech:", students_words)
    print("Correct Speech:", reference_words)
    print("Missing words:", missing_words)
    print(f"Word Error Rate: {wer_percentage:.2f}%")

word_error_rate("audio/country_philippines.mp3", "I lived at philippines")
word_error_rate("audio/i_pizza.mp3", "I love pizza")