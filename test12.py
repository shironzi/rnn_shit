import torch
import torchaudio
from torchaudio.transforms import Resample
from test2 import GreedyCTCDecoder
from phonemes import text_to_phonemes
import editdistance

def speech_to_text(raw_audio):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Wav2Vec2 model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    # Load and resample audio
    waveform, sample_rate = torchaudio.load(raw_audio)
    if sample_rate != bundle.sample_rate:
        waveform = Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)(waveform)

    waveform = waveform.to(device)

    with torch.inference_mode():
        emissions, _ = model(waveform)

    # Use custom greedy decoder
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emissions[0].cpu())

    return transcript

def pronunciation(raw_audio, reference):

    transcript = speech_to_text(raw_audio)

    student_phonemes = text_to_phonemes(transcript)
    reference_phonemes = text_to_phonemes(reference)

    student_phonemes_flat = []
    reference_phonemes_flat = []
    
    for word_phonemes in student_phonemes:
        student_phonemes_flat.extend(word_phonemes)
    
    for word_phonemes in reference_phonemes:
        reference_phonemes_flat.extend(word_phonemes)

    distance = editdistance.eval(reference_phonemes_flat, student_phonemes_flat)

    max_len = max(len(reference_phonemes_flat), len(student_phonemes_flat))

    print(max_len)

    similarity = (1 - (distance / max_len))*100 if max_len > 0 else 0.0

    student_str = " ".join(student_phonemes_flat)
    reference_str = " ".join(reference_phonemes_flat)

    incorrect_pronunciation = []
    min_len = min(len(reference_phonemes_flat), len(student_phonemes_flat))

    for i in range(min_len):
        if reference_phonemes_flat[i] != student_phonemes_flat[i]:
            incorrect_pronunciation.append([reference_phonemes_flat[i], student_phonemes_flat[i]])

    if len(student_phonemes_flat) > len(reference_phonemes_flat):
        for i in range(min_len, len(student_phonemes_flat)):
            incorrect_pronunciation.append([student_phonemes_flat[i], ""])

    print("Student Speech:", transcript)
    print("Correct Speech:", reference)
    print("Student Phonemes:", student_str)
    print("Correct Phonemes:", reference_str)
    print("Incorrect Phonemes:", incorrect_pronunciation)
    print(f"Similarity: {similarity:.2f}%")

pronunciation("audio/attack_shark.mp3", "attack shark")

def word_error_rate(raw_audio, reference):
    transcript = speech_to_text(raw_audio)

    student_words = transcript.replace("|", " ").lower().split()
    reference_words = reference.lower().split()

    distance = editdistance.eval(reference_words, student_words)
    
    # WER = (Substitutions + Insertions + Deletions) / Number of words in reference
    wer = distance / max(len(reference_words), 1)

    wer_percentage = wer * 100.0
    similarity = 100.0 - wer_percentage
    
    print("Student Speech:", ' '.join(student_words))
    print("Correct Speech:", reference)
    print(f"Word Error Rate: {wer_percentage:.2f}%")
    print(f"Similarity: {similarity:.2f}%")
    
    return wer


# word_error_rate("audio/hello_world.mp3", "hello world everyone")


