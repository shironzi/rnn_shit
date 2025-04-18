import torch
import torchaudio
from torchaudio.transforms import Resample
from test2 import GreedyCTCDecoder
from phonemes import text_to_phonemes
import editdistance

def pronunciation(raw_audio, reference):
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

    # Converts text into phonemes
    student_phonemes = text_to_phonemes(transcript)
    reference_phonemes = text_to_phonemes(reference)

    student_str = ""
    reference_str = ""
    for phonemeList in student_phonemes:
        for phoneme in phonemeList:
            student_str += phoneme + " "

    for phonemeList in reference_phonemes:
        for phoneme in phonemeList:
            reference_str += phoneme + " "

    # Levenshtein Distance
    distance = editdistance.eval(reference_str, student_str)

    max_len = max(len(student_str), len(reference_str))
    similarity = (1 - (distance / max_len)) if max_len > 0 else 0.0

    print("Student Speech:", transcript)
    print("Correct Speech:", reference)
    print("Student Phonemes:", student_str)
    print("Correct Phonemes:", reference_str)
    print(f"Similarity: {similarity:.2f}%")


pronunciation("audio/hello.mp3", "hello")
