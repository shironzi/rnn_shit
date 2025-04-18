from g2p_en import G2p
import nltk
from nltk.corpus import cmudict
import nltk

nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download("cmudict", quiet=True)
pron_dict = cmudict.dict()

g2p = G2p()


def text_to_phonemes(words):
    words = words.lower()
    words = words.replace("|", " ").split()
    phonemes = []

    for word in words:
        try:
            phoneme = g2p(word)
            phonemes.append(phoneme)
        except Exception as e:
            phonemes.append("")

    return phonemes