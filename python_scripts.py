# make_vocab_from_wordnet.py

import nltk
from nltk.corpus import wordnet as wn

def build_lemma_vocab(output_file: str = "vocab.txt") -> None:
    # Make sure WordNet is available (safe if already downloaded)
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet")
        wn.ensure_loaded()

    lemmas = set()

    # Iterate over all synsets and their lemmas
    for syn in wn.all_synsets():
        for l in syn.lemmas():
            name = l.name().lower()  # e.g., "dog", "ice_cream"

            # Keep only purely alphabetic single-word lemmas
            if name.isalpha():
                lemmas.add(name)

    lemmas = sorted(lemmas)

    with open(output_file, "w", encoding="utf-8") as f:
        for lemma in lemmas:
            f.write(lemma + "\n")

    print(f"Saved {len(lemmas)} lemmas to {output_file}")


if __name__ == "__main__":
    build_lemma_vocab()
