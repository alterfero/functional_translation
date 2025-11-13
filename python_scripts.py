# make_vocab_from_wordnet.py

import nltk
from nltk.corpus import wordnet as wn


POS_MAP = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",  # satellite adjectives → adjective bucket
    "r": "adverb",
}


def build_lemma_vocab(output_file: str = "vocab.txt") -> None:
    """Generate a vocabulary of (lemma, POS) pairs from WordNet.

    The resulting file stores one entry per line in the format
    ``lemma\tpos`` where ``pos`` is one of ``noun``, ``verb``,
    ``adjective`` or ``adverb``. Lemmas containing non-alphabetic
    characters are skipped to keep the vocabulary consistent with the
    previous behaviour (single word, alphabetic entries).
    """

    # Make sure WordNet is available (safe if already downloaded)
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet")
        wn.ensure_loaded()

    entries = set()

    # Iterate over all synsets and their lemmas
    for syn in wn.all_synsets():
        pos = POS_MAP.get(syn.pos())
        if not pos:
            continue
        for lemma in syn.lemmas():
            name = lemma.name().lower()  # e.g., "dog", "ice_cream"

            # Keep only purely alphabetic single-word lemmas
            if name.isalpha():
                entries.add((name, pos))

    sorted_entries = sorted(entries)

    with open(output_file, "w", encoding="utf-8") as f:
        for lemma, pos in sorted_entries:
            f.write(f"{lemma}\t{pos}\n")

    print(f"Saved {len(sorted_entries)} lemma/POS entries to {output_file}")


if __name__ == "__main__":
    build_lemma_vocab()
