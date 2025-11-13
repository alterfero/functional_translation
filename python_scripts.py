# make_vocab_from_wordnet.py

import nltk
from nltk.corpus import wordnet as wn

POS_MAP = {
    "n": "Noun",
    "v": "Verb",
    "a": "Adjective",
    "s": "Adjective",  # satellite adjectives → treat as adjectives
    "r": "Adverb",
}


def build_lemma_vocab(output_file: str = "vocab.txt") -> None:
    """Build a vocabulary file with lemmas and their WordNet POS tags.

    Each line in the generated file has the format ``lemma\tPOS[,POS...]`` where
    POS values are human-readable categories ("Noun", "Verb", ...). Lemmas that
    appear under multiple parts of speech will list all of them.
    """

    # Make sure WordNet is available (safe if already downloaded)
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet")
        wn.ensure_loaded()

    lemma_pos: dict[str, set[str]] = {}

    # Iterate over all synsets and their lemmas
    for syn in wn.all_synsets():
        pos_label = POS_MAP.get(syn.pos())
        if not pos_label:
            continue  # Skip unexpected POS values

        for lemma in syn.lemmas():
            name = lemma.name().lower()  # e.g., "dog", "ice_cream"

            # Keep only purely alphabetic single-word lemmas
            if not name.isalpha():
                continue

            bucket = lemma_pos.setdefault(name, set())
            bucket.add(pos_label)

    words = sorted(lemma_pos.keys())

    with open(output_file, "w", encoding="utf-8") as f:
        for word in words:
            pos_values = sorted(lemma_pos[word])
            if not pos_values:
                continue
            f.write(f"{word}\t{','.join(pos_values)}\n")

    print(f"Saved {len(words)} lemmas to {output_file}")


if __name__ == "__main__":
    build_lemma_vocab()
