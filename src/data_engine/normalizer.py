
import re
import unicodedata

def normalize_arabic(text: str) -> str:
    """
    Conservative Arabic text normalization.

    Preserves dialectal vocabulary and original
    spelling choices as much as possible.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Remove tatweel
    text = text.replace("\u0640", "")

    # Remove Arabic diacritics from normalized copy
    arabic_diacritics = re.compile(
        r"[\u064B-\u065F\u0670]"
    )
    text = arabic_diacritics.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

if __name__ == "__main__":

    samples = [
        "شْلُونَك؟",
        "كيف حالكــــ؟",
        "شلونك؟    شخبارك؟"
    ]

    for sample in samples:
        print("Original:", sample)
        print("Normalized:", normalize_arabic(sample))
        print("-" * 30)