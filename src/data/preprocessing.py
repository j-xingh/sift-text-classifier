import re


def clean_text(text: str) -> str:
    """
    Basic text normalization for SIFT.

    Steps:
    1. Convert input to string.
    2. Convert to lowercase.
    3. Normalize whitespace.
    """

    text = str(text)

    # Normalize case
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text