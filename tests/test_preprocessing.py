from src.data.preprocessing import clean_text


def test_clean_text_lowercases_text():
    text = "FREE MONEY"
    
    result = clean_text(text)

    assert result == "free money"


def test_clean_text_normalizes_whitespace():
    text = "free     money\nnow"

    result = clean_text(text)

    assert result == "free money now"


def test_clean_text_preserves_punctuation():
    text = "WIN!!! FREE!!!"

    result = clean_text(text)

    assert result == "win!!! free!!!"