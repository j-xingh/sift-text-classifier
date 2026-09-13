import numpy as np

from src.features.tfidf import (
    tokenize,
    build_vocabulary,
    compute_tf,
    compute_document_frequency,
    compute_idf,
    compute_tfidf_vector,
    transform_documents,
)


def test_tokenize():
    text = "Free money!"
    result = tokenize(text)

    assert result == ["free", "money", "!"]


def test_build_vocabulary():
    documents = [
        "hello friend",
        "free prize",
        "hello prize",
    ]

    vocabulary = build_vocabulary(documents)

    assert set(vocabulary.keys()) == {
        "hello",
        "friend",
        "free",
        "prize",
    }

    assert len(vocabulary) == 4


def test_compute_tf():
    tokens = [
        "free",
        "free",
        "prize",
        "now",
    ]

    result = compute_tf(tokens)

    assert result["free"] == 0.5
    assert result["prize"] == 0.25
    assert result["now"] == 0.25


def test_compute_document_frequency():
    documents = [
        "free money",
        "free prize",
        "hello friend",
    ]

    vocabulary = build_vocabulary(documents)

    result = compute_document_frequency(
        documents,
        vocabulary,
    )

    assert result["free"] == 2
    assert result["money"] == 1
    assert result["prize"] == 1
    assert result["hello"] == 1


def test_compute_idf():
    document_frequency = {
        "free": 2,
        "money": 1,
    }

    result = compute_idf(
        document_frequency,
        num_documents=3,
    )

    assert np.isclose(
        result["free"],
        np.log(3 / 2),
    )

    assert np.isclose(
        result["money"],
        np.log(3),
    )


def test_tfidf_vector_length():
    documents = [
        "free money",
        "free prize",
    ]

    vocabulary = build_vocabulary(documents)

    df = compute_document_frequency(
        documents,
        vocabulary,
    )

    idf = compute_idf(
        df,
        len(documents),
    )

    vector = compute_tfidf_vector(
        "free prize",
        vocabulary,
        idf,
    )

    assert len(vector) == len(vocabulary)


def test_transform_documents_shape():
    documents = [
        "free money",
        "free prize",
        "hello friend",
    ]

    vocabulary = build_vocabulary(documents)

    df = compute_document_frequency(
        documents,
        vocabulary,
    )

    idf = compute_idf(
        df,
        len(documents),
    )

    matrix = transform_documents(
        documents,
        vocabulary,
        idf,
    )

    assert matrix.shape == (
        len(documents),
        len(vocabulary),
    )

    assert isinstance(matrix, np.ndarray)


def test_tfidf_contains_no_nan_or_inf():
    documents = [
        "free money",
        "free prize",
        "hello friend",
    ]

    vocabulary = build_vocabulary(documents)

    df = compute_document_frequency(
        documents,
        vocabulary,
    )

    idf = compute_idf(
        df,
        len(documents),
    )

    matrix = transform_documents(
        documents,
        vocabulary,
        idf,
    )

    assert not np.isnan(matrix).any()
    assert not np.isinf(matrix).any()