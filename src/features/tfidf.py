import re
import math
import numpy as np

def tokenize(text: str) -> list[str]:
    """
    Convert a cleaned text string into a list of tokens.
    """

    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())


def build_vocabulary(documents: list[str]) -> dict[str, int]:
    """
    Build a vocabulary from training documents.
    """

    vocabulary = {}

    for document in documents:
        tokens = tokenize(document)

        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    return vocabulary


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """
    Compute term frequency for a single document.

    TF = number of occurrences of a term / total number of terms.
    """

    if not tokens:
        return {}

    term_counts = {}

    for token in tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    total_terms = len(tokens)

    tf = {
        token: count / total_terms
        for token, count in term_counts.items()
    }

    return tf

def compute_document_frequency(
    documents: list[str],
    vocabulary: dict[str, int]
) -> dict[str, int]:
    """
    Calculate how many documents contain each vocabulary token.

    A token is counted at most once per document.
    """

    document_frequency = {
        token: 0
        for token in vocabulary
    }

    for document in documents:
        tokens = set(tokenize(document))

        for token in tokens:
            if token in vocabulary:
                document_frequency[token] += 1

    return document_frequency

def compute_idf(
    document_frequency: dict[str, int],
    num_documents: int
) -> dict[str, float]:
    """
    Compute inverse document frequency for each vocabulary token.

    IDF = log(N / DF)

    where:
        N  = total number of documents
        DF = number of documents containing the token
    """

    idf = {}

    for token, df in document_frequency.items():
        idf[token] = math.log(num_documents / df)

    return idf

def compute_tfidf_vector(
    document: str,
    vocabulary: dict[str, int],
    idf: dict[str, float]
) -> list[float]:
    """
    Convert one document into a TF-IDF vector.
    """

    tokens = tokenize(document)
    tf = compute_tf(tokens)

    vector = [0.0] * len(vocabulary)

    for token, tf_value in tf.items():
        if token in vocabulary:
            index = vocabulary[token]
            vector[index] = tf_value * idf[token]

    return vector

def transform_documents(
    documents: list[str],
    vocabulary: dict[str, int],
    idf: dict[str, float]
) -> np.ndarray:
    """
    Transform multiple documents into a TF-IDF matrix.
    """

    matrix = np.zeros(
        (len(documents), len(vocabulary)),
        dtype=float
    )

    for row_index, document in enumerate(documents):
        vector = compute_tfidf_vector(
            document,
            vocabulary,
            idf
        )

        matrix[row_index] = vector

    return matrix