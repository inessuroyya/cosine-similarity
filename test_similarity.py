import numpy as np
import pytest

from src.similarity import (
    compare_with_sklearn,
    cosine_similarity,
    rank_pairs,
    similarity_matrix,
)
from src.vectorizer import TextVectorizer


def test_identical_vectors():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_scaled_vectors_still_one():
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_orthogonal_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_opposite_vectors():
    assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)


def test_zero_vector_returns_zero():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_different_length_raises():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])


def test_empty_raises():
    with pytest.raises(ValueError):
        cosine_similarity([], [])


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    data = rng.random((6, 10))
    assert compare_with_sklearn(data) < 1e-9


def test_matrix_diagonal_and_symmetry():
    m = similarity_matrix([[1, 2], [3, 4], [5, 1]])
    assert np.allclose(np.diag(m.to_numpy()), 1.0)
    assert np.allclose(m.to_numpy(), m.to_numpy().T)


def test_rank_pairs_sorted():
    m = similarity_matrix([[1, 0], [1, 0.1], [0, 1]])
    ranked = rank_pairs(m)
    assert ranked["similarity"].is_monotonic_decreasing


def test_text_identical_docs():
    mat = TextVectorizer("tfidf").fit_transform(["kucing makan ikan", "kucing makan ikan"])
    assert cosine_similarity(mat[0], mat[1]) == pytest.approx(1.0)


def test_text_no_common_words():
    mat = TextVectorizer("bow").fit_transform(["apel jeruk", "mobil motor"])
    assert cosine_similarity(mat[0], mat[1]) == pytest.approx(0.0)


def test_stopwords_removed():
    vec = TextVectorizer("bow", stopwords=["dan"])
    vec.fit_transform(["kucing dan anjing"])
    assert "dan" not in vec.vocabulary_
