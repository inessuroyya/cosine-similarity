import pytest

pytest.importorskip("Sastrawi")

from src.similarity import cosine_similarity
from src.vectorizer import TextVectorizer


def test_stemming_raises_similarity():
    teks = ["kucing makan ikan", "ikan dimakan kucing"]
    tanpa = TextVectorizer("bow").fit_transform(teks)
    dengan = TextVectorizer("bow", stem_id=True).fit_transform(teks)
    assert cosine_similarity(dengan[0], dengan[1]) > cosine_similarity(tanpa[0], tanpa[1])


def test_stemming_identical_after_stem():
    mat = TextVectorizer("bow", stem_id=True).fit_transform(["dimakan", "makanan"])
    assert cosine_similarity(mat[0], mat[1]) == pytest.approx(1.0)


def test_stemming_off_by_default():
    vec = TextVectorizer("bow")
    vec.fit_transform(["dimakan"])
    assert vec.vocabulary_ == ["dimakan"]
