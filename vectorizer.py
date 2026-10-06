"""Bag-of-Words dan TF-IDF yang diimplementasikan manual."""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from preprocessing import get_indonesian_stemmer, tokenize


class TextVectorizer:
    """Ubah kumpulan teks menjadi vektor angka.

    Args:
        method: "bow" (jumlah kata) atau "tfidf".
        stopwords: daftar stopword bebas (bahasa apa pun).
        stem_id: True untuk stemming bahasa Indonesia (butuh Sastrawi).
    """

    def __init__(
        self,
        method: str = "tfidf",
        stopwords: Iterable[str] | None = None,
        stem_id: bool = False,
    ):
        if method not in ("bow", "tfidf"):
            raise ValueError('method harus "bow" atau "tfidf".')
        self.method = method
        self.stopwords = list(stopwords) if stopwords else []
        self.stemmer = get_indonesian_stemmer() if stem_id else None
        self.vocabulary_: list[str] = []
        self.idf_: dict[str, float] = {}
        # Data perantara (untuk keperluan penjelasan langkah demi langkah)
        self.docs_: list[list[str]] = []
        self.counts_: np.ndarray = np.zeros((0, 0))
        self.df_: dict[str, int] = {}
        self.n_docs_: int = 0

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        docs = [tokenize(t, self.stopwords, stemmer=self.stemmer) for t in texts]
        self.vocabulary_ = sorted({w for d in docs for w in d})
        index = {w: i for i, w in enumerate(self.vocabulary_)}
        n_docs = len(docs)
        self.docs_, self.n_docs_ = docs, n_docs

        # IDF (versi smoothing): ln((1 + N) / (1 + df)) + 1
        df = Counter(w for d in docs for w in set(d))
        self.idf_ = {
            w: math.log((1 + n_docs) / (1 + df[w])) + 1 for w in self.vocabulary_
        }

        self.df_ = {w: df[w] for w in self.vocabulary_}
        matrix = np.zeros((n_docs, len(self.vocabulary_)))
        self.counts_ = np.zeros((n_docs, len(self.vocabulary_)))
        for i, doc in enumerate(docs):
            counts = Counter(doc)
            total = len(doc)
            for word, c in counts.items():
                self.counts_[i, index[word]] = c
                if self.method == "bow":
                    matrix[i, index[word]] = c
                else:
                    tf = c / total                      # TF = frekuensi / total kata
                    matrix[i, index[word]] = tf * self.idf_[word]
        return matrix
