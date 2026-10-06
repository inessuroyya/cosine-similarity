"""Fungsi inti Cosine Similarity.

Rumus: cos(θ) = (A · B) / (||A|| × ||B||)
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Hitung cosine similarity dua vektor.

    Returns:
        Nilai antara -1 dan 1. Jika salah satu vektor nol, hasilnya 0.0
        (menghindari pembagian dengan nol).

    Raises:
        ValueError: jika panjang vektor berbeda atau vektor kosong.
    """
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    if va.ndim != 1 or vb.ndim != 1:
        raise ValueError("Input harus berupa vektor 1 dimensi.")
    if va.shape != vb.shape:
        raise ValueError(f"Panjang vektor berbeda: {va.shape[0]} vs {vb.shape[0]}.")
    if va.size == 0:
        raise ValueError("Vektor tidak boleh kosong.")

    dot = float(np.dot(va, vb))                 # A · B
    norm_a = float(np.sqrt(np.sum(va ** 2)))    # ||A||
    norm_b = float(np.sqrt(np.sum(vb ** 2)))    # ||B||
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    # Clip untuk menghilangkan galat floating point (mis. 1.0000000002)
    return float(np.clip(dot / (norm_a * norm_b), -1.0, 1.0))


def similarity_matrix(
    vectors: Sequence[Sequence[float]], labels: Sequence[str] | None = None
) -> pd.DataFrame:
    """Matriks kemiripan pairwise (n x n) dihitung secara manual dengan NumPy."""
    m = np.asarray(vectors, dtype=float)
    if m.ndim != 2:
        raise ValueError("Input harus berupa matriks 2 dimensi (daftar vektor).")
    norms = np.linalg.norm(m, axis=1)
    dots = m @ m.T
    denom = np.outer(norms, norms)
    # Bagi hanya jika penyebut != 0, selain itu 0.0
    sim = np.divide(dots, denom, out=np.zeros_like(dots), where=denom != 0)
    sim = np.clip(sim, -1.0, 1.0)
    names = list(labels) if labels is not None else [f"Item {i + 1}" for i in range(len(m))]
    if len(names) != len(m):
        raise ValueError("Jumlah label harus sama dengan jumlah vektor.")
    return pd.DataFrame(sim, index=names, columns=names)


def rank_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    """Urutkan pasangan item dari yang paling mirip ke yang paling tidak mirip."""
    rows = [
        (matrix.index[i], matrix.columns[j], float(matrix.iat[i, j]))
        for i, j in combinations(range(len(matrix)), 2)
    ]
    df = pd.DataFrame(rows, columns=["item_1", "item_2", "similarity"])
    return df.sort_values("similarity", ascending=False).reset_index(drop=True)


def compare_with_sklearn(vectors: Sequence[Sequence[float]]) -> float:
    """Verifikasi: selisih maksimum antara implementasi manual dan scikit-learn."""
    from sklearn.metrics.pairwise import cosine_similarity as sk_cos

    manual = similarity_matrix(vectors).to_numpy()
    reference = sk_cos(np.asarray(vectors, dtype=float))
    return float(np.max(np.abs(manual - reference)))
