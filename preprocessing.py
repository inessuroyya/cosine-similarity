"""Preprocessing teks umum (tidak terikat bahasa tertentu)."""
from __future__ import annotations

import re
import string
from functools import lru_cache
from typing import Callable, Iterable


def clean_text(text: str, remove_numbers: bool = True) -> str:
    """Case folding, hapus tanda baca (dan angka), rapikan spasi."""
    text = text.lower()
    # Ganti semua tanda baca dengan spasi
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def get_indonesian_stemmer() -> Callable[[str], str]:
    """Buat stemmer bahasa Indonesia (Sastrawi). Dibuat sekali lalu di-cache.

    Raises:
        ImportError: jika library Sastrawi belum terpasang.
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Stemming bahasa Indonesia memerlukan Sastrawi: pip install Sastrawi"
        ) from exc
    return StemmerFactory().create_stemmer().stem


def tokenize(
    text: str,
    stopwords: Iterable[str] | None = None,
    remove_numbers: bool = True,
    stemmer: Callable[[str], str] | None = None,
) -> list[str]:
    """Bersihkan teks, pecah per kata, lalu buang stopword (opsional).

    Args:
        text: teks mentah.
        stopwords: kumpulan kata yang diabaikan (bebas, sesuai bahasa pengguna).
        remove_numbers: hapus angka jika True.
        stemmer: fungsi stemming (opsional), mis. get_indonesian_stemmer().
    """
    stop = {w.lower() for w in stopwords} if stopwords else set()
    tokens = clean_text(text, remove_numbers).split()
    tokens = [t for t in tokens if t not in stop]   # stopword dibuang sebelum stemming
    if stemmer is not None:
        tokens = [stemmer(t) for t in tokens]
        tokens = [t for t in tokens if t and t not in stop]  # cek ulang setelah stemming
    return tokens
