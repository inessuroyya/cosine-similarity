"""CLI Cosine Similarity.

Contoh:
  python main.py vectors "1,2,3" "2,4,6"
  python main.py text "kucing makan ikan" "ikan dimakan kucing" --method tfidf
  python main.py files data/sample/doc1.txt data/sample/doc2.txt data/sample/doc3.txt
  python main.py csv data/sample/vectors.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from similarity import (
    compare_with_sklearn,
    cosine_similarity,
    rank_pairs,
    similarity_matrix,
)
from vectorizer import TextVectorizer


def _load_stopwords(path: str | None) -> list[str]:
    if not path:
        return []
    return Path(path).read_text(encoding="utf-8").split()


def _report(vectors, labels) -> None:
    pd.set_option("display.precision", 4)
    matrix = similarity_matrix(vectors, labels)
    print("\n=== Matriks Kemiripan ===")
    print(matrix.round(4))
    if len(labels) > 1:
        print("\n=== Ranking Pasangan ===")
        print(rank_pairs(matrix).round(4).to_string(index=False))
    diff = compare_with_sklearn(vectors)
    print(f"\nVerifikasi vs scikit-learn -> selisih maksimum: {diff:.2e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hitung kemiripan dengan Cosine Similarity.")
    sub = p.add_subparsers(dest="mode", required=True)

    v = sub.add_parser("vectors", help="Vektor angka dipisah koma, mis. '1,2,3'")
    v.add_argument("vectors", nargs="+")

    for name, helptext in (("text", "Teks langsung"), ("files", "File .txt")):
        t = sub.add_parser(name, help=helptext)
        t.add_argument("items", nargs="+")
        t.add_argument("--method", choices=["bow", "tfidf"], default="tfidf")
        t.add_argument("--stopwords", help="File stopword (satu/lebih kata per baris)")
        t.add_argument("--stem-id", action="store_true",
                       help="Aktifkan stemming bahasa Indonesia (Sastrawi)")

    c = sub.add_parser("csv", help="CSV: tiap baris = satu vektor (kolom pertama boleh label)")
    c.add_argument("path")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "vectors":
        vecs = [[float(x) for x in s.split(",")] for s in args.vectors]
        if len(vecs) == 2:
            print(f"Cosine similarity: {cosine_similarity(vecs[0], vecs[1]):.6f}")
        _report(vecs, [f"V{i + 1}" for i in range(len(vecs))])

    elif args.mode in ("text", "files"):
        if args.mode == "files":
            texts = [Path(f).read_text(encoding="utf-8") for f in args.items]
            labels = [Path(f).name for f in args.items]
        else:
            texts = args.items
            labels = [f"Teks {i + 1}" for i in range(len(texts))]
        vec = TextVectorizer(args.method, _load_stopwords(args.stopwords), args.stem_id)
        _report(vec.fit_transform(texts), labels)

    elif args.mode == "csv":
        df = pd.read_csv(args.path)
        labels = [f"Baris {i + 1}" for i in range(len(df))]
        if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            labels = df.iloc[:, 0].astype(str).tolist()
            df = df.iloc[:, 1:]
        _report(df.to_numpy(dtype=float), labels)


if __name__ == "__main__":
    main()
