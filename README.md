# Cosine Similarity (Python)

Proyek umum untuk menghitung kemiripan dengan rumus cosine, untuk vektor angka maupun teks.

    cos(θ) = (A · B) / (||A|| × ||B||)

## Instalasi
    pip install -r requirements.txt

## Antarmuka Web (Streamlit)
    streamlit run app.py

Buka alamat yang tampil di terminal (biasanya http://localhost:8501).

Aplikasi menerima 2 sampai 10 kalimat. Untuk lebih dari 2 kalimat, hasilnya berupa matriks kemiripan,
peringkat pasangan, dan penjelasan langkah demi langkah untuk pasangan yang dipilih.

## Cara Pakai (CLI)
    # Dua/lebih vektor angka
    python main.py vectors "1,2,3" "2,4,6"

    # Teks langsung (TF-IDF atau bow)
    python main.py text "kucing makan ikan" "ikan dimakan kucing" --method tfidf

    # Stemming bahasa Indonesia (Sastrawi): tambahkan --stem-id
    python main.py text "dia makan ikan" "ikan dimakan dia" --stem-id

    # File .txt dengan stopword buatan sendiri
    python main.py files data/sample/doc1.txt data/sample/doc2.txt data/sample/doc3.txt --stopwords data/sample/stopwords.txt

    # CSV (kolom pertama non-angka dianggap label)
    python main.py csv data/sample/vectors.csv

## Pakai sebagai library
    from src.similarity import cosine_similarity
    cosine_similarity([1, 2, 3], [2, 4, 6])   # 1.0

## Contoh Hitung Manual
A = (1, 2, 3), B = (4, 5, 6)
- A · B = 1*4 + 2*5 + 3*6 = 32
- ||A|| = sqrt(1+4+9) = 3.7417
- ||B|| = sqrt(16+25+36) = 8.7750
- cos = 32 / (3.7417 * 8.7750) = 0.9746

## Tes
    pytest -v

## Struktur
- `src/similarity.py`    cosine, matriks pairwise, ranking, verifikasi sklearn
- `src/vectorizer.py`    Bag-of-Words & TF-IDF manual
- `src/preprocessing.py` pembersihan teks (stopword bisa diatur sendiri)
- `app.py`               Antarmuka web Streamlit
- `main.py`              CLI
- `tests/`               unit test
