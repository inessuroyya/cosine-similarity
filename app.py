"""Antarmuka web Streamlit untuk menghitung kemiripan dua kalimat atau lebih.

Jalankan:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st

from similarity import (
    compare_with_sklearn,
    cosine_similarity,
    rank_pairs,
    similarity_matrix,
)
from vectorizer import TextVectorizer

st.set_page_config(page_title="Kemiripan Kalimat - Cosine Similarity", page_icon="📐")

CONTOH = [
    "saya akan ke mekkah bersama keluarga besar saya",
    "saya ingin ke pasar dengan keluarga",
]


# ----------------------------------------------------------------- utilitas
def f4(x: float) -> str:
    return f"{x:.4f}"


def gabung(suku: list[str], maks: int = 10) -> str:
    """Gabungkan suku penjumlahan (LaTeX); dipotong jika terlalu panjang."""
    if len(suku) > maks:
        return " + ".join(suku[:maks]) + r" + \ldots"
    return " + ".join(suku)


def daftar(tokens: list[str]) -> str:
    return ", ".join(tokens) if tokens else "(kosong)"


def warna(v: float) -> str:
    """Warna latar sel matriks: makin mirip makin biru."""
    alpha = float(np.clip(v, 0.0, 1.0)) * 0.65
    return f"background-color: rgba(31, 119, 180, {alpha:.2f})"


def label_kemiripan(skor: float) -> str:
    if skor >= 0.8:
        return "sangat mirip"
    if skor >= 0.5:
        return "cukup mirip"
    if skor >= 0.2:
        return "kemiripan rendah"
    return "hampir tidak mirip"


# ------------------------------------------------------------------ tampilan
st.title("Sistem Perhitungan Kemiripan Kalimat Menggunakan Cosine Similarity")
st.write(
    "Masukkan dua kalimat atau lebih, lalu tekan tombol **Hitung Kemiripan** untuk "
    "melihat hasil perhitungan Cosine Similarity secara otomatis."
)

jumlah = int(st.number_input("Jumlah kalimat", min_value=2, max_value=10, value=2, step=1))
kalimat = [
    st.text_area(
        f"Kalimat {i + 1}",
        value=CONTOH[i] if i < len(CONTOH) else "",
        key=f"kalimat_{i}",
    )
    for i in range(jumlah)
]

with st.expander("Pengaturan (opsional)"):
    metode_label = st.radio(
        "Metode pembobotan kata",
        ["TF-IDF", "Bag-of-Words (jumlah kata)"],
        horizontal=True,
    )
    stem_id = st.checkbox(
        "Gunakan stemming bahasa Indonesia (Sastrawi)",
        help="Mengubah kata ke bentuk dasar, mis. 'dimakan' → 'makan'. Hanya untuk bahasa Indonesia.",
    )
    stop_input = st.text_input(
        "Stopword (pisahkan dengan spasi atau koma)",
        placeholder="contoh: yang dan di ke dengan",
    )


def hitung() -> dict | None:
    """Proses semua kalimat; kembalikan hasil atau None jika input bermasalah."""
    if any(not k.strip() for k in kalimat):
        st.warning("Semua kalimat harus diisi.")
        return None
    metode = "tfidf" if metode_label == "TF-IDF" else "bow"
    stopwords = stop_input.replace(",", " ").split()
    try:
        vec = TextVectorizer(metode, stopwords, stem_id)
    except ImportError as err:
        st.error(str(err))
        return None
    with st.spinner("Memproses kalimat..."):
        matriks = vec.fit_transform(kalimat)
    if matriks.shape[1] == 0:
        st.error("Tidak ada kata yang tersisa setelah preprocessing. Periksa input/stopword.")
        return None
    return {
        "vec": vec,
        "matriks": matriks,
        "metode": metode,
        "stopwords": stopwords,
        "stem_id": stem_id,
        "labels": [f"Kalimat {i + 1}" for i in range(len(kalimat))],
    }


def jelaskan(res: dict, i: int, j: int) -> None:
    """Penjelasan langkah demi langkah untuk pasangan kalimat ke-i dan ke-j."""
    vec, M = res["vec"], res["matriks"]
    kata = vec.vocabulary_
    n = len(kata)
    N = vec.n_docs_
    pakai_tfidf = res["metode"] == "tfidf"
    a, b = M[i], M[j]
    skor = cosine_similarity(a, b)
    ni, nj = i + 1, j + 1

    # baris tabel: hanya kata yang muncul di salah satu dari dua kalimat terpilih
    baris = [k for k in range(n) if vec.counts_[i, k] > 0 or vec.counts_[j, k] > 0]

    st.divider()
    st.subheader("Penjelasan Langkah demi Langkah")
    st.write(
        f"Ide dasarnya: setiap kalimat diubah menjadi deretan angka (**vektor**), lalu "
        f"dihitung seberapa searah dua vektor itu. Makin searah, makin mirip. "
        f"Angka 1 berarti searah sempurna, angka 0 berarti tidak ada kata yang sama. "
        f"Penjelasan di bawah memakai pasangan **Kalimat {ni}** dan **Kalimat {nj}**."
    )

    # Langkah 1
    st.markdown("#### Langkah 1: Pra-pemrosesan teks")
    proses = "huruf diubah menjadi kecil, tanda baca dan angka dibuang, lalu kalimat dipecah per kata"
    if res["stopwords"]:
        proses += ", kata stopword dibuang"
    if res["stem_id"]:
        proses += ", dan setiap kata diubah ke bentuk dasar (stemming)"
    st.write(f"Kalimat dirapikan dulu: {proses}.")
    st.markdown("\n".join(f"- **Kalimat {k + 1}:** {daftar(d)}" for k, d in enumerate(vec.docs_)))

    # Langkah 2
    st.markdown("#### Langkah 2: Menyusun kosakata")
    st.write(
        f"Semua kata unik dari seluruh {N} kalimat digabung menjadi satu daftar yang disebut "
        f"**kosakata**. Urutan daftar ini menjadi urutan angka pada vektor. "
        f"Jumlahnya **{n} kata**:"
    )
    st.markdown(f"`{daftar(kata)}`")

    # Langkah 3
    if pakai_tfidf:
        st.markdown("#### Langkah 3: Memberi bobot pada setiap kata (TF-IDF)")
        st.write(
            "Setiap kata diberi angka agar kata yang penting bobotnya lebih besar. "
            "Ada tiga bagian hitungan:"
        )
        st.latex(r"TF = \frac{\text{jumlah kemunculan kata}}{\text{total kata dalam kalimat}}")
        st.write("**TF** (Term Frequency) menunjukkan seberapa sering kata muncul di satu kalimat.")
        st.latex(r"IDF = \ln\left(\frac{1 + N}{1 + df}\right) + 1")
        st.write(
            f"**IDF** (Inverse Document Frequency) menilai kelangkaan kata. *N* = jumlah "
            f"seluruh kalimat (di sini {N}), *df* = jumlah kalimat yang mengandung kata "
            "tersebut. Kata yang ada di banyak kalimat bobotnya lebih kecil, karena tidak "
            "membantu membedakan kalimat."
        )
        st.latex(r"TF\text{-}IDF = TF \times IDF")

        panjang = [len(d) for d in vec.docs_]
        tf = np.array(
            [vec.counts_[x] / panjang[x] if panjang[x] else vec.counts_[x] for x in (i, j)]
        )
        idf = np.array([vec.idf_[w] for w in kata])
        dfv = np.array([vec.df_[w] for w in kata])

        contoh = next((k for k in baris if a[k] > 0), None)
        if contoh is not None:
            st.markdown(f"**Contoh perhitungan untuk kata “{kata[contoh]}” pada Kalimat {ni}:**")
            st.markdown(
                f"- TF = {int(vec.counts_[i, contoh])} / {panjang[i]} = **{f4(tf[0, contoh])}**  \n"
                f"- IDF = ln((1 + {N}) / (1 + {int(dfv[contoh])})) + 1 = **{f4(idf[contoh])}**  \n"
                f"- TF-IDF = {f4(tf[0, contoh])} × {f4(idf[contoh])} = **{f4(a[contoh])}**"
            )
        st.write("Dengan cara yang sama, semua kata dihitung:")
        tabel_bobot = pd.DataFrame(
            {
                "Kata": [kata[k] for k in baris],
                f"Jumlah (K{ni})": [int(vec.counts_[i, k]) for k in baris],
                f"Jumlah (K{nj})": [int(vec.counts_[j, k]) for k in baris],
                f"TF (K{ni})": [tf[0, k] for k in baris],
                f"TF (K{nj})": [tf[1, k] for k in baris],
                "df": [int(dfv[k]) for k in baris],
                "IDF": [idf[k] for k in baris],
                f"TF-IDF (K{ni})": [a[k] for k in baris],
                f"TF-IDF (K{nj})": [b[k] for k in baris],
            }
        )
        st.dataframe(tabel_bobot.round(4), hide_index=True)
    else:
        st.markdown("#### Langkah 3: Memberi bobot pada setiap kata (Bag-of-Words)")
        st.write(
            "Setiap kata cukup dihitung berapa kali muncul di tiap kalimat. "
            "Angka inilah yang menjadi isi vektor."
        )
        tabel_bobot = pd.DataFrame(
            {
                "Kata": [kata[k] for k in baris],
                f"Jumlah (K{ni})": [int(a[k]) for k in baris],
                f"Jumlah (K{nj})": [int(b[k]) for k in baris],
            }
        )
        st.dataframe(tabel_bobot, hide_index=True)
    if len(baris) < n:
        st.caption(
            f"Tabel hanya menampilkan kata yang muncul di Kalimat {ni} atau {nj}. "
            "Kata lain bernilai 0 untuk pasangan ini."
        )

    # Langkah 4
    st.markdown("#### Langkah 4: Menyusun vektor")
    st.write("Bobot tiap kata, berurutan sesuai kosakata, menjadi vektor tiap kalimat:")
    fmt = (lambda v: ", ".join(f4(x) for x in v)) if pakai_tfidf else (
        lambda v: ", ".join(str(int(x)) for x in v)
    )
    st.code(f"A (Kalimat {ni}) = [{fmt(a)}]\nB (Kalimat {nj}) = [{fmt(b)}]", language=None)

    # Langkah 5
    st.markdown("#### Langkah 5: Rumus Cosine Similarity")
    st.latex(r"\cos(\theta) = \frac{A \cdot B}{\lVert A \rVert \times \lVert B \rVert}")
    st.markdown(
        "- **A · B** (dot product): kalikan angka yang posisinya sama pada A dan B, lalu jumlahkan semuanya.  \n"
        "- **‖A‖** dan **‖B‖** (panjang vektor): kuadratkan setiap angka, jumlahkan, lalu ambil akar kuadratnya.  \n"
        "- Hasil bagi inilah skor kemiripan."
    )

    # Langkah 6
    dot = float(np.dot(a, b))
    na = float(np.sqrt(np.sum(a ** 2)))
    nb = float(np.sqrt(np.sum(b ** 2)))

    st.markdown("#### Langkah 6: Menghitung dot product (A · B)")
    bersama = [k for k in range(n) if a[k] > 0 and b[k] > 0]
    if bersama:
        st.write(
            "Kata yang hanya ada di satu kalimat bernilai 0 di kalimat lainnya, sehingga hasil "
            "perkaliannya 0. Jadi hanya **kata yang sama di kedua kalimat** yang berpengaruh "
            f"(yaitu: {', '.join(kata[k] for k in bersama)})."
        )
        tabel_sama = pd.DataFrame(
            {
                "Kata yang sama": [kata[k] for k in bersama],
                f"Jumlah di K{ni}": [int(vec.counts_[i, k]) for k in bersama],
                f"Jumlah di K{nj}": [int(vec.counts_[j, k]) for k in bersama],
                "Bobot A": [a[k] for k in bersama],
                "Bobot B": [b[k] for k in bersama],
                "A × B": [a[k] * b[k] for k in bersama],
            }
        )
        st.dataframe(tabel_sama.round(4), hide_index=True)
        saja_i = [kata[k] for k in range(n) if a[k] > 0 and b[k] == 0]
        saja_j = [kata[k] for k in range(n) if b[k] > 0 and a[k] == 0]
        st.caption(
            f"Hanya di Kalimat {ni}: {daftar(saja_i)}  |  Hanya di Kalimat {nj}: {daftar(saja_j)}  "
            "(tidak ikut menambah dot product)."
        )
        suku = [f"({f4(a[k])} \\times {f4(b[k])})" for k in bersama]
        st.latex(r"A \cdot B = " + gabung(suku) + f" = {f4(dot)}")
    else:
        st.write("Tidak ada kata yang sama di kedua kalimat, sehingga semua hasil perkalian bernilai 0.")
        st.latex(r"A \cdot B = 0")

    # Langkah 7
    st.markdown("#### Langkah 7: Menghitung panjang vektor (‖A‖ dan ‖B‖)")
    for nama, v, panj in (("A", a, na), ("B", b, nb)):
        ada = [k for k in range(n) if v[k] > 0]
        if ada:
            kuadrat = [f"{f4(v[k])}^2" for k in ada]
            jumlah_kuadrat = float(np.sum(v ** 2))
            st.latex(
                rf"\lVert {nama} \rVert = \sqrt{{{gabung(kuadrat)}}} = "
                rf"\sqrt{{{f4(jumlah_kuadrat)}}} = {f4(panj)}"
            )
        else:
            st.latex(rf"\lVert {nama} \rVert = 0")
    st.caption("Kata yang bernilai 0 tidak ditulis karena tidak mengubah hasil penjumlahan.")

    # Langkah 8
    st.markdown("#### Langkah 8: Hasil akhir")
    if na == 0 or nb == 0:
        st.warning(
            "Salah satu kalimat tidak memiliki kata tersisa setelah pra-pemrosesan "
            "(vektornya nol), sehingga pembagian tidak bisa dilakukan. Skor ditetapkan 0."
        )
        st.latex(r"\cos(\theta) = 0")
    else:
        st.latex(
            rf"\cos(\theta) = \frac{{{f4(dot)}}}{{{f4(na)} \times {f4(nb)}}} "
            rf"= \frac{{{f4(dot)}}}{{{f4(na * nb)}}} = \mathbf{{{f4(skor)}}}"
        )
    st.caption("Angka dalam penjelasan dibulatkan 4 desimal, sehingga selisih kecil di digit terakhir mungkin terjadi.")

    with st.expander("Lihat tabel perkalian lengkap per kata"):
        tabel = pd.DataFrame(
            {
                "Kata": kata,
                "A": a,
                "B": b,
                "A × B": a * b,
                "A²": a ** 2,
                "B²": b ** 2,
            }
        )
        st.dataframe(tabel.round(4), hide_index=True)


def tampilkan(res: dict) -> None:
    vec, M, labels = res["vec"], res["matriks"], res["labels"]
    n_kal = len(labels)
    sim = similarity_matrix(M, labels)
    pasangan = rank_pairs(sim)

    st.divider()
    st.subheader("Hasil")

    if n_kal > 2:
        st.markdown("**Matriks kemiripan** (makin biru makin mirip)")
        gaya = sim.style.format("{:.4f}")
        gaya = gaya.map(warna) if hasattr(gaya, "map") else gaya.applymap(warna)
        st.dataframe(gaya)

        st.markdown("**Peringkat pasangan kalimat** (dari yang paling mirip)")
        tampil = pd.DataFrame(
            {
                "Peringkat": range(1, len(pasangan) + 1),
                "Pasangan": pasangan["item_1"] + " & " + pasangan["item_2"],
                "Skor": pasangan["similarity"],
            }
        )
        st.dataframe(
            tampil,
            hide_index=True,
            column_config={
                "Skor": st.column_config.ProgressColumn(
                    "Skor", min_value=0.0, max_value=1.0, format="%.4f"
                )
            },
        )
        terbaik = pasangan.iloc[0]
        st.success(
            f"Pasangan paling mirip: **{terbaik['item_1']}** dan **{terbaik['item_2']}** "
            f"dengan skor **{terbaik['similarity']:.4f}**."
        )
        st.caption(
            "Catatan: pada metode TF-IDF, bobot sebuah kata ikut dipengaruhi seluruh kalimat "
            "yang dimasukkan, sehingga skor sebuah pasangan bisa berubah jika kalimat lain ditambah."
        )

        pilihan = st.selectbox(
            "Pilih pasangan kalimat untuk dilihat perhitungannya",
            options=list(range(len(pasangan))),
            format_func=lambda k: (
                f"{pasangan.at[k, 'item_1']} & {pasangan.at[k, 'item_2']} "
                f"(skor {pasangan.at[k, 'similarity']:.4f})"
            ),
            key=f"pasangan_{n_kal}",
        )
        i = labels.index(pasangan.at[pilihan, "item_1"])
        j = labels.index(pasangan.at[pilihan, "item_2"])
    else:
        i, j = 0, 1

    skor = cosine_similarity(M[i], M[j])
    st.metric(
        f"Cosine Similarity (Kalimat {i + 1} & Kalimat {j + 1})",
        f"{skor:.4f}",
        help="0 = tidak mirip, 1 = identik",
    )
    st.progress(float(np.clip(skor, 0.0, 1.0)))
    st.caption(f"Tingkat kemiripan: {skor * 100:.2f}%")
    sudut = float(np.degrees(np.arccos(np.clip(skor, -1.0, 1.0))))
    st.info(
        f"**Interpretasi:** kedua kalimat tergolong **{label_kemiripan(skor)}** "
        f"(sudut antar vektor ≈ {sudut:.1f}°). "
        "Pengelompokan ini hanya pedoman umum, bukan aturan baku."
    )
    st.caption(
        f"Verifikasi dengan scikit-learn: selisih maksimum seluruh matriks "
        f"{compare_with_sklearn(M):.1e}."
    )

    jelaskan(res, i, j)


if st.button("Hitung Kemiripan"):
    hasil = hitung()
    if hasil is None:
        st.session_state.pop("hasil", None)
    else:
        st.session_state["hasil"] = hasil

if "hasil" in st.session_state:
    tampilkan(st.session_state["hasil"])
