"""Antarmuka web Streamlit untuk menghitung kemiripan dua kalimat atau lebih.

Jalankan:  streamlit run app.py
"""
import math
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.similarity import (
    compare_with_sklearn,
    cosine_similarity,
    rank_pairs,
    similarity_matrix,
)
from src.vectorizer import TextVectorizer

# ----------------------------------------------------------------- konfig halaman
st.set_page_config(
    page_title="Cosine Similarity Explorer - Kemiripan Kalimat",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONTOH = [
    
]

# ----------------------------------------------------------------- tema warna & CSS
THEMES = {
    "Indigo Modern": {
        "primary": "#6366f1",
        "accent": "#a855f7",
        "bg_gradient": "linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%)",
        "card_bg": "rgba(99, 102, 241, 0.08)",
        "card_border": "rgba(99, 102, 241, 0.25)",
        "colorscale": "Purples",
    },
    "Ocean Teal": {
        "primary": "#0d9488",
        "accent": "#0284c7",
        "bg_gradient": "linear-gradient(135deg, #0f172a 0%, #115e59 50%, #0284c7 100%)",
        "card_bg": "rgba(13, 148, 136, 0.08)",
        "card_border": "rgba(13, 148, 136, 0.25)",
        "colorscale": "Teal",
    },
    "Emerald Forest": {
        "primary": "#10b981",
        "accent": "#059669",
        "bg_gradient": "linear-gradient(135deg, #064e3b 0%, #047857 50%, #10b981 100%)",
        "card_bg": "rgba(16, 185, 129, 0.08)",
        "card_border": "rgba(16, 185, 129, 0.25)",
        "colorscale": "Greens",
    },
    "Sunset Rose": {
        "primary": "#e11d48",
        "accent": "#be185d",
        "bg_gradient": "linear-gradient(135deg, #4c0519 0%, #881337 50%, #e11d48 100%)",
        "card_bg": "rgba(225, 29, 72, 0.08)",
        "card_border": "rgba(225, 29, 72, 0.25)",
        "colorscale": "Reds",
    },
}


def inject_custom_css(theme_key: str):
    t = THEMES.get(theme_key, THEMES["Indigo Modern"])
    css = f"""
    <style>
    /* Custom Styling */
    .main-header {{
        background: {t['bg_gradient']};
        padding: 1.8rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }}
    .main-header h1 {{
        color: #ffffff !important;
        font-weight: 800 !important;
        margin-bottom: 0.3rem !important;
        font-size: 2.2rem !important;
    }}
    .main-header p {{
        color: #e2e8f0 !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }}
    .metric-card {{
        background: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.1);
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {t['primary']};
        margin: 0.2rem 0;
    }}
    .metric-label {{
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.8;
        font-weight: 600;
    }}
    .badge-label {{
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .badge-sangat-mirip {{ background-color: #10b981; color: white; }}
    .badge-cukup-mirip {{ background-color: #0284c7; color: white; }}
    .badge-rendah {{ background-color: #f59e0b; color: white; }}
    .badge-tidak-mirip {{ background-color: #ef4444; color: white; }}
    .step-box {{
        background: {t['card_bg']};
        border-left: 4px solid {t['primary']};
        padding: 1.2rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 1.2rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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


def get_badge_html(skor: float) -> str:
    lbl = label_kemiripan(skor)
    if skor >= 0.8:
        cls = "badge-sangat-mirip"
    elif skor >= 0.5:
        cls = "badge-cukup-mirip"
    elif skor >= 0.2:
        cls = "badge-rendah"
    else:
        cls = "badge-tidak-mirip"
    return f'<span class="badge-label {cls}">{lbl}</span>'


# ----------------------------------------------------------------- chart helper (Plotly)
def fig_gauge_chart(score: float, primary_color: str) -> go.Figure:
    """Gauge visual speedometer untuk skor Cosine Similarity."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".4f", "font": {"size": 36, "color": primary_color}},
            title={"text": "Skor Cosine Similarity", "font": {"size": 15, "color": "#888"}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#888"},
                "bar": {"color": primary_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 1,
                "bordercolor": "#ccc",
                "steps": [
                    {"range": [0.0, 0.2], "color": "rgba(239, 68, 68, 0.2)"},
                    {"range": [0.2, 0.5], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [0.5, 0.8], "color": "rgba(2, 132, 199, 0.2)"},
                    {"range": [0.8, 1.0], "color": "rgba(16, 185, 129, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": primary_color, "width": 4},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fig_angle_chart(score: float, primary_color: str) -> go.Figure:
    """Diagram 2D arah vektor untuk memperlihatkan sudut θ (arccos)."""
    sudut_rad = np.arccos(np.clip(score, -1.0, 1.0))
    sudut_deg = float(np.degrees(sudut_rad))

    fig = go.Figure()
    # Vector A (0 deg)
    fig.add_trace(
        go.Scatterpolar(
            r=[0, 1],
            theta=[0, 0],
            mode="lines+markers+text",
            name="Vektor A",
            line=dict(color="#3b82f6", width=4),
            marker=dict(size=8),
            text=["", "Vektor A"],
            textposition="top right",
        )
    )
    # Vector B (theta deg)
    fig.add_trace(
        go.Scatterpolar(
            r=[0, 1],
            theta=[0, sudut_deg],
            mode="lines+markers+text",
            name="Vektor B",
            line=dict(color=primary_color, width=4),
            marker=dict(size=8),
            text=["", f"Vektor B ({sudut_deg:.1f}°)"],
            textposition="top right",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.1]),
            angularaxis=dict(tickfont=dict(size=10), rotation=90, direction="counterclockwise"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.1),
        height=240,
        margin=dict(l=20, r=20, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fig_matrix_heatmap(sim_df: pd.DataFrame, colorscale: str) -> go.Figure:
    """Heatmap interaktif untuk matriks kemiripan banyak kalimat."""
    labels = list(sim_df.columns)
    z = sim_df.to_numpy()
    z_text = [[f"{val:.4f}" for val in row] for row in z]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 13, "weight": "bold"},
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Skor", len=0.8),
        )
    )
    fig.update_layout(
        title="Matriks Kemiripan Pairwise",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def fig_vector_bar_chart(
    words: list[str], val_a: np.ndarray, val_b: np.ndarray, label_a: str, label_b: str
) -> go.Figure:
    """Diagram batang perbandingan bobot kata antara dua vektor."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=words,
            y=val_a,
            name=label_a,
            marker_color="#3b82f6",
            text=[f"{v:.3f}" if v > 0 else "" for v in val_a],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=words,
            y=val_b,
            name=label_b,
            marker_color="#8b5cf6",
            text=[f"{v:.3f}" if v > 0 else "" for v in val_b],
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Perbandingan Bobot Kata Terpilih",
        xaxis_title="Kata",
        yaxis_title="Bobot (Weight)",
        height=320,
        margin=dict(l=20, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.1, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ----------------------------------------------------------------- sidebar & konfig tema
with st.sidebar:
    st.markdown("### 🎨 Pengaturan Tema & Visual")
    theme_choice = st.selectbox(
        "Pilih Tema Warna", list(THEMES.keys()), index=0, key="select_theme"
    )
    inject_custom_css(theme_choice)

    st.divider()
    st.markdown("### 📚 Panduan Singkat")
    st.markdown(
        """
        - **Cosine Similarity** mengukur sudut antara dua vektor teks (0 = beda total, 1 = identik).
        - **TF-IDF** memberi bobot tinggi pada kata unik/penting.
        - **Bag-of-Words** menghitung murni frekuensi kemunculan kata.
        - **Stemming (Sastrawi)** mengembalikan kata ke bentuk dasar bahasa Indonesia.
        """
    )
    st.caption("v2.0 • Antar-Muka Interaktif Terbuka")

# ----------------------------------------------------------------- header utama
st.markdown(
    """
    <div class="main-header">
        <h1>📐 Sistem Kalkulasi Cosine Similarity</h1>
        <p>Analisis kemiripan teks interaktif dengan visualisasi vektor &amp; langkah matematis transparan</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- input kalimat
col_inputs, col_settings = st.columns([3, 2])

with col_inputs:
    st.markdown("#### 📝 Kalimat Input")
    jumlah = int(
        st.number_input(
            "Jumlah Kalimat yang Dibandingkan",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            key="jumlah_input",
        )
    )

    kalimat = [
        st.text_area(
            f"Kalimat {i + 1}",
            value=CONTOH[i] if i < len(CONTOH) else "",
            key=f"kalimat_{i}",
            height=80,
            placeholder=f"Masukkan teks kalimat ke-{i + 1}...",
        )
        for i in range(jumlah)
    ]

with col_settings:
    st.markdown("#### ⚙️ Pengaturan Pembobotan")
    with st.container():
        metode_label = st.radio(
            "Metode Pembobotan Kata",
            ["TF-IDF", "Bag-of-Words (jumlah kata)"],
            horizontal=True,
            key="radio_metode",
        )
        stem_id = st.checkbox(
            "Gunakan Stemming Bahasa Indonesia (Sastrawi)",
            help="Mengubah kata ke bentuk dasar, contoh: 'dimakan' -> 'makan'.",
            key="checkbox_stem",
        )
        stop_input = st.text_input(
            "Stopword Kustom (pisahkan dengan spasi/koma)",
            placeholder="contoh: yang dan di ke dengan",
            key="input_stopword",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    btn_calc = st.button("🚀 Hitung Kemiripan Teks", type="primary", use_container_width=True)


# ----------------------------------------------------------------- logika proses
def hitung() -> dict | None:
    """Proses semua kalimat; kembalikan hasil atau None jika input bermasalah."""
    if any(not k.strip() for k in kalimat):
        st.warning("⚠️ Semua kolom kalimat harus diisi teks.")
        return None
    metode = "tfidf" if metode_label == "TF-IDF" else "bow"
    stopwords = stop_input.replace(",", " ").split()
    try:
        vec = TextVectorizer(metode, stopwords, stem_id)
    except ImportError as err:
        st.error(str(err))
        return None

    with st.spinner("⚡ Memproses vektor & menghitung Cosine Similarity..."):
        matriks = vec.fit_transform(kalimat)

    if matriks.shape[1] == 0:
        st.error("❌ Tidak ada kata yang tersisa setelah preprocessing. Periksa input/stopword.")
        return None

    return {
        "vec": vec,
        "matriks": matriks,
        "metode": metode,
        "stopwords": stopwords,
        "stem_id": stem_id,
        "labels": [f"Kalimat {i + 1}" for i in range(len(kalimat))],
    }


if btn_calc:
    hasil = hitung()
    if hasil is None:
        st.session_state.pop("hasil", None)
    else:
        st.session_state["hasil"] = hasil


# ----------------------------------------------------------------- fungsi jelaskan & tampilkan
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

    baris = [k for k in range(n) if vec.counts_[i, k] > 0 or vec.counts_[j, k] > 0]

    st.markdown(f"### 🔍 Detail Perhitungan Matematis: Kalimat {ni} vs Kalimat {nj}")
    st.write(
        f"Proses di bawah memperlihatkan konversi **Kalimat {ni}** dan **Kalimat {nj}** "
        f"menjadi vektor numerik hingga perhitungan akhir skor Cosine Similarity."
    )

    tab_s1, tab_s2, tab_s3, tab_s4, tab_s5 = st.tabs(
        [
            "1️⃣ Pra-pemrosesan Teks",
            "2️⃣ Kosakata & Bobot Kata",
            "3️⃣ Vektor & Visualisasi",
            "4️⃣ Dot Product (A · B)",
            "5️⃣ Norm & Hasil Akhir",
        ]
    )

    with tab_s1:
        st.markdown("#### Langkah 1: Pra-pemrosesan Teks")
        proses = "Huruf diubah menjadi kecil (case folding), tanda baca/angka dibuang, dan teks di-tokenize"
        if res["stopwords"]:
            proses += ", kata stopword dibuang"
        if res["stem_id"]:
            proses += ", lalu kata diubah ke kata dasar (stemming)"
        st.info(f"**Aturan Pra-pemrosesan:** {proses}.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Token Hasil Pra-pemrosesan Kalimat {ni}:**")
            st.code(daftar(vec.docs_[i]), language=None)
        with col_b:
            st.markdown(f"**Token Hasil Pra-pemrosesan Kalimat {nj}:**")
            st.code(daftar(vec.docs_[j]), language=None)

        st.markdown("**Ringkasan Hasil Semua Kalimat:**")
        for k_idx, d_tok in enumerate(vec.docs_):
            st.markdown(f"- **Kalimat {k_idx + 1}:** `{daftar(d_tok)}`")

    with tab_s2:
        st.markdown("#### Langkah 2: Menyusun Kosakata (Vocabulary)")
        st.write(
            f"Gabungan kata unik dari seluruh {N} kalimat menghasilkan **{n} kata unik** "
            f"sebagai dimensi ruang vektor:"
        )
        st.code(daftar(kata), language=None)

        st.markdown("#### Langkah 3: Perhitungan Bobot Kata")
        if pakai_tfidf:
            st.write("Menggunakan pembobotan **TF-IDF** (Term Frequency - Inverse Document Frequency):")
            st.latex(r"TF = \frac{\text{jumlah kemunculan kata}}{\text{total kata dalam kalimat}}")
            st.latex(r"IDF = \ln\left(\frac{1 + N}{1 + df}\right) + 1")
            st.latex(r"TF\text{-}IDF = TF \times IDF")

            panjang = [len(d) for d in vec.docs_]
            tf = np.array(
                [vec.counts_[x] / panjang[x] if panjang[x] else vec.counts_[x] for x in (i, j)]
            )
            idf = np.array([vec.idf_[w] for w in kata])
            dfv = np.array([vec.df_[w] for w in kata])

            contoh = next((k for k in baris if a[k] > 0), None)
            if contoh is not None:
                st.success(
                    f"**Contoh Perhitungan kata “{kata[contoh]}” pada Kalimat {ni}:**  \n"
                    f"- TF = {int(vec.counts_[i, contoh])} / {panjang[i]} = **{f4(tf[0, contoh])}**  \n"
                    f"- IDF = ln((1 + {N}) / (1 + {int(dfv[contoh])})) + 1 = **{f4(idf[contoh])}**  \n"
                    f"- TF-IDF = {f4(tf[0, contoh])} × {f4(idf[contoh])} = **{f4(a[contoh])}**"
                )

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
            st.dataframe(tabel_bobot.round(4), hide_index=True, use_container_width=True)
        else:
            st.write("Menggunakan pembobotan **Bag-of-Words** (murni jumlah frekuensi kata):")
            tabel_bobot = pd.DataFrame(
                {
                    "Kata": [kata[k] for k in baris],
                    f"Jumlah (K{ni})": [int(a[k]) for k in baris],
                    f"Jumlah (K{nj})": [int(b[k]) for k in baris],
                }
            )
            st.dataframe(tabel_bobot, hide_index=True, use_container_width=True)

    with tab_s3:
        st.markdown("#### Langkah 4: Array Vektor & Visualisasi Bobot Kata")
        st.write("Setiap kalimat direpresentasikan dalam bentuk susunan vektor numerik:")

        fmt = (lambda v: ", ".join(f4(x) for x in v)) if pakai_tfidf else (
            lambda v: ", ".join(str(int(x)) for x in v)
        )
        st.code(f"A (Kalimat {ni}) = [{fmt(a)}]\nB (Kalimat {nj}) = [{fmt(b)}]", language=None)

        words_subset = [kata[k] for k in baris]
        fig_bar = fig_vector_bar_chart(words_subset, a[baris], b[baris], f"Kalimat {ni}", f"Kalimat {nj}")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab_s4:
        st.markdown("#### Langkah 5 & 6: Formula & Menghitung Dot Product (A · B)")
        st.latex(r"\cos(\theta) = \frac{A \cdot B}{\lVert A \rVert \times \lVert B \rVert}")
        st.write(
            "Dot product menghitung perkalian elemen searah pada kedua vektor. "
            "Hanya kata yang muncul di **kedua kalimat** yang menghasilkan nilai non-nol."
        )

        dot = float(np.dot(a, b))
        bersama = [k for k in range(n) if a[k] > 0 and b[k] > 0]

        if bersama:
            st.markdown(f"**Kata yang sama di kedua kalimat:** `{', '.join(kata[k] for k in bersama)}`")
            tabel_sama = pd.DataFrame(
                {
                    "Kata Sama": [kata[k] for k in bersama],
                    f"Jumlah di K{ni}": [int(vec.counts_[i, k]) for k in bersama],
                    f"Jumlah di K{nj}": [int(vec.counts_[j, k]) for k in bersama],
                    "Bobot A": [a[k] for k in bersama],
                    "Bobot B": [b[k] for k in bersama],
                    "A × B": [a[k] * b[k] for k in bersama],
                }
            )
            st.dataframe(tabel_sama.round(4), hide_index=True, use_container_width=True)

            suku = [f"({f4(a[k])} \\times {f4(b[k])})" for k in bersama]
            st.latex(r"A \cdot B = " + gabung(suku) + f" = {f4(dot)}")
        else:
            st.warning("Tidak ada kata yang sama di antara kedua kalimat, sehingga Dot Product = 0.")
            st.latex(r"A \cdot B = 0")

    with tab_s5:
        st.markdown("#### Langkah 7 & 8: Menghitung Norma Vektor (‖A‖, ‖B‖) dan Hasil Akhir")
        na = float(np.sqrt(np.sum(a ** 2)))
        nb = float(np.sqrt(np.sum(b ** 2)))

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

        st.markdown("---")
        st.markdown("#### Hasil Perhitungan Cosine Similarity:")
        if na == 0 or nb == 0:
            st.warning("Salah satu vektor bernilai 0 (kosong setelah preprocessing). Skor = 0.")
            st.latex(r"\cos(\theta) = 0")
        else:
            st.latex(
                rf"\cos(\theta) = \frac{{{f4(dot)}}}{{{f4(na)} \times {f4(nb)}}} "
                rf"= \frac{{{f4(dot)}}}{{{f4(na * nb)}}} = \mathbf{{{f4(skor)}}}"
            )

        with st.expander("📊 Lihat Tabel Perkalian Per-Kata Lengkap"):
            tabel_full = pd.DataFrame(
                {
                    "Kata": kata,
                    "A": a,
                    "B": b,
                    "A × B": a * b,
                    "A²": a ** 2,
                    "B²": b ** 2,
                }
            )
            st.dataframe(tabel_full.round(4), hide_index=True, use_container_width=True)


def tampilkan(res: dict) -> None:
    vec, M, labels = res["vec"], res["matriks"], res["labels"]
    n_kal = len(labels)
    sim = similarity_matrix(M, labels)
    pasangan = rank_pairs(sim)

    theme_colors = THEMES.get(st.session_state.get("select_theme", "Indigo Modern"), THEMES["Indigo Modern"])
    primary_color = theme_colors["primary"]

    st.divider()
    st.markdown("## 📊 Hasil Analisis & Visualisasi Kemiripan")

    main_tab1, main_tab2, main_tab3 = st.tabs(
        ["📈 Dashboard Hasil", "🔤 Matriks & Peringkat", "📚 Teori & Panduan Math"]
    )

    with main_tab1:
        if n_kal > 2:
            terbaik = pasangan.iloc[0]
            st.success(
                f"🏆 **Pasangan Paling Mirip:** **{terbaik['item_1']}** dan **{terbaik['item_2']}** "
                f"dengan skor **{terbaik['similarity']:.4f}**."
            )
            pilihan = st.selectbox(
                "Pilih pasangan kalimat yang ingin dianalisis secara detail:",
                options=list(range(len(pasangan))),
                format_func=lambda k: (
                    f"{pasangan.at[k, 'item_1']} & {pasangan.at[k, 'item_2']} "
                    f"(Skor: {pasangan.at[k, 'similarity']:.4f})"
                ),
                key=f"select_pasangan_{n_kal}",
            )
            i = labels.index(pasangan.at[pilihan, "item_1"])
            j = labels.index(pasangan.at[pilihan, "item_2"])
        else:
            i, j = 0, 1

        skor = cosine_similarity(M[i], M[j])
        sudut = float(np.degrees(np.arccos(np.clip(skor, -1.0, 1.0))))

        # Summary Metric Cards & Charts
        col_m1, col_m2, col_m3 = st.columns([1, 1, 1])

        with col_m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Cosine Similarity</div>
                    <div class="metric-value">{skor:.4f}</div>
                    <div style="margin-top: 0.4rem;">{get_badge_html(skor)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Sudut Vektor (θ)</div>
                    <div class="metric-value">{sudut:.1f}°</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.4rem;">
                        0° = Identik | 90° = Tegak Lurus
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_m3:
            diff_sk = compare_with_sklearn(M)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Verifikasi Scikit-Learn</div>
                    <div class="metric-value" style="font-size: 1.6rem; color: #10b981;">{diff_sk:.1e}</div>
                    <div style="font-size: 0.85rem; opacity: 0.8; margin-top: 0.4rem;">
                        Selisih Maksimum Matrix
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_g = fig_gauge_chart(skor, primary_color)
            st.plotly_chart(fig_g, use_container_width=True)
        with col_g2:
            fig_a = fig_angle_chart(skor, primary_color)
            st.plotly_chart(fig_a, use_container_width=True)

        # Detail langkah
        jelaskan(res, i, j)

    with main_tab2:
        if n_kal > 2:
            col_hm, col_rk = st.columns([3, 2])
            with col_hm:
                st.markdown("#### Matriks Kemiripan (Heatmap Interaktif)")
                fig_hm = fig_matrix_heatmap(sim, theme_colors["colorscale"])
                st.plotly_chart(fig_hm, use_container_width=True)

                gaya = sim.style.format("{:.4f}")
                gaya = gaya.map(warna) if hasattr(gaya, "map") else gaya.applymap(warna)
                st.dataframe(gaya, use_container_width=True)

            with col_rk:
                st.markdown("#### Peringkat Pasangan Kalimat")
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
                    use_container_width=True,
                    column_config={
                        "Skor": st.column_config.ProgressColumn(
                            "Skor", min_value=0.0, max_value=1.0, format="%.4f"
                        )
                    },
                )
        else:
            st.info("ℹ️ Matriks kemiripan dan peringkat lengkap akan aktif jika jumlah kalimat > 2.")

        st.markdown("#### 📖 Matriks Bobot Vektor Lengkap (Seluruh Kata Kosakata)")
        kata = vec.vocabulary_
        df_vec_full = pd.DataFrame(M, index=labels, columns=kata)
        st.dataframe(df_vec_full.round(4), use_container_width=True)

    with main_tab3:
        st.markdown("### 📚 Penjelasan Teori & Formula Matematis")
        st.markdown(
            """
            #### 1. Apa itu Cosine Similarity?
            Cosine similarity mengukur kemiripan antara dua vektor non-nol dalam ruang bersudut banyak. 
            Teknik ini menghitung nilai **cosinus sudut ($\theta$)** di antara dua vektor tersebut.
            
            - **Skor 1.0 (0°):** Kedua vektor searah sempurna (kalimat sangat mirip / identik secara konteks kata).
            - **Skor 0.0 (90°):** Kedua vektor tegak lurus (tidak ada kata yang sama).
            
            #### 2. Formula Cosine Similarity
            $$\\cos(\\theta) = \\frac{A \\cdot B}{\\lVert A \\rVert \\times \\lVert B \\rVert} = \\frac{\\sum_{i=1}^{n} A_i B_i}{\\sqrt{\\sum_{i=1}^{n} A_i^2} \\times \\sqrt{\\sum_{i=1}^{n} B_i^2}}$$
            
            #### 3. Pembobotan Kata
            - **Bag-of-Words (BoW):** Vektor diisi frekuensi mentah kemunculan kata $f_{t,d}$.
            - **TF-IDF:** Menyeimbangkan frekuensi kata dengan kelangkaan kata di seluruh dokumen.
              - **Term Frequency (TF):** $TF(t,d) = \\frac{f_{t,d}}{\\text{total kata dalam } d}$
              - **Inverse Document Frequency (IDF):** $IDF(t) = \\ln\\left(\\frac{1 + N}{1 + df(t)}\\right) + 1$
              - **TF-IDF:** $TF\\text{-}IDF(t,d) = TF(t,d) \\times IDF(t)$
            """
        )


if "hasil" in st.session_state:
    tampilkan(st.session_state["hasil"])
