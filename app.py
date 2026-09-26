import os

import streamlit as st
from dotenv import load_dotenv

from rag_chatbot import (
    KNOWLEDGE_DIR,
    SYSTEM_PROMPT_PATH,
    TOP_K,
    buat_model,
    buat_model_vision,
    muat_dokumen,
    cari_media,
    bangun_vectorstore,
    muat_system_prompt,
    buat_rag_chain,
)


# ============================================================
# 1. PENGATURAN HALAMAN
# ============================================================
# Wajib jadi perintah Streamlit pertama: judul tab browser dan ikonnya.

st.set_page_config(
    page_title="Asisten Keamanan Cerdas",
    page_icon=":material/security:",
)

# ============================================================
# 2. CEK API KEY
# ============================================================
# Di laptop, GROQ_API_KEY dibaca dari file .env.
# Di Streamlit Cloud, GROQ_API_KEY diisi lewat menu Secrets, dan Streamlit
# otomatis menjadikannya environment variable. Jadi kode yang sama ini
# jalan di dua tempat tanpa perlu diubah.

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY belum diisi. Cek file .env (di laptop) "
        "atau menu Secrets (di Streamlit Cloud)."
    )
    st.stop()

    # ============================================================
# 3. SIAPKAN MESIN CHATBOT (sekali saja, lalu disimpan)
# ============================================================
# Streamlit menjalankan ulang SELURUH file ini dari atas setiap kali
# pengguna berinteraksi (misalnya mengirim pertanyaan).
# @st.cache_resource membuat fungsi di bawah ini cukup dijalankan SEKALI.
# Hasilnya disimpan, lalu dipakai ulang, sehingga dokumen tidak dimuat
# ulang dan vector store tidak dibangun ulang di setiap pertanyaan.

@st.cache_resource(show_spinner="Menyiapkan chatbot, mohon tunggu sebentar...")
def siapkan_chatbot():
    model = buat_model()

    # File gambar (png/jpg/webp) tidak bisa dibaca seperti teks, jadi isinya
    # lebih dulu "dibuat" menjadi deskripsi teks oleh model vision. Model itu
    # hanya dibuat kalau folder knowledge_docs/ memang berisi gambar.
    _, path_gambar = cari_media(KNOWLEDGE_DIR)
    if path_gambar:
        st.caption(f"Membaca {len(path_gambar)} file gambar di knowledge_docs/ ...")
        model_vision = buat_model_vision()
        dokumen = muat_dokumen(
            KNOWLEDGE_DIR,
            model=model_vision,
            on_progress=lambda path: st.caption(f"Sedang membaca: {os.path.basename(path)}"),
        )
    else:
        dokumen = muat_dokumen(KNOWLEDGE_DIR)

    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    return buat_rag_chain(retriever, model, system_prompt)


rag_chain = siapkan_chatbot()


# ============================================================
# 4. BUKU CATATAN PERCAKAPAN
# ============================================================
# st.session_state adalah tempat menyimpan data yang tidak ikut hilang
# saat file ini dijalankan ulang. Di sini dipakai untuk mencatat riwayat
# percakapan: siapa yang bicara ("user" atau "assistant") dan isinya.
# Sama saja dengan menjaga percakapan terus muncul di atas chat baru

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

# ============================================================
# 5. TAMPILAN (Versi AI Keamanan Siber & Malware Bisnis)
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --surface-main: #0E1512;
        --surface-panel: #18231E;
        --surface-sidebar: #121C17;
        --accent: #4FD1B0;
        --accent-soft: rgba(79, 209, 176, 0.06);
        --text-main: #EAF2ED;
        --text-muted: #A8B9AF;
        --line-soft: rgba(147, 183, 164, 0.16);
        --radius-ui: 8px;
    }
    html, body, [class*="css"] {
        font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--surface-main);
        background-image:
            linear-gradient(135deg, rgba(31, 70, 54, 0.2), transparent 42%),
            linear-gradient(rgba(111, 163, 139, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(111, 163, 139, 0.035) 1px, transparent 1px);
        background-size: 32px 32px;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    .block-container {
        max-width: 920px;
        padding-top: 2.2rem;
        padding-bottom: 6rem;
    }
    [data-testid="stSidebar"] {
        background: var(--surface-sidebar);
        border-right: 1px solid rgba(147, 183, 164, 0.14);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    .sidebar-label, .app-eyebrow {
        color: #85A697;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .sidebar-label {
        margin-bottom: 0.75rem;
    }
    .sidebar-status {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 1.25rem 0 1.5rem;
        padding: 0.7rem 0.8rem;
        border: 1px solid rgba(79, 209, 176, 0.2);
        border-radius: 6px;
        background: var(--accent-soft);
        color: #C2D5CA;
        font-size: 0.82rem;
    }
    .sidebar-section {
        margin-top: 1.7rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(147, 183, 164, 0.14);
        color: #85A697;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .app-eyebrow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.55rem;
        margin-bottom: 0.7rem;
    }
    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.65rem;
    }
    .brand-mark {
        display: grid;
        flex: 0 0 64px;
        width: 64px;
        height: 64px;
        place-items: center;
        border: 1px solid rgba(79, 209, 176, 0.35);
        border-radius: 12px;
        background: linear-gradient(145deg, rgba(79, 209, 176, 0.16), rgba(24, 35, 30, 0.92));
        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.2);
    }
    .brand-mark svg {
        display: block;
        width: 42px;
        height: 42px;
    }
    .brand-copy {
        min-width: 0;
    }
    .brand-copy .app-eyebrow {
        margin-bottom: 0.4rem;
    }
    .status-light {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4FD1B0;
        box-shadow: 0 0 12px rgba(79, 209, 176, 0.65);
    }
    .app-title {
        margin: 0;
        color: #EAF2ED;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .app-title span {
        color: #4FD1B0;
    }
    .app-subtitle {
        max-width: 650px;
        margin: 0.7rem 0 1.35rem;
        color: #A8B9AF;
        font-size: 1rem;
        line-height: 1.6;
    }
    .starter-label {
        margin: 1rem 0 0.65rem;
        color: #85A697;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    [data-testid="stChatMessage"] {
        margin-bottom: 0.75rem;
        padding: 1rem 1.1rem;
        border: 1px solid var(--line-soft);
        border-radius: var(--radius-ui);
        background: rgba(24, 35, 30, 0.92);
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.65;
    }
    [data-testid="stChatInput"] {
        width: 100%;
        border: 1px solid rgba(147, 183, 164, 0.28);
        border-radius: var(--radius-ui);
        background: var(--surface-panel);
    }
    [data-testid="stChatInput"] textarea {
        padding-left: 33px;
    }
    [data-testid="stBottomBlockContainer"] {
        width: 100%;
        max-width: 920px;
        margin-right: auto;
        margin-left: auto;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }
    .stButton > button {
        min-height: 2.8rem;
        border-radius: 6px;
        border-color: rgba(147, 183, 164, 0.3);
        background: rgba(24, 35, 30, 0.82);
        transition: border-color 150ms ease, background 150ms ease;
    }
    .stButton > button:hover {
        border-color: #4FD1B0;
        color: #4FD1B0;
    }
    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 5rem;
        }
        .app-title {
            font-size: 1.65rem;
        }
        .brand-lockup {
            gap: 0.75rem;
        }
        .brand-mark {
            flex-basis: 52px;
            width: 52px;
            height: 52px;
            border-radius: 10px;
        }
        .brand-mark svg {
            width: 35px;
            height: 35px;
        }
        .app-subtitle {
            font-size: 0.94rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-label">CYBERSECURITY / BUSINESS</div>', unsafe_allow_html=True)
    st.header("Ruang Keamanan")
    st.write(
        "Pusat informasi untuk membantu bisnis memahami dan mengurangi risiko digital."
    )
    st.markdown(
        '<div class="sidebar-status"><span class="status-light"></span> ASISTEN SIAP</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section">Cakupan informasi</div>', unsafe_allow_html=True)
    st.caption("Malware, phishing, dan kepatuhan keamanan siber bisnis.")
    st.markdown('<div class="sidebar-section">Sumber pengetahuan</div>', unsafe_allow_html=True)
    st.caption("Jawaban faktual dirujuk dari materi keamanan yang tersedia.")
    if st.button("Mulai percakapan baru", icon=":material/add:"):
        st.session_state.riwayat = []

st.markdown(
    """
    <div class="brand-lockup">
        <div class="brand-mark" role="img" aria-label="Logo perisai dan kunci Asisten Keamanan Cerdas">
            <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <path d="M32 5L53 13V28C53 41.5 44.4 52.2 32 59C19.6 52.2 11 41.5 11 28V13L32 5Z" fill="#14251F" stroke="#4FD1B0" stroke-width="2.5"/>
                <path d="M24 29V24.5C24 20.1 27.6 16.5 32 16.5C36.4 16.5 40 20.1 40 24.5V29" stroke="#A9EAD7" stroke-width="3" stroke-linecap="round"/>
                <rect x="21" y="28" width="22" height="19" rx="4" fill="#4FD1B0"/>
                <circle cx="32" cy="36" r="2.5" fill="#102019"/>
                <path d="M32 38.5V41" stroke="#102019" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
        <div class="brand-copy">
            <div class="app-eyebrow">PUSAT INFORMASI KEAMANAN</div>
            <h1 class="app-title">Asisten Keamanan <span>Cerdas</span></h1>
        </div>
    </div>
    <p class="app-subtitle">Tanyakan hal seputar pencegahan malware, phishing, dan kepatuhan keamanan siber untuk bisnis.</p>
    """,
    unsafe_allow_html=True,
)

# Salam pembuka dan topik pilihan hanya tampil saat percakapan masih kosong.
pertanyaan_pilihan = None
if not st.session_state.riwayat:
    with st.chat_message("assistant"):
        st.markdown(
            "Halo! Saya Asisten Keamanan Cerdas. Tanyakan topik seperti "
            "*langkah mencegah ransomware* atau *cara mengenali phishing*."
        )
    st.markdown('<div class="starter-label">Mulai dari topik populer</div>', unsafe_allow_html=True)
    kolom_topik = st.columns(3)
    topik_awal = (
        ("Phishing", "Bagaimana cara mengenali dan mencegah serangan phishing?"),
        ("Ransomware", "Apa langkah pencegahan ransomware untuk bisnis?"),
        (
            "Kepatuhan",
            "Apa kewajiban perusahaan sebagai Pengendali Data Pribadi menurut UU PDP, "
            "termasuk pemberitahuan jika terjadi kebocoran data?",
        ),
    )
    for kolom, (label, teks_pertanyaan) in zip(kolom_topik, topik_awal):
        with kolom:
            if st.button(label, use_container_width=True, key=f"topik_{label.lower()}"):
                pertanyaan_pilihan = teks_pertanyaan

# Tampilkan ulang seluruh riwayat percakapan dari buku catatan.
for pesan in st.session_state.riwayat:
    with st.chat_message(pesan["role"]):
        st.markdown(pesan["isi"])


# ============================================================
# 6. TANYA JAWAB
# ============================================================

pertanyaan = st.chat_input("Tulis pertanyaan Anda di sini...") or pertanyaan_pilihan

if pertanyaan:
    # Tampilkan pertanyaan, lalu catat ke buku catatan.
    with st.chat_message("user"):
        st.markdown(pertanyaan)
    st.session_state.riwayat.append({"role": "user", "isi": pertanyaan})

    # Minta jawaban ke mesin RAG. .stream() + st.write_stream() membuat
    # jawaban muncul bertahap, kata demi kata, seperti sedang diketik.
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban di dokumen..."):
            jawaban = st.write_stream(rag_chain.stream(pertanyaan))
    st.session_state.riwayat.append({"role": "assistant", "isi": jawaban})

    


