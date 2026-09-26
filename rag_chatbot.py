"""
RAG Chatbot sederhana pakai LangChain + Groq + ChromaDB (lokal)

Script ini adalah versi .py dari materi yang sudah dibedah di sesi LangChain:
- Blok 1: Model, Prompt Template, Output Parser
- Blok 2: LCEL (pipe, RunnableParallel, RunnablePassthrough)
- Blok 3: Data Ingestion (Document, Chunking, Embedding, Vector Store)
- Blok 4: RAG Chain (Context Injection, chain final)

Knowledge document yang digunakan dalam aplikasi ini berisi kumpulan materi 
terkait keamanan siber, antara lain keamanan aplikasi web, malware 
dan indikator kompromi (IoC), phishing dan rekayasa sosial, 
perlindungan data pribadi dan kepatuhan terhadap regulasi
maupun standar keamanan seperti UU PDP, ISO/IEC 27001, dan NIST, ransomware, 
serta pencegahan serangan siber pada lingkungan bisnis. 
Seluruh dokumen tersebut disimpan dalam format PDF pada folder `knowledge_docs/` 
dan digunakan sebagai sumber pengetahuan bagi sistem.

Loader PDF-nya pakai PyMuPDF4LLMLoader dari package langchain-pymupdf4llm,
package resmi terpisah (bukan dari langchain_community yang sudah sunset).

Cara jalanin:
    python rag_chatbot.py

Prasyarat:
    - File .env berisi GROQ_API_KEY di folder yang sama dengan script ini
    - Folder knowledge_docs/ berisi file .pdf yang mau dijadikan sumber
    - Package sudah terinstall (lihat requirements di bagian bawah file)
"""

import base64
import glob
import mimetypes
import os
from collections.abc import Callable

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# 1. KONFIGURASI
# ============================================================

CHAT_MODEL = "openai/gpt-oss-120b"
COLLECTION_NAME = "keamanan_siber_bisnis"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Folder berisi knowledge document. Semua file .pdf di dalamnya akan
# dianggap sebagai satu sumber pengetahuan terpisah.
KNOWLEDGE_DIR = "./knowledge_docs"

# System prompt disimpan terpisah dari kode, supaya bisa diubah/di-review
# tanpa menyentuh logika program, dan supaya jejak revisinya jelas kalau
# dipakai bersama version control (git).
SYSTEM_PROMPT_PATH = "./system_prompt.md"


# ============================================================
# 2. SETUP MODEL
# ============================================================

def buat_model() -> ChatGroq:
    """Siapkan koneksi ke model chat lewat Groq."""
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=0,
        reasoning_effort="low",
    )


def buat_model_vision() -> ChatGroq:
    """Siapkan model vision Groq untuk mendeskripsikan gambar sumber."""
    return ChatGroq(
        model=os.getenv(
            "GROQ_VISION_MODEL",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ),
        temperature=0,
    )


def cari_media(folder: str) -> tuple[list[str], list[str]]:
    """Pisahkan file PDF dan gambar yang didukung dari folder knowledge."""
    path_pdf = []
    path_gambar = []
    ekstensi_gambar = {".jpg", ".jpeg", ".png", ".webp"}

    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if not os.path.isfile(path):
            continue
        ekstensi = os.path.splitext(path)[1].lower()
        if ekstensi == ".pdf":
            path_pdf.append(path)
        elif ekstensi in ekstensi_gambar:
            path_gambar.append(path)

    return path_pdf, path_gambar

# ============================================================
# 3. DATA INGESTION (Load -> Split -> Embed -> Store)
# ============================================================

def muat_dokumen(
    folder: str,
    model: ChatGroq | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[Document]:
    """
    Load semua file .pdf di dalam folder jadi list of Document, pakai
    PyMuPDF4LLMLoader langsung dari library LangChain (langchain-pymupdf4llm).

    Setiap file di-load lewat loader resmi ini, jadi kita tidak perlu
    menulis logika parsing PDF sendiri, cukup panggil .load() untuk
    masing-masing file lalu digabung jadi satu list.
    """
    daftar_dokumen = []
    path_file, path_gambar = cari_media(folder)
    # glob.glob(...) bertugas mencari file yang sudah ditentukan di os.path.join(), 
    # dan hasilnya berupa daftar nama file yang cocok, dalam bentuk list Python.

    if not path_file and not path_gambar:
        raise FileNotFoundError(
            f"Tidak ada file PDF atau gambar yang didukung di folder '{folder}'. "
            "Pastikan folder knowledge_docs/ berisi file sumber."
        )

    for path in path_file:
        if on_progress:
            on_progress(path)
        # mode="single" -> satu file PDF jadi satu Document (bukan per halaman).
        # use_layout=False -> ekstraksi teks polos, tanpa mesin deteksi layout/
        # OCR yang tidak perlu untuk PDF berbasis teks seperti artikel ini
        # (mempercepat proses dan menghindari pesan "Using Tesseract..." di
        # konsol yang bisa bikin peserta bingung).
        loader = PyMuPDF4LLMLoader(file_path=path, mode="single", use_layout=False)
        daftar_dokumen.extend(loader.load())

    for path in path_gambar:
        if on_progress:
            on_progress(path)
        if model is None:
            model = buat_model_vision()

        tipe_mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as file_gambar:
            gambar_base64 = base64.b64encode(file_gambar.read()).decode("ascii")

        respons = model.invoke([
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        "Jelaskan isi gambar secara faktual untuk basis pengetahuan "
                        "keamanan siber bisnis. Catat hanya teks, label, indikator, "
                        "atau langkah yang benar-benar terlihat. Gunakan Bahasa Indonesia."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{tipe_mime};base64,{gambar_base64}"},
                },
            ])
        ])
        isi = respons.content
        if not isinstance(isi, str):
            isi = "\n".join(
                bagian["text"]
                for bagian in isi
                if isinstance(bagian, dict) and isinstance(bagian.get("text"), str)
            )
        if isi.strip():
            daftar_dokumen.append(
                Document(
                    page_content=f"Deskripsi gambar {os.path.basename(path)}:\n{isi}",
                    metadata={"source": path, "media_type": tipe_mime},
                )
            )

    return daftar_dokumen


def bangun_vectorstore(dokumen: list) -> Chroma:
    """
    Ubah kumpulan Document jadi vector store siap dicari.

    Tahapannya persis yang sudah dibedah di Blok 3:
    1. Load   -> sudah dilakukan di muat_dokumen()
    2. Split  -> potong jadi chunk kecil
    3. Embed + Store -> simpan ke ChromaDB
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    potongan = splitter.split_documents(dokumen)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
    )
    # reset_collection() dipakai supaya aman dijalankan berulang kali
    # (script ini bisa dijalankan ulang tanpa bikin data dobel atau error).
    vectorstore.reset_collection()
    vectorstore.add_documents(potongan)

    return vectorstore


# ============================================================
# 4. RAG CHAIN (Context Injection -> Prompt -> Model -> Parser)
# ============================================================

def format_docs(daftar_dokumen: list[Document]) -> str:
    """Gabungkan beberapa chunk hasil retrieval jadi satu teks konteks."""
    return "\n\n".join(dok.page_content for dok in daftar_dokumen)

def muat_system_prompt(path: str) -> str:
    """
    Baca system prompt dari file .md terpisah.

    Dipisah dari kode supaya system prompt bisa direvisi (oleh siapa pun
    yang bertanggung jawab atas kualitas jawaban chatbot) tanpa perlu
    menyentuh atau memahami kode Python-nya sama sekali.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()

def buat_rag_chain(retriever, model: ChatGroq, system_prompt: str):
    """
    Satukan retriever, prompt, model, dan parser jadi satu chain LCEL.

    system_prompt (dari file system_prompt.md) berisi instruksi statis:
    peran asisten dan aturan anti-halusinasi. Placeholder {context} dan
    {question} sengaja TIDAK ditaruh di file itu, karena dua-duanya
    bukan instruksi tetap, melainkan bagian yang diisi ulang setiap kali
    ada pertanyaan baru, jadi tetap disusun di sini sebagai pesan "human".
    """
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Konteks:\n{context}\n\nPertanyaan: {question}"),
    ])

    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return rag_chain


# ============================================================
# 5. PROGRAM UTAMA
# ============================================================

def main():
    # load_dotenv() harus dipanggil sebelum ChatGroq dibuat, supaya
    # GROQ_API_KEY sudah ada di environment variable saat dibutuhkan.
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY tidak ditemukan. Pastikan file .env ada di folder "
            "yang sama dengan script ini dan berisi GROQ_API_KEY=..."
        )

    print("Menyiapkan model...")
    model = buat_model()

    print(f"Memuat dokumen dari folder '{KNOWLEDGE_DIR}'...")
    dokumen = muat_dokumen(KNOWLEDGE_DIR)
    print(f"  -> {len(dokumen)} dokumen berhasil dimuat.")

    print("Membangun vector store dari dokumen...")
    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print(f"Memuat system prompt dari '{SYSTEM_PROMPT_PATH}'...")
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    
    print("Merakit RAG chain...")
    rag_chain = buat_rag_chain(retriever, model, system_prompt)

    print("\nRAG chatbot siap. Ketik pertanyaan, atau 'keluar' untuk berhenti.\n")

    while True:
        pertanyaan = input("Pertanyaan: ").strip()
        if pertanyaan.lower() in {"keluar", "exit", "quit"}:
            print("Sampai jumpa.")
            break
        if not pertanyaan:
            continue

        jawaban = rag_chain.invoke(pertanyaan)
        print(f"Jawaban : {jawaban}\n")


if __name__ == "__main__":
    main()

# ============================================================
# KENAPA ADA "if __name__ == '__main__':" DI SINI?
# ============================================================
#
# Python punya "label" tersembunyi bernama __name__ di setiap file .py.
# Isinya beda tergantung cara file ini dipakai:
#
#   - Kalau file ini dijalankan LANGSUNG (python rag_chatbot.py),
#     maka __name__ otomatis berisi teks "__main__".
#
#   - Kalau file ini "dipinjam" fungsinya oleh file lain, misalnya
#     lewat "from rag_chatbot import muat_dokumen", maka __name__
#     berisi nama file ini sendiri, yaitu "rag_chatbot", BUKAN "__main__".
#
# Baris "if __name__ == '__main__':" ini jadi semacam pertanyaan yang
# ditanyakan Python ke dirinya sendiri: "Apakah saya sedang dijalankan
# langsung, atau cuma dipinjam file lain?"
#
# Kalau jawabannya "dijalankan langsung", baru main() dipanggil, dan
# seluruh program (load dokumen, bikin vectorstore, tanya-jawab) jalan.
#
# Kalau jawabannya "cuma dipinjam", main() TIDAK dipanggil. Ini penting
# supaya kalau suatu saat ada script lain yang cuma mau meminjam satu
# fungsi kecil dari sini (misalnya muat_dokumen() saja), seluruh RAG
# chatbot ini tidak ikut menyala tanpa diminta.
#
# Letaknya WAJIB di paling bawah file, setelah semua fungsi (def ...)
# selesai didefinisikan. Python membaca file dari atas ke bawah, jadi
# kalau baris ini ditaruh sebelum fungsi-fungsi yang dipanggil di dalam
# main() selesai "dikenalkan" ke Python, akan muncul error NameError.
# ============================================================
