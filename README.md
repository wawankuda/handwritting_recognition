# 🖋️ Multi-Task Handwriting Recognition & OCR

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Handwriting-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/landlord/handwriting-recognition)

Repositori ini berisi implementasi *End-to-End* model **Deep Learning Optical Character Recognition (OCR)** untuk mengenali teks tulisan tangan (*Handwritten Text Recognition / HTR*). Proyek ini menggabungkan arsitektur modern berbasis **ConvNeXt-Tiny** dan **Hybrid Vision Transformer** dengan pendekatan **Multi-Task Learning** serta dekoding berbasis **Connectionist Temporal Classification (CTC)**.

---

## 📌 Daftar Isi
- [Ringkasan Proyek](#-ringkasan-proyek)
- [Arsitektur & Pendekatan](#-arsitektur--pendekatan)
- [Multi-Task Learning Strategy](#-multi-task-learning-strategy)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Pipeline Pelatihan & Evaluasi](#-pipeline-pelatihan--evaluasi)
- [Metrik Evaluasi & Hasil](#-metrik-evaluasi--hasil)
- [Struktur Direktori](#-struktur-direktori)
- [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)
- [Kontributor & Lisensi](#-kontributor--lisensi)

---

## 🚀 Ringkasan Proyek

Sistem OCR tulisan tangan konvensional seringkali mengalami kendala saat menghadapi variasi gaya tulisan, goresan tipis/tebal, noise latar belakang, serta adanya sampel gambar kosong (*blank/empty images*).

Proyek ini menghadirkan solusi komprehensif:
1. **Modern Feature Extractor**: Memanfaatkan **ConvNeXt-Tiny** yang mengadaptasi keunggulan Vision Transformer dengan efisiensi konvolusi 2D (kernel 7x7) untuk menangkap kontur tulisan tangan.
2. **Sequential & Attention Modeling**: Menggunakan mekanisme **Positional Encoding** dan lapisan **Transformer / BiLSTM** untuk memodelkan ketergantungan antar-karakter secara temporal/sekuensial.
3. **Multi-Task Learning**: Melatih model dengan dua kepala (*heads*):
   - **Primary Head (Transcription)**: Memprediksi teks tulisan tangan menggunakan `nn.CTCLoss`.
   - **Auxiliary Head (Detection)**: Mengklasifikasi apakah gambar berisi tulisan valid atau berlabel `EMPTY`.
4. **Metrik Komprehensif**: Evaluasi menggunakan **Character Error Rate (CER)** berbasis *Levenshtein Distance*, **Exact Match Accuracy**, serta **F1-Score**.

---

## 🏗️ Arsitektur & Pendekatan

Alur inferensi dan ekstraksi fitur dirancang secara modular:

```text
[Input Citra Tulisan Tangan (1 × H × W)]
                  │
                  ▼
┌───────────────────────────────────────────────┐
│          Feature Extractor Backbone           │
│   (ConvNeXt-Tiny modifikasi resolusi OCR /    │
│      DeiT-Tiny Vision Transformer)            │
└───────────────────────┬───────────────────────┘
                        │
       Feature Map (B × C × H' × W')
                        │  (Collapse Height dimension)
                        ▼
┌───────────────────────────────────────────────┐
│        Positional Encoding & Sequence         │
│          Temporal / Transformer Layer         │
└───────────────┬───────────────────────────────┘
                │
        Sequence Representations (B × Time × D)
                │
        ┌───────┴───────────────────────────────┐
        ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│      Head 1: CTC Decoder      │   │  Head 2: Binary Classifier    │
│     (Text Transcription)      │   │      (EMPTY vs VALID)         │
│       Loss: nn.CTCLoss        │   │        Loss: BCE / CE         │
└───────────────────────────────┘   └───────────────────────────────┘
```

### Mengapa ConvNeXt & Transformer?
- **ConvNeXt-Tiny**: Menggabungkan stabilitas CNN dengan desain modern (large 7x7 kernels, GELU, LayerNorm) yang sangat ideal untuk menangkap goresan huruf yang menyambung (*cursive*).
- **CTC (Connectionist Temporal Classification)**: Mengatasi permasalahan tanpa perlunya anotasi segmentasi per-karakter (*alignment-free training*).

---

## 🎯 Multi-Task Learning Strategy

Seringkali pada dataset dunia nyata terdapat gambar yang rusak atau tidak berisi tulisan tangan sama sekali (ditandai dengan label `EMPTY`). Alih-alih hanya membuangnya, model ini dilatih secara simultan untuk:
- Mengidentifikasi keberadaan konten tulisan valid melalui *Auxiliary Classification Head*.
- Menerjemahkan urutan karakter hanya pada tulisan yang valid melalui *CTC Transcription Head*.

Total Loss diformulasikan sebagai kombinasi berbobot:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}} + \lambda \cdot \mathcal{L}_{\text{Auxiliary}}$$

---

## 📊 Dataset & Preprocessing

Proyek ini diuji menggunakan **Kaggle Handwriting Recognition Dataset** (`landlord/handwriting-recognition`):
- **Data Latih (Train)**: ~330,000 sampel
- **Data Validasi (Validation)**: ~41,000 sampel
- **Data Uji (Test)**: ~41,000 sampel

### Tahapan Pembersihan & Preprocessing:
1. **Data Cleansing**: Menghapus nilai *missing values* (`NaN`) pada kolom identitas label.
2. **Vocabulary Mapping**: Membangun kamus karakter unik dengan alokasi khusus token `[blank]` pada indeks `0` sesuai standar PyTorch CTC Loss.
3. **Image Normalization & Resize**: Penyesuaian aspek rasio tinggi gambar tetap dan penyesuaian lebar secara konsisten untuk pemrosesan sekuensial.
4. **Dynamic Batch Collation**: `collate_fn_multitask` kustom untuk membungkus *tensor images*, *labels*, dan *target lengths* secara efisien.

---

## 🔄 Pipeline Pelatihan & Evaluasi

Proses eksperimen dibagi menjadi beberapa fase terstruktur:

1. **Fase 1 - Eksperimen & Benchmark Arsitektur (Epoch Pendek)**:
   - Pengujian komparatif antara Baseline CRNN, DeiT-Tiny, dan ConvNeXt-Tiny + Transformer.
   - Analisis konvergensi loss dan stabilitas gradient.
2. **Fase 2 - Final Training Model Terbaik**:
   - Pelatihan model **Multi-Task Hybrid Transformer** pada skala sampel besar (50.000+ data) dengan *early monitoring*.
   - Optimasi menggunakan optimizer `AdamW` disertai penyesuaian *learning rate scheduler*.
3. **Fase 3 - Evaluasi Independen & Ekspor Prediksi**:
   - Pengujian akhir pada data uji independen yang bersih.
   - Ekspor hasil inferensi dan visualisasi komparasi *Ground Truth* vs *Prediction*.

---

## 📈 Metrik Evaluasi & Hasil

Model dievaluasi menggunakan dua metrik utama:
- **Character Error Rate (CER)**: Mengukur jarak edit (Levenshtein distance) dibagi panjang string target:
  $$\text{CER} = \frac{S + D + I}{N}$$
  *(S: Substitutions, D: Deletions, I: Insertions, N: Jumlah Karakter)*
- **Exact Match Accuracy**: Persentase prediksi kata yang 100% tepat dan cocok sempurna dengan label asli.
- **Empty Detection F1-Score**: Akurasi deteksi gambar kosong.

---

## 📁 Struktur Direktori

```plaintext
handwritting-ocr/
│
├── handwritting-ocr2 (3).ipynb    # Notebook utama (Preprocessing, Modeling, Training & Evaluasi)
├── README.md                      # Dokumentasi lengkap proyek
├── requirements.txt               # Daftar pustaka & dependensi
└── .gitignore                     # Filter file cache & checkpoints
```

---

## 🛠️ Panduan Instalasi & Penggunaan

### 1. Clone Repository
```bash
git clone https://github.com/<username-anda>/handwriting-ocr.git
cd handwriting-ocr
```

### 2. Buat Virtual Environment & Install Dependensi
```bash
# Menggunakan venv
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependensi
pip install torch torchvision timm pandas numpy matplotlib scikit-learn nltk Pillow
```

### 3. Menjalankan Notebook
Buka Jupyter Notebook atau jalankan di Google Colab / Kaggle:
```bash
jupyter notebook "handwritting-ocr2 (3).ipynb"
```

> **Tips:** Pastikan akselerator GPU (CUDA) aktif untuk mempercepat proses pelatihan dan inferensi.

---

