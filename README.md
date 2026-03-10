# 💧 Prediksi Kondisi Air Paiton

> Sistem prediksi time-series kualitas air berbasis **ARIMA + Monte Carlo Simulation** dengan dashboard interaktif menggunakan Streamlit. Dikembangkan sebagai bagian dari Kerja Praktik (KP).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](#)

---

## 🚀 Demo Singkat

| Fitur | Deskripsi |
|---|---|
| 📂 Multi-file Excel | Upload & pilih file data langsung dari direktori |
| 📈 Prediksi ARIMA | Auto-tuning parameter (p, d, q) menggunakan AIC terbaik |
| 🎲 Simulasi Monte Carlo | Visualisasi distribusi prediksi dengan scatter simulasi |
| 📊 Evaluasi Model | MAE & MSE otomatis dihitung dan ditampilkan per kolom |
| 📥 Export JSON | Download hasil prediksi beserta metadata & metrik evaluasi |

---

## 🧠 Arsitektur Model

Proyek ini mengeksplorasi **3 pendekatan model** untuk prediksi kualitas air:

```
Models/
├── SAmodel/       → TabTransformer (PyTorch) — Deep Learning untuk data tabular
├── CNNmodel/      → Neural Network Regression (PyTorch) — prediksi Quality & Cost
├── MLmodel/       → Formula-based ML approach
└── Benchmarks/    → Perbandingan performa antar model
```

### Model Utama (Streamlit App)
- **ARIMA** (AutoRegressive Integrated Moving Average) dengan seleksi parameter otomatis via ADF Test + grid search AIC
- **Monte Carlo Simulation** untuk menghasilkan interval/distribusi prediksi yang lebih realistis

---

## 🗂️ Struktur Proyek

```
prediksi_kondisi_air_paiton/
│
├── app.py                  # Main Streamlit dashboard
├── data/                   # Dataset Excel (.xlsx)
├── Models/
│   ├── SAmodel/            # TabTransformer (PyTorch + tab_transformer_pytorch)
│   ├── CNNmodel/           # CNN/NN Regression model
│   ├── MLmodel/            # Formula-based prediction
│   └── Benchmarks/         # Comparison script antar model
├── geo/                    # Data geospasial (opsional)
└── excel_files/            # Output / file tambahan
```

---

## ⚙️ Instalasi & Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/adeardiansa/Prediksi_Air_Paiton.git
cd Prediksi_Air_Paiton
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy plotly statsmodels scikit-learn openpyxl
```

Untuk model deep learning (opsional):
```bash
pip install torch tab-transformer-pytorch matplotlib
```

### 3. Jalankan dashboard
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

---

## 📖 Cara Penggunaan

1. **Pilih file Excel** dari sidebar (semua file `.xlsx` di folder `data/` akan terdeteksi otomatis)
2. **Pilih sheet** yang ingin dianalisis
3. Data historis akan ditampilkan secara otomatis dalam tabel
4. **Tentukan rentang tanggal** untuk prediksi
5. Klik **"Prediksi Rentang Tanggal"**
6. Lihat hasil: tabel prediksi, metrik MAE/MSE, dan grafik interaktif
7. **Download JSON** untuk menyimpan hasil prediksi + metadata

---

## 📊 Contoh Output

**Metrik Evaluasi**
| Kolom | MAE | MSE |
|---|---|---|
| Parameter A | 0.42 | 0.31 |
| Parameter B | 1.15 | 2.03 |

**Grafik** menampilkan:
- 🟢 Data historis
- 🔴 Garis prediksi ARIMA (dashed)
- 🩷 Titik simulasi Monte Carlo
- ⬜ Garis pemisah historis vs prediksi

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Dashboard | `Streamlit`, `Plotly` |
| ARIMA / Statistik | `statsmodels` |
| Preprocessing | `Pandas`, `NumPy`, `scikit-learn` |
| Deep Learning | `PyTorch`, `tab-transformer-pytorch` |
| Data | Excel (`.xlsx`) via `openpyxl` |

---

## 📝 Catatan

- Data yang digunakan merupakan data aktual dari monitoring kualitas air PLTU Paiton
- Model ARIMA melakukan **auto-tuning** parameter (p, d, q) setiap prediksi berdasarkan karakteristik data terkini
- Folder `Models/` berisi eksperimen model alternatif yang dikembangkan selama riset

---

## 👤 Author

**Ade Ardiansa**  
Mahasiswa KP — Sistem Prediksi Kondisi Air Paiton  
[![GitHub](https://img.shields.io/badge/GitHub-adeardiansa-black?logo=github)](https://github.com/adeardiansa)
