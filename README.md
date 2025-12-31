cat << 'EOF' > README.md
# VARK Learning Style Classification API

Backend API untuk mengklasifikasikan gaya belajar mahasiswa berdasarkan model
VARK (Visual, Auditory, Read/Write, Kinesthetic) menggunakan algoritma
Machine Learning Random Forest.

Sistem ini dikembangkan sebagai bagian dari penelitian/tesis dan digunakan
oleh aplikasi mobile (Flutter) secara online / realtime melalui REST API.

---

## Tujuan Sistem
- Mengolah hasil kuesioner VARK mahasiswa
- Mengklasifikasikan gaya belajar dominan
- Memberikan rekomendasi gaya belajar secara otomatis

---

## Metode Machine Learning
Beberapa algoritma Machine Learning yang digunakan dan dibandingkan:
- K-Nearest Neighbor (KNN)
- Decision Tree
- Random Forest

Berdasarkan hasil pengujian, Random Forest menghasilkan akurasi tertinggi
sehingga dipilih sebagai model utama.

---

## Arsitektur Sistem
Aplikasi Flutter mengirimkan data kuesioner ke backend FastAPI.
Backend memproses data menggunakan model Random Forest dan mengembalikan
hasil prediksi gaya belajar dalam format JSON.

Flutter App
→ FastAPI Backend
→ Random Forest Model
→ Prediksi Gaya Belajar

---

## Struktur Folder
ml_edutest/
├── app.py
├── train_vark_model.py
├── dataset/
│   └── vark_dataset.csv
├── model/
│   ├── vark_random_forest.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── README.md
└── .gitignore

---

## Menjalankan Aplikasi

Aktifkan virtual environment:
source venv/bin/activate

Install dependency:
pip install fastapi uvicorn scikit-learn joblib numpy

Jalankan API:
uvicorn app:app --reload

API akan berjalan pada:
http://127.0.0.1:8000

---

## Pengujian API
Gunakan Swagger UI:
http://127.0.0.1:8000/docs

Contoh input:
{
  "visual": 12,
  "auditory": 8,
  "readwrite": 10,
  "kinesthetic": 15
}

Contoh output:
{
  "predicted_style": "K"
}

---

## Integrasi Aplikasi
Aplikasi Flutter mengakses endpoint /predict menggunakan metode POST
dan menerima hasil klasifikasi gaya belajar secara realtime.

---

## Catatan Akademik
Model Random Forest diimplementasikan pada backend service karena
tidak kompatibel dengan TensorFlow Lite.
Pendekatan ini sesuai untuk aplikasi online dan mendukung performa
yang lebih stabil.

---

## Penulis
Hefri Juanto
Penelitian/Tesis Klasifikasi Gaya Belajar Mahasiswa

---

## Referensi
Fleming, N. D., & Mills, C. (1992). Not Another Inventory, Rather a Catalyst for Reflection. To Improve the Academy, 11(1), 137-155.
Dokumentasi FastAPI: fastapi.tiangolo.com
Scikit-Learn Documentation: scikit-learn.org
