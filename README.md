# Machine Learning - VARK Learning Style Classification API

REST API untuk mengklasifikasikan gaya belajar mahasiswa berdasarkan model
VARK (Visual, Auditory, Read/Write, Kinesthetic) menggunakan Machine Learning
Random Forest. Sistem ini dikembangkan sebagai bagian dari penelitian/tesis
dan digunakan oleh aplikasi mobile Flutter secara online / realtime.

## Tujuan
Mengolah hasil kuesioner VARK, mengklasifikasikan gaya belajar dominan,
dan memberikan rekomendasi gaya belajar secara otomatis.

## Metode
Algoritma yang digunakan dan dibandingkan:
- K-Nearest Neighbor (KNN)
- Decision Tree
- Random Forest

Hasil pengujian menunjukkan Random Forest memiliki akurasi tertinggi
sehingga dipilih sebagai model utama.

## Arsitektur
Flutter App → FastAPI Backend → Random Forest Model → Prediksi (JSON)

## Struktur Folder
ml_edutest/
- app.py
- train_vark_model.py
- dataset/vark_dataset.csv
- model/vark_random_forest.pkl
- model/scaler.pkl
- model/label_encoder.pkl
- README.md
- .gitignore

## Menjalankan Aplikasi
Aktifkan virtual environment:
source venv/bin/activate

Install dependency:
pip install fastapi uvicorn scikit-learn joblib numpy

Jalankan API:
uvicorn app:app --reload

API berjalan pada:
http://127.0.0.1:8000

## Pengujian
Swagger UI:
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

## Integrasi Flutter
Aplikasi Flutter mengakses endpoint POST /predict dan menerima hasil
klasifikasi gaya belajar secara realtime.

## Catatan Akademik
Model Random Forest dijalankan di backend karena tidak kompatibel dengan
TensorFlow Lite. Pendekatan ini sesuai untuk aplikasi online dengan
performa dan akurasi yang stabil.

## Penulis
Hefri Juanto  
Penelitian / Tesis Klasifikasi Gaya Belajar Mahasiswa

## Referensi
Fleming, N. D., & Mills, C. (1992). Not Another Inventory, Rather a Catalyst for Reflection.
FastAPI Documentation: https://fastapi.tiangolo.com
Scikit-Learn Documentation: https://scikit-learn.org
