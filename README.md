# Machine Learning - VARK Learning Style Classification API Version 2.0.0

REST API untuk mengklasifikasikan gaya belajar mahasiswa berdasarkan model
VARK (Visual, Auditory, Read/Write, Kinesthetic) menggunakan Machine Learning
Random Forest. Sistem ini dikembangkan sebagai bagian dari penelitian/tesis
dan digunakan oleh aplikasi mobile Flutter secara online / realtime.

## Fitur Utama (Versi 2.0)
- **Klasifikasi Gaya Belajar**: Prediksi gaya belajar dominan (V/A/R/K) dengan confidence score
- **Rekomendasi Pembelajaran**: Saran metode belajar berdasarkan gaya belajar
- **Sistem Autentikasi**: Register, Login, dan OTP verification
- **JWT Authentication**: Proteksi endpoint dengan JSON Web Token
- **Probabilitas Prediksi**: Menampilkan probabilitas untuk setiap gaya belajar

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

Model dilatih dengan:
- Preprocessing 16 pertanyaan kuesioner menjadi 4 skor VARK
- Cross-validation 5-fold untuk validasi model
- Feature scaling menggunakan StandardScaler
- Label encoding untuk konversi label kategori

## Arsitektur
Flutter App → FastAPI Backend → Random Forest Model → Prediksi (JSON)

## Struktur Folder
ml_edutest/
- app.py (API utama dengan autentikasi)
- train_vark_model.py (script training model)
- dataset/vark_dataset.csv (dataset kuesioner VARK)
- model/vark_random_forest.pkl (model Random Forest)
- model/scaler.pkl (StandardScaler untuk feature scaling)
- model/label_encoder.pkl (LabelEncoder untuk label)
- model/feature_columns.pkl (daftar fitur yang digunakan)
- README.md
- .gitignore

## Endpoint API

### Prediksi (Public)
- `POST /predict` - Prediksi gaya belajar VARK
- `GET /health` - Health check endpoint

### Autentikasi (Public)
- `POST /register` - Registrasi pengguna baru
- `POST /login` - Login pengguna
- `POST /otp/request` - Request OTP verification
- `POST /otp/verify` - Verify OTP code

### Proteksi (Requires JWT Token)
- `GET /user/profile` - Get user profile

## Install Dependency
Aktifkan virtual environment:
```bash
source venv/bin/activate
```

Install dependency:
```bash
pip install fastapi uvicorn scikit-learn joblib numpy passlib[bcrypt] python-jose pyjwt
```

## Menjalankan Aplikasi
```bash
uvicorn app:app --reload
```

API berjalan pada:
http://127.0.0.1:8000

## Dokumentasi API
Swagger UI:
http://127.0.0.1:8000/docs

## Contoh Input/Output

### Request Prediksi:
```json
{
  "visual": 3.5,
  "auditory": 2.0,
  "readwrite": 4.0,
  "kinesthetic": 1.5
}
```

### Response Prediksi:
```json
{
  "predicted_style": "R",
  "style_name": "Reading/Writing Learner",
  "confidence": 0.85,
  "probabilities": {
    "A": 0.05,
    "K": 0.05,
    "R": 0.85,
    "V": 0.05
  },
  "recommendations": [
    "Buat catatan tertulis yang detail dan terstruktur",
    "Baca buku teks, artikel, dan jurnal",
    "Tulis summary dan rangkuman setelah belajar",
    "Buat outline dan bullet points",
    "Reorganisasi informasi dalam format tertulis"
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Request Registrasi:
```json
{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "John Doe"
}
```

### Request Login:
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### Response Login (JWT Token):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

## Integrasi Flutter
Aplikasi Flutter dapat mengakses API dengan:
1. Register/Login untuk mendapatkan JWT token
2. Include header `Authorization: Bearer <token>` untuk protected endpoints
3. Kirim POST request ke `/predict` dengan skor VARK
4. Terima hasil prediksi lengkap dengan rekomendasi

## Catatan Akademik
Model Random Forest dijalankan di backend karena tidak kompatibel dengan
TensorFlow Lite. Pendekatan ini sesuai untuk aplikasi online dengan
performa dan akurasi yang stabil.

## Catatan Keamanan
- Password di-hash menggunakan bcrypt
- JWT token dengan expiration 30 menit
- OTP berlaku selama 5 menit
- Ganti SECRET_KEY di production environment

## Penulis
Hefri Juanto  
Penelitian / Tesis Klasifikasi Gaya Belajar Mahasiswa

## Referensi
Fleming, N. D., & Mills, C. (1992). Not Another Inventory, Rather a Catalyst for Reflection.
FastAPI Documentation: https://fastapi.tiangolo.com
Scikit-Learn Documentation: https://scikit-learn.org
