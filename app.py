"""
VARK Learning Style Classification API
FastAPI Backend - FIXED VERSION

Endpoints:
- POST /predict: Predict learning style
- POST /register: Register new user
- POST /login: User authentication  
- POST /otp/request: Request OTP
- POST /otp/verify: Verify OTP
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import joblib
import numpy as np
import uvicorn
from datetime import datetime, timedelta
import os
import random
import string
from passlib.context import CryptContext
import jwt

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI
app = FastAPI(
    title="VARK Learning Style API",
    description="REST API untuk klasifikasi gaya belajar menggunakan Random Forest",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ganti dengan domain Flutter app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# In-memory storage (untuk development - ganti dengan database di production)
users_db = {}
otp_storage = {}

# Load ML Model
model_loaded = False
try:
    model_path = 'model/vark_random_forest.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load('model/scaler.pkl')
        label_encoder = joblib.load('model/label_encoder.pkl')
        feature_columns = joblib.load('model/feature_columns.pkl')
        print(f"✅ Model loaded successfully from {model_path}")
        print(f"✅ Expected features: {feature_columns}")
        model_loaded = True
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("⚠️  Please run train_vark_model.py first to generate model files")
        print("⚠️  Prediction endpoint will return 503 Service Unavailable")
        model = None
        scaler = None
        label_encoder = None
        feature_columns = None
except FileNotFoundError as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  Please run train_vark_model.py first to generate model files")
    print("⚠️  Prediction endpoint will return 503 Service Unavailable")
    model = None
    scaler = None
    label_encoder = None
    feature_columns = None
except Exception as e:
    print(f"❌ Unexpected error loading model: {e}")
    print("⚠️  Prediction endpoint will return 503 Service Unavailable")
    model = None
    scaler = None
    label_encoder = None
    feature_columns = None

# ============================================================================
# Pydantic Models
# ============================================================================

class VARKInput(BaseModel):
    """Input model untuk prediksi VARK"""
    visual: float = Field(..., ge=0, le=16, description="Visual score (0-16)")
    auditory: float = Field(..., ge=0, le=16, description="Auditory score (0-16)")
    readwrite: float = Field(..., ge=0, le=16, description="Reading/Writing score (0-16)")
    kinesthetic: float = Field(..., ge=0, le=16, description="Kinesthetic score (0-16)")
    
    @validator('visual', 'auditory', 'readwrite', 'kinesthetic')
    def validate_scores(cls, v):
        if not 0 <= v <= 16:
            raise ValueError('Score must be between 0 and 16')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "visual": 3.5,
                "auditory": 2.0,
                "readwrite": 4.0,
                "kinesthetic": 1.5
            }
        }

class VARKPrediction(BaseModel):
    """Output model untuk prediksi VARK"""
    predicted_style: str = Field(..., description="Predicted learning style (V/A/R/K)")
    style_name: str = Field(..., description="Full name of learning style")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Probability for each style")
    recommendations: List[str] = Field(..., description="Study recommendations")
    timestamp: str = Field(..., description="Prediction timestamp")

class UserRegister(BaseModel):
    email: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: str
    password: str

class OTPRequest(BaseModel):
    email: str

class OTPVerify(BaseModel):
    email: str
    otp_code: str

class Token(BaseModel):
    access_token: str
    token_type: str

# ============================================================================
# Utility Functions
# ============================================================================

def get_style_name(style_code: str) -> str:
    """Convert style code to full name"""
    style_names = {
        'V': 'Visual Learner',
        'A': 'Auditory Learner',
        'R': 'Reading/Writing Learner',
        'K': 'Kinesthetic Learner'
    }
    return style_names.get(style_code, 'Unknown')

def get_recommendations(style_code: str) -> List[str]:
    """Get study recommendations based on learning style"""
    recommendations = {
        'V': [
            "Gunakan diagram, chart, dan mind map saat belajar",
            "Highlight teks dengan warna berbeda untuk kategorisasi",
            "Tonton video pembelajaran dan tutorial visual",
            "Buat flashcard dengan gambar dan ilustrasi",
            "Gunakan infografis untuk merangkum informasi"
        ],
        'A': [
            "Dengarkan podcast atau audiobook tentang materi",
            "Belajar sambil menjelaskan konsep dengan suara keras",
            "Ikuti diskusi kelompok atau study group",
            "Rekam penjelasan dan dengarkan kembali",
            "Gunakan mnemonic dan rhymes untuk mengingat"
        ],
        'R': [
            "Buat catatan tertulis yang detail dan terstruktur",
            "Baca buku teks, artikel, dan jurnal",
            "Tulis summary dan rangkuman setelah belajar",
            "Buat outline dan bullet points",
            "Reorganisasi informasi dalam format tertulis"
        ],
        'K': [
            "Praktikkan langsung apa yang dipelajari",
            "Gunakan simulasi dan hands-on activities",
            "Belajar sambil bergerak atau berjalan",
            "Buat model atau demonstrasi fisik",
            "Ikuti lab praktikum dan workshop"
        ]
    }
    return recommendations.get(style_code, ["Gunakan kombinasi berbagai metode belajar"])

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_otp() -> str:
    """Generate 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VARK Learning Style Classification API",
        "version": "2.0.0",
        "status": "running" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        },
        "instructions": "Run train_vark_model.py first if model is not loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_path": "model/vark_random_forest.pkl",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=VARKPrediction)
async def predict_learning_style(data: VARKInput):
    """
    Predict learning style based on VARK scores
    
    Args:
        data: VARKInput with visual, auditory, readwrite, kinesthetic scores
    
    Returns:
        VARKPrediction with predicted style and recommendations
    """
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model not loaded",
                "message": "Please run train_vark_model.py first to generate model files",
                "solution": "Execute: python train_vark_model.py"
            }
        )
    
    try:
        # Prepare input array - HARUS SESUAI URUTAN SAAT TRAINING
        input_array = np.array([[
            data.visual,
            data.auditory,
            data.readwrite,
            data.kinesthetic
        ]])
        
        # Validate input shape
        if input_array.shape[1] != len(feature_columns):
            raise ValueError(
                f"Input has {input_array.shape[1]} features, "
                f"but model expects {len(feature_columns)} features"
            )
        
        # Scale input
        input_scaled = scaler.transform(input_array)
        
        # Predict
        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Decode prediction
        predicted_style = label_encoder.inverse_transform(prediction)[0]
        
        # Get confidence (max probability)
        confidence = float(np.max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            label: float(prob) 
            for label, prob in zip(label_encoder.classes_, probabilities)
        }
        
        # Get recommendations
        recommendations = get_recommendations(predicted_style)
        
        return VARKPrediction(
            predicted_style=predicted_style,
            style_name=get_style_name(predicted_style),
            confidence=confidence,
            probabilities=prob_dict,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Prediction error",
                "message": str(e),
                "type": type(e).__name__
            }
        )

# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    """Register new user"""
    if user.email in users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = pwd_context.hash(user.password)
    users_db[user.email] = {
        "email": user.email,
        "password": hashed_password,
        "full_name": user.full_name,
        "created_at": datetime.now().isoformat()
    }
    
    return {"message": "User registered successfully", "email": user.email}

@app.post("/login", response_model=Token)
async def login(user: UserLogin):
    """User login"""
    if user.email not in users_db:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    stored_user = users_db[user.email]
    if not pwd_context.verify(user.password, stored_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/otp/request")
async def request_otp(data: OTPRequest):
    """Request OTP for email verification"""
    otp_code = generate_otp()
    
    # Store OTP with expiration (5 minutes)
    otp_storage[data.email] = {
        "code": otp_code,
        "expires_at": datetime.now() + timedelta(minutes=5)
    }
    
    # TODO: Send OTP via email (implement email service)
    print(f"OTP for {data.email}: {otp_code}")  # For development
    
    return {
        "message": "OTP sent to email",
        "email": data.email,
        "otp_for_dev": otp_code  # Remove in production
    }

@app.post("/otp/verify")
async def verify_otp(data: OTPVerify):
    """Verify OTP"""
    if data.email not in otp_storage:
        raise HTTPException(status_code=400, detail="No OTP found for this email")
    
    stored_otp = otp_storage[data.email]
    
    if datetime.now() > stored_otp["expires_at"]:
        del otp_storage[data.email]
        raise HTTPException(status_code=400, detail="OTP expired")
    
    if data.otp_code != stored_otp["code"]:
        raise HTTPException(status_code=400, detail="Invalid OTP")
    
    # OTP verified, remove from storage
    del otp_storage[data.email]
    
    return {"message": "OTP verified successfully"}

# ============================================================================
# Protected Endpoints (require authentication)
# ============================================================================

@app.get("/user/profile")
async def get_profile(email: str = Depends(verify_token)):
    """Get user profile (protected)"""
    if email not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = users_db[email]
    return {
        "email": user["email"],
        "full_name": user["full_name"],
        "created_at": user["created_at"]
    }

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Starting VARK Learning Style API Server")
    print("="*60)
    print(f"Model loaded: {model_loaded}")
    if feature_columns:
        print(f"Expected features: {feature_columns}")
    if not model_loaded:
        print("="*60)
        print("⚠️  WARNING: Model not loaded!")
        print("⚠️  Prediction endpoint will return 503 error")
        print("⚠️  Run 'python train_vark_model.py' first")
        print("="*60)
    print("="*60)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
