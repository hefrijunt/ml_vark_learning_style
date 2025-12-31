from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model & tools
model = joblib.load("model/vark_random_forest.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

app = FastAPI(title="VARK Learning Style API")

class VarkInput(BaseModel):
    visual: float
    auditory: float
    readwrite: float
    kinesthetic: float

@app.post("/predict")
def predict_style(data: VarkInput):
    X = np.array([[ 
        data.visual,
        data.auditory,
        data.readwrite,
        data.kinesthetic
    ]])

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    label = label_encoder.inverse_transform([pred])[0]

    return {
        "predicted_style": label
    }
