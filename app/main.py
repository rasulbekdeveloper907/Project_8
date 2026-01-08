import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# --------------------------------------------------
# Logging konfiguratsiyasi
# --------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model path (cross-platform)
# --------------------------------------------------
MODEL_PATH = Path("Models") / "Pipeline_Models" / "RandomForestRegressor_Fast.joblib"

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Cars Kilometer Prediction API",
    version="1.0"
)

pipeline = None

# --------------------------------------------------
# Load model on startup
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Failed to load model:", e)
        pipeline = None

# --------------------------------------------------
# Schemas
# --------------------------------------------------
from typing import Optional

class DatasetInput(BaseModel):
    index: int
    dateCrawled: str
    name: str
    seller: str
    offerType: str
    price: int
    abtest: str
    vehicleType: str
    yearOfRegistration: int
    gearbox: str
    powerPS: int
    model: str
    kilometer: int
    monthOfRegistration: int
    fuelType: str
    brand: str
    notRepairedDamage: str
    dateCreated: str
    nrOfPictures: int
    postalCode: int
    lastSeen: str

class PredictionOutput(BaseModel):
    predicted_cluster: int
    cluster_probability: float

# --------------------------------------------------
# Health endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# Predict endpoint
# --------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: DatasetInput):

    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Input → DataFrame (cross Pydantic version)
    df = pd.DataFrame([data.dict()])

    # Predict probabilities (multiclass classifier)
    try:
        proba_all = pipeline.predict_proba(df)[0]
    except AttributeError:
        raise HTTPException(status_code=500, detail="Model is not a classifier and does not support predict_proba")

    logger.debug("DEBUG predict_proba: %s", proba_all)

    # Eng yuqori ehtimollikdagi cluster
    predicted_cluster = int(proba_all.argmax())
    cluster_probability = float(proba_all[predicted_cluster])

    return PredictionOutput(
        predicted_cluster=predicted_cluster,
        cluster_probability=round(cluster_probability, 4)
    )
