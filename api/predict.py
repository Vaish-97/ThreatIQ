from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Load model and scaler ONCE at startup
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = int(model.predict(features_scaled)[0])
    confidence = float(max(model.predict_proba(features_scaled)[0]))

    return {
        "prediction": prediction,
        "confidence": confidence
    }
