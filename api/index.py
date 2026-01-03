from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()   # 🔑 REQUIRED ENTRYPOINT

class InputData(BaseModel):
    features: list[float]

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "../model/model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "../model/scaler.pkl"))

@app.get("/")
def root():
    return {"status": "API IS REACHABLE"}

@app.post("/predict")
def predict():
    return {"status": "PREDICT ROUTE HIT"}
async def predict(data: InputData):
    X = scaler.transform([data.features])
    pred = model.predict(X)[0]
    prob = max(model.predict_proba(X)[0])
    return {
        "prediction": int(pred),
        "confidence": float(prob)
    }
