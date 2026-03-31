from fastapi import FastAPI
import pickle
import pandas as pd

from src.config import MODEL_PATH

app = FastAPI()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {"fraud": int(pred), "probability": float(prob)}