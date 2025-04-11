from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Disease Prediction API",
    description="Predicts disease based on symptoms using BioBERT",
    version="1.0"
)

# Load Model & Label Encoder
model_path = "biobert_disease_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
label_encoder = joblib.load("label_encoder.joblib")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Input Schema
class SymptomRequest(BaseModel):
    symptoms_text: str

@app.get("/")
def home():
    return {"message": "Welcome to Disease Prediction API 🚑"}

@app.post("/predict")
def predict_disease(input_data: SymptomRequest):
    symptoms_text = input_data.symptoms_text
    
    # Preprocess and tokenize
    inputs = tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = float(probabilities[predicted_idx]) * 100
    
    predicted_disease = label_encoder.inverse_transform([predicted_idx])[0]
    
    return {
        "predicted_disease": predicted_disease,
        "confidence": f"{confidence:.2f} %"
    }
