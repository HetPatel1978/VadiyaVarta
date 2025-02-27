import torch
import joblib
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ✅ Load Disease Prediction Model
model_path = r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/Temp/trained_disease_model"  # Ensure the correct model path
disease_model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
label_encoder = joblib.load(f"{model_path}/label_encoder.joblib")

def predict_disease(symptoms_list):
    """
    Predicts the disease based on extracted symptoms.
    Returns top 3 possible diseases with confidence scores.
    """
    symptoms_text = " ".join(symptoms_list)
    
    # ✅ DEBUG: Print tokenized input for verification
    encoding = tokenizer(symptoms_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    print("\n🔎 Tokenized Input:", encoding)

    # Model Prediction
    disease_model.eval()
    with torch.no_grad():
        output = disease_model(**encoding)
        logits = output.logits

        # Get softmax probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        top_3_indices = torch.topk(probabilities, 3).indices.tolist()
        top_3_confidences = torch.topk(probabilities, 3).values.tolist()

    # Convert indices to disease names
    top_3_diseases = label_encoder.inverse_transform(top_3_indices)
    
    return list(zip(top_3_diseases, top_3_confidences))

# ✅ Example Usage
if __name__ == "__main__":
    example_symptoms = ["headache", "dizziness", "nausea"]
    predictions = predict_disease(example_symptoms)

    print("\n🩺 Predicted Diseases:")
    for disease, confidence in predictions:
        print(f"➡ {disease} ({confidence:.4f} confidence)")
