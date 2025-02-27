# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# # Load the BioBERT model and tokenizer
# model_name = "dmis-lab/biobert-base-cased-v1.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

# # Create a pipeline for NER
# symptom_extractor = pipeline("ner", model=model, tokenizer=tokenizer)

# # Example input text
# text = "I have a sore throat and high fever."
# results = symptom_extractor(text)

# # Process results to extract symptoms
# symptoms = [res['word'] for res in results if 'LABEL' in res['entity']]
# print(f"Extracted Symptoms: {symptoms}")



# import spacy
# import re
# import json
# from typing import List, Dict

# class SymptomExtractor:
#     def __init__(self, medical_dictionary_path='medical_symptoms.json'):
#         # Load medical NLP model
#         self.nlp = spacy.load('en_core_web_sm')
        
#         # Load medical symptoms dictionary
#         try:
#             with open(medical_dictionary_path, 'r') as f:
#                 self.symptoms_dict = json.load(f)
#         except FileNotFoundError:
#             # Default symptoms list if no dictionary found
#             self.symptoms_dict = {
#                 "symptoms": [
#                     "fever", "cough", "headache", "pain", "fatigue", 
#                     "nausea", "dizziness", "shortness of breath", 
#                     "chest pain", "stomach ache"
#                 ]
#             }
        
#     def preprocess_text(self, text: str) -> str:
#         """Clean and normalize input text"""
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         return text
    
#     def extract_symptoms(self, text: str) -> List[Dict[str, str]]:
#         """
#         Extract potential medical symptoms from text
#         Returns list of dictionaries with symptom details
#         """
#         preprocessed_text = self.preprocess_text(text)
#         doc = self.nlp(preprocessed_text)
        
#         found_symptoms = []
        
#         # Check against predefined symptom dictionary
#         for symptom in self.symptoms_dict['symptoms']:
#             if symptom in preprocessed_text:
#                 # Find context around symptom
#                 context = self._get_symptom_context(preprocessed_text, symptom)
#                 found_symptoms.append({
#                     "symptom": symptom,
#                     "context": context
#                 })
        
#         return found_symptoms
    
#     def _get_symptom_context(self, text: str, symptom: str, window: int = 10) -> str:
#         """Get context around a symptom"""
#         symptom_index = text.find(symptom)
#         start = max(0, symptom_index - window)
#         end = min(len(text), symptom_index + len(symptom) + window)
#         return text[start:end].strip()
    
#     def detect_potential_diseases(self, symptoms: List[Dict[str, str]]) -> List[str]:
#         """
#         Basic disease inference based on symptoms
#         Note: This is a simplified heuristic and not a medical diagnosis tool
#         """
#         disease_mapping = {
#             "respiratory": ["cough", "shortness of breath"],
#             "cardiovascular": ["chest pain", "fatigue"],
#             "digestive": ["nausea", "stomach ache"],
#             "neurological": ["headache", "dizziness"]
#         }
        
#         detected_diseases = []
#         for category, related_symptoms in disease_mapping.items():
#             if any(sym['symptom'] in related_symptoms for sym in symptoms):
#                 detected_diseases.append(category)
        
#         return detected_diseases

# def main():
#     extractor = SymptomExtractor()
    
#     # Example usage
#     patient_transcript = input("Enter patient's voice transcript: ")
    
#     symptoms = extractor.extract_symptoms(patient_transcript)
#     potential_diseases = extractor.detect_potential_diseases(symptoms)
    
#     print("Extracted Symptoms:")
#     for symptom in symptoms:
#         print(f"- {symptom['symptom']} (Context: {symptom['context']})")
    
#     print("\nPotential Disease Categories:")
#     print(potential_diseases)

# if __name__ == "__main__":
#     main()







#this code from here represent the proper use of predefined symptoms 

# import spacy
# import re
# import json
# from typing import List, Dict

# class SymptomExtractor:
#     def __init__(self, medical_dictionary_path='medical_symptoms.json'):
#         self.nlp = spacy.load('en_core_web_sm')
        
#         try:
#             with open(medical_dictionary_path, 'r') as f:
#                 self.symptoms_dict = json.load(f)
#         except FileNotFoundError:
#             self.symptoms_dict = {
#                 "symptoms": [
#                     "fever", "cough", "headache", "pain", "fatigue", 
#                     "nausea", "dizziness", "shortness of breath", 
#                     "chest pain", "stomach ache"
#                 ]
#             }
        
#     def preprocess_text(self, text: str) -> str:
#         text = text.lower()
#         text = re.sub(r'[^\w\s]', '', text)
#         return text
    
#     def extract_symptoms(self, text: str) -> List[Dict[str, str]]:
#         preprocessed_text = self.preprocess_text(text)
        
#         found_symptoms = []
        
#         for symptom in self.symptoms_dict['symptoms']:
#             if symptom in preprocessed_text:
#                 context = self._get_symptom_context(preprocessed_text, symptom)
#                 found_symptoms.append({
#                     "symptom": symptom,
#                     "context": context
#                 })
        
#         return found_symptoms
    
#     def _get_symptom_context(self, text: str, symptom: str, window: int = 10) -> str:
#         symptom_index = text.find(symptom)
#         start = max(0, symptom_index - window)
#         end = min(len(text), symptom_index + len(symptom) + window)
#         return text[start:end].strip()

# def main():
#     extractor = SymptomExtractor()
    
#     patient_transcript = input("Enter patient's voice transcript: ")
    
#     symptoms = extractor.extract_symptoms(patient_transcript)
    
#     print("Extracted Symptoms:")
#     for symptom in symptoms:
#         print(f"- {symptom['symptom']} (Context: {symptom['context']})")

# if __name__ == "__main__":
#     main()












#Spacy + Medical NER combined with Keyword Matching(part-2)


# import spacy
# import re
# from transformers import pipeline

# # Load the alternative SciSpacy model (compatible with your spaCy version)
# nlp_spacy = spacy.load("en_core_sci_sm")

# # Load Transformer-based NER model (using a biomedical NER model)
# nlp_bert = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

# # Predefined symptom list for keyword matching
# SYMPTOM_LIST = [
#     "fever", "cough", "headache", "fatigue", "sore throat", "runny nose",
#     "shortness of breath", "chills", "body aches", "loss of taste", "nausea",
#     "vomiting", "diarrhea", "dizziness", "sweating", "rash", "chest pain"
# ]

# # Regex patterns for symptom variations
# SYMPTOM_PATTERNS = [
#     r"\bfever\b",
#     r"\bcough\b",
#     r"\bhead\s?ache\b",   # Matches "headache" and "head ache"
#     r"\brunny nose\b",
#     r"\bshort\s?ness\s?of\s?breath\b",
# ]

# def extract_symptoms(text):
#     """
#     Extracts symptoms using a hybrid approach:
#       1. Keyword Matching
#       2. Regex Matching
#       3. SpaCy Medical NER (using en_core_sci_sm)
#       4. Transformer-based Medical NER
#     Returns a list of unique symptoms.
#     """
#     symptoms = set()
#     text_lower = text.lower()
    
#     # 1. Keyword Matching
#     for symptom in SYMPTOM_LIST:
#         if symptom in text_lower:
#             symptoms.add(symptom)
    
#     # 2. Regex Pattern Matching
#     for pattern in SYMPTOM_PATTERNS:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             symptoms.add(match.group().lower())
    
#     # 3. SpaCy Medical NER Extraction
#     doc = nlp_spacy(text)
#     for ent in doc.ents:
#         symptoms.add(ent.text.lower())
    
#     # 4. Transformer-based NER Extraction
#     ner_results = nlp_bert(text)
#     for ent in ner_results:
#         symptoms.add(ent['word'].lower())
    
#     return list(symptoms)

# # Example usage
# if __name__ == "__main__":
#     sample_text = "i have bloating, acid reflux and loss of appetiet."
#     extracted = extract_symptoms(sample_text)
#     print("Extracted Symptoms:", extracted)






# this code from here is for the connection between symptoms and disease_model

import spacy
import re
import torch
import joblib
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# -------------------------------
# 1. Load NER Models for Symptom Extraction
# -------------------------------

# Load the alternative SciSpacy model (compatible with your spaCy version)
nlp_spacy = spacy.load("en_core_sci_sm")

# Load a transformer-based biomedical NER model from Hugging Face
nlp_bert = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

# -------------------------------
# 2. Load Your Trained Disease Classification Model
# -------------------------------

model_path = "./trained_disease_model"
disease_model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
label_encoder = joblib.load(f"{model_path}/label_encoder.joblib")

# -------------------------------
# 3. Define Hybrid Symptom Extraction Methods
# -------------------------------

# Predefined symptom list for quick keyword matching
SYMPTOM_LIST = [
    "fever", "cough", "headache", "fatigue", "sore throat", "runny nose",
    "shortness of breath", "chills", "body aches", "loss of taste", "nausea",
    "vomiting", "diarrhea", "dizziness", "sweating", "rash", "chest pain"
]

# Regular expression patterns for common symptom variations
SYMPTOM_PATTERNS = [
    r"\bfever\b",
    r"\bcough\b",
    r"\bhead\s?ache\b",   # Matches both "headache" and "head ache"
    r"\brunny nose\b",
    r"\bshort\s?ness\s?of\s?breath\b",
]

def extract_symptoms(text):
    """
    Extracts symptoms from text using a hybrid approach:
      1. Keyword Matching from a predefined list
      2. Regex Pattern Matching for variations
      3. SciSpacy Medical NER Extraction
      4. Transformer-based Medical NER Extraction
    Returns a list of unique symptoms (all in lowercase).
    """
    symptoms = set()
    text_lower = text.lower()
    
    # Method 1: Keyword Matching
    for symptom in SYMPTOM_LIST:
        if symptom in text_lower:
            symptoms.add(symptom)
    
    # Method 2: Regex Pattern Matching
    for pattern in SYMPTOM_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            symptoms.add(match.group().lower())
    
    # Method 3: SciSpacy Medical NER
    doc = nlp_spacy(text)
    for ent in doc.ents:
        symptoms.add(ent.text.lower())
    
    # Method 4: Transformer-based Medical NER
    ner_results = nlp_bert(text)
    for ent in ner_results:
        symptoms.add(ent['word'].lower())
    
    return list(symptoms)

# -------------------------------
# 4. Integrate Extraction with Disease Prediction
# -------------------------------

def predict_disease_from_text(user_text):
    """
    Extracts symptoms from the user-provided text using the hybrid extractor,
    displays the extracted symptoms, then predicts the disease using the
    trained DistilBert model.
    Returns a string showing both the symptoms and the predicted disease with confidence.
    """
    # Step 1: Extract symptoms
    extracted_symptoms = extract_symptoms(user_text)
    print("\n🔎 Extracted Symptoms:", extracted_symptoms)

    if not extracted_symptoms:
        return "⚠️ No recognizable symptoms found."
    
    # Step 2: Prepare the extracted symptoms for the model (join into a single string)
    symptoms_text = " ".join(extracted_symptoms)
    encoding = tokenizer(symptoms_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    # Step 3: Use the trained model to predict the disease
    disease_model.eval()
    with torch.no_grad():
        output = disease_model(**encoding)
        logits = output.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]
    
    return f"\n🩺 Predicted Disease: {predicted_disease} (Confidence: {confidence:.4f})"

# -------------------------------
# 5. Main Execution: Get User Input and Display Results
# -------------------------------

if __name__ == "__main__":
    user_input = input("Describe your symptoms: ")
    result = predict_disease_from_text(user_input)
    print(result)
