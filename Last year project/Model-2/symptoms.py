import spacy
from transformers import pipeline

# ✅ Load Spacy Transformer-based NLP Model
nlp_spacy = spacy.load("en_core_web_trf")

# ✅ Load BioBERT-based Biomedical NER Model
bio_ner = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

# ✅ Define Common Symptom Keywords
SYMPTOM_LIST = [
    "fever", "cough", "headache", "fatigue", "sore throat",
    "shortness of breath", "chest pain", "vomiting", "dizziness", "nausea", "stomach pain"
]

# ✅ List of Words to Ignore (Adjectives, Intensifiers)
EXCLUDE_WORDS = ["severe", "mild", "slight", "intense", "extreme", "chronic"]

def extract_symptoms(text):
    """
    Extracts symptoms using:
    1️⃣ Spacy Transformer-based NER (`en_core_web_trf`)
    2️⃣ BioBERT-based Biomedical NER
    3️⃣ Keyword Matching
    4️⃣ Removes non-symptom words (e.g., "severe", "chronic")
    """
    symptoms = set()

    # 1️⃣ Extract Symptoms Using Spacy NER
    doc = nlp_spacy(text)
    for ent in doc.ents:
        symptom = ent.text.lower()
        if symptom not in EXCLUDE_WORDS:
            symptoms.add(symptom)

    # 2️⃣ Extract Symptoms Using BioBERT NER
    ner_results = bio_ner(text)
    for ent in ner_results:
        symptom = ent['word'].lower()
        if symptom not in EXCLUDE_WORDS:
            symptoms.add(symptom)

    # 3️⃣ Extract Symptoms Using Keyword Matching
    for symptom in SYMPTOM_LIST:
        if symptom in text.lower():
            symptoms.add(symptom)

    return list(symptoms)

# ✅ Example Usage
if __name__ == "__main__":
    text = "My stomach has been hurting, and I feel bloated after meals."
    extracted_symptoms = extract_symptoms(text)
    print("🔎 Extracted Symptoms:", extracted_symptoms)
