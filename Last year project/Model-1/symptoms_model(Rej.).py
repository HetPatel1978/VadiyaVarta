# import pandas as pd
# import nltk
# from nltk.tokenize import word_tokenize
# import json

# # Ensure NLTK tokenizer is available
# nltk.download('punkt')

# # Load the ST21pv dataset (Modify the file path if needed)
# file_path = r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/Temp/final_symptoms.csv"  # Change this to your actual dataset path
# df = pd.read_csv(file_path)

# # Selecting necessary columns
# df = df[['PMID', 'StartIdx', 'EndIdx', 'EntityText', 'SemanticTypeDesc']]  

# # Function to tokenize and label text using BIO tagging
# def convert_to_bio(text, entity_start, entity_end, entity_text):
#     tokens = word_tokenize(text)
#     labels = ["O"] * len(tokens)  # Initialize all tokens as "O"

#     entity_tokens = word_tokenize(entity_text)
    
#     # Find entity position in tokenized text
#     start_idx = -1
#     for i in range(len(tokens) - len(entity_tokens) + 1):
#         if tokens[i:i+len(entity_tokens)] == entity_tokens:
#             start_idx = i
#             break

#     if start_idx != -1:
#         labels[start_idx] = "B-Symptom"  # Beginning of entity
#         for j in range(start_idx + 1, start_idx + len(entity_tokens)):
#             labels[j] = "I-Symptom"  # Inside entity
    
#     return list(zip(tokens, labels))

# # Process each row
# processed_data = []
# for _, row in df.iterrows():
#     text = row['EntityText']
#     start = row['StartIdx']
#     end = row['EndIdx']
#     entity_text = row['EntityText']
    
#     tokenized_data = convert_to_bio(text, start, end, entity_text)
#     processed_data.append(tokenized_data)

# # Save as JSON for training
# with open("bio_ner_data.json", "w") as f:
#     json.dump(processed_data, f, indent=4)

# print("✅ Data preprocessing complete! Saved as bio_ner_data.json.")









import spacy
import re
from typing import List, Dict

class SymptomExtractor:
    def __init__(self):
        # Load medical NLP model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Predefined list of common symptoms
        self.symptom_keywords = [
            'headache', 'headaches', 'running nose', 'runny nose', 
            'fever', 'temperature', 'cough', 'pain', 
            'sore throat', 'fatigue', 'tired', 'dizzy'
        ]
    
    def extract_symptoms(self, text: str) -> List[Dict[str, str]]:
        # Convert to lowercase for easier matching
        text_lower = text.lower()
        
        found_symptoms = []
        
        # Check against predefined symptom keywords
        for symptom in self.symptom_keywords:
            if symptom in text_lower:
                # Find the original word in the text with correct casing
                original_symptom = re.findall(rf'\b{symptom}\b', text, re.IGNORECASE)
                if original_symptom:
                    # Get context around the symptom
                    context = self._get_symptom_context(text, original_symptom[0])
                    found_symptoms.append({
                        "symptom": original_symptom[0],
                        "context": context
                    })
        
        return found_symptoms
    
    def _get_symptom_context(self, text: str, symptom: str, window: int = 20) -> str:
        """Extract context around the symptom"""
        # Find the index of the symptom
        match = re.search(re.escape(symptom), text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            return text[start:end].strip()
        return ""

def main():
    extractor = SymptomExtractor()
    
    # Get input from user
    patient_transcript = input("Enter patient's voice transcript: ")
    
    # Extract symptoms
    symptoms = extractor.extract_symptoms(patient_transcript)
    
    # Print results
    if symptoms:
        print("Extracted Symptoms:")
        for symptom in symptoms:
            print(f"- {symptom['symptom']} (Context: {symptom['context']})")
    else:
        print("No symptoms detected. Please check the input or expand the symptom keywords.")

if __name__ == "__main__":
    main()
