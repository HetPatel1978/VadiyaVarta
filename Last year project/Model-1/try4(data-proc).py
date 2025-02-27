# import spacy
# import re
# from typing import List, Dict

# class SymptomExtractor:
#     def __init__(self):
#         # Load medical NLP model
#         self.nlp = spacy.load('en_core_web_sm')
        
#         # Predefined list of common symptoms
#         self.symptom_keywords = [
#             'headache', 'headaches', 'running nose', 'runny nose', 
#             'fever', 'temperature', 'cough', 'pain', 
#             'sore throat', 'fatigue', 'tired', 'dizzy'
#         ]
    
#     def extract_symptoms(self, text: str) -> List[Dict[str, str]]:
#         # Convert to lowercase for easier matching
#         text_lower = text.lower()
        
#         found_symptoms = []
        
#         # Check against predefined symptom keywords
#         for symptom in self.symptom_keywords:
#             if symptom in text_lower:
#                 # Find the original word in the text with correct casing
#                 original_symptom = re.findall(rf'\b{symptom}\b', text, re.IGNORECASE)
#                 if original_symptom:
#                     # Get context around the symptom
#                     context = self._get_symptom_context(text, original_symptom[0])
#                     found_symptoms.append({
#                         "symptom": original_symptom[0],
#                         "context": context
#                     })
        
#         return found_symptoms
    
#     def _get_symptom_context(self, text: str, symptom: str, window: int = 20) -> str:
#         """Extract context around the symptom"""
#         # Find the index of the symptom
#         match = re.search(re.escape(symptom), text, re.IGNORECASE)
#         if match:
#             start = max(0, match.start() - window)
#             end = min(len(text), match.end() + window)
#             return text[start:end].strip()
#         return ""

# def main():
#     extractor = SymptomExtractor()
    
#     # Get input from user
#     patient_transcript = input("Enter patient's voice transcript: ")
    
#     # Extract symptoms
#     symptoms = extractor.extract_symptoms(patient_transcript)
    
#     # Print results
#     if symptoms:
#         print("Extracted Symptoms:")
#         for symptom in symptoms:
#             print(f"- {symptom['symptom']} (Context: {symptom['context']})")
#     else:
#         print("No symptoms detected. Please check the input or expand the symptom keywords.")

# if __name__ == "__main__":
#     main()








# lets extract symptoms with the help of trained dataset:

# import pandas as pd
# import numpy as np

# class MedMentionsProcessor:
#     # Expanded set of potentially relevant semantic types
#     SYMPTOM_SEMANTIC_TYPES = {
#         'T033',  # Finding
#         'T038',  # Disorder
#         'T037',  # Injury or Poisoning
#     }
    
#     @classmethod
#     def load_dataset(cls, file_path: str) -> pd.DataFrame:
#         """
#         Robust dataset loading for PubTator format
#         """
#         try:
#             data = []
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     parts = line.strip().split('\t')
                    
#                     # Check if line is an annotation
#                     if len(parts) == 6:
#                         data.append({
#                             'PMID': parts[0],
#                             'StartIdx': parts[1],
#                             'EndIdx': parts[2],
#                             'EntityText': parts[3],
#                             'SemanticType': parts[4],
#                             'UMLS_ID': parts[5]
#                         })
            
#             df = pd.DataFrame(data)
#             return df
        
#         except Exception as e:
#             print(f"Error loading dataset: {e}")
#             return pd.DataFrame()

#     @classmethod
#     def filter_symptom_entities(cls, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Filter potentially symptom-related entities
#         """
#         symptoms_df = df[df['SemanticType'].isin(cls.SYMPTOM_SEMANTIC_TYPES)].copy()
        
#         print(f"Found {len(symptoms_df)} potential symptom/disorder entities")
#         return symptoms_df

#     def process_dataset(self, input_file: str, output_file: str):
#         """
#         Process MedMentions dataset
#         """
#         df = self.load_dataset(input_file)
#         symptoms_df = self.filter_symptom_entities(df)
        
#         if not symptoms_df.empty:
#             symptoms_df.to_csv(output_file, index=False)
#             print(f"Saved {len(symptoms_df)} entities to {output_file}")
#         else:
#             print("No symptom entities found to save.")

# def main():
#     processor = MedMentionsProcessor()
#     processor.process_dataset(
#         input_file='C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/st21pv/data/corpus_pubtator.txt',
#         output_file='symptom_entities.csv'
#     )

# if __name__ == '__main__':
#     main()








# lets go for the high quality:

# import pandas as pd
# import re
# import numpy as np

# class MedMentionsQualityFilter:
#     @staticmethod
#     def load_stopwords():
#         """Create a set of medical and general stopwords"""
#         return {
#             'disease', 'syndrome', 'condition', 'disorder', 
#             'infection', 'type', 'clinical', 'medical', 
#             'patient', 'therapy', 'treatment', 'management'
#         }

#     @classmethod
#     def filter_entities(cls, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply multiple quality filters with robust error handling
#         """
#         stopwords = cls.load_stopwords()
        
#         def is_valid_entity(row):
#             # Handle NaN or non-string values
#             if pd.isna(row['EntityText']) or not isinstance(row['EntityText'], str):
#                 return False
            
#             text = row['EntityText'].lower().strip()
            
#             # Length filter (3-50 characters)
#             if len(text) < 3 or len(text) > 50:
#                 return False
            
#             # Remove entries with too many stopwords
#             words = text.split()
#             word_count = len(words)
#             stopword_count = sum(1 for word in words if word in stopwords)
#             if word_count > 0 and stopword_count / word_count > 0.4:
#                 return False
            
#             # Remove purely numeric or special character entries
#             if re.match(r'^[\d\W]+$', text):
#                 return False
            
#             # Remove very generic terms
#             if text in {'event', 'finding', 'condition', 'state', 'type'}:
#                 return False
            
#             return True
        
#         # Apply filters with error handling
#         try:
#             # Ensure no NaN values in EntityText column
#             df_cleaned = df.dropna(subset=['EntityText'])
            
#             # Apply filtering
#             filtered_df = df_cleaned[df_cleaned.apply(is_valid_entity, axis=1)].copy()
            
#             print(f"Original entities: {len(df_cleaned)}")
#             print(f"Filtered entities: {len(filtered_df)}")
#             print(f"Removed: {len(df_cleaned) - len(filtered_df)} low-quality entities")
            
#             return filtered_df
        
#         except Exception as e:
#             print(f"Error during filtering: {e}")
#             return pd.DataFrame()

# def main():
#     # Load previously extracted entities
#     try:
#         df = pd.read_csv('symptom_entities.csv')
        
#         # Initialize quality filter
#         quality_filter = MedMentionsQualityFilter()
        
#         # Apply filters
#         filtered_df = quality_filter.filter_entities(df)
        
#         # Save filtered results
#         if not filtered_df.empty:
#             filtered_df.to_csv('high_quality_symptoms.csv', index=False)
#             print("Filtered entities saved successfully.")
#         else:
#             print("No entities passed the quality filter.")
    
#     except Exception as e:
#         print(f"Error processing file: {e}")

# if __name__ == '__main__':
#     main()










# lets do some further data processing 

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AdvancedSymptomProcessor:
    SEMANTIC_TYPES = {
        'T033': 'Finding',
        'T038': 'Disorder',
        'T037': 'Injury',
        'T184': 'Sign/Symptom',
        'T047': 'Disease',
        'T048': 'Mental/Behavioral'
    }
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.setup_logging()
        self.similarity_threshold = similarity_threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('symptom_processing.log'),
                logging.StreamHandler()
            ]
        )
        
    def process_symptoms(self, input_file: str, output_file: str):
        try:
            logging.info(f"Loading data from {input_file}")
            df = pd.read_csv(input_file)
            
            # Add quality metrics
            df = self.add_quality_metrics(df)
            
            # Remove similar symptoms
            df = self.remove_similar_symptoms(df)
            
            # Add semantic type descriptions
            df['SemanticTypeDesc'] = df['SemanticType'].map(self.SEMANTIC_TYPES)
            
            # Save enhanced dataset
            df.to_csv(output_file, index=False)
            logging.info(f"Enhanced dataset saved to {output_file}")
            
            self.print_statistics(df)
            
        except Exception as e:
            logging.error(f"Error processing symptoms: {e}")
            raise
            
    def add_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Adding quality metrics")
        try:
            df['SymptomLength'] = df['EntityText'].str.len()
            df['WordCount'] = df['EntityText'].str.split().str.len()
            df['HasNumbers'] = df['EntityText'].str.contains('\d').astype(int)
            df['Complexity'] = df.apply(self.calculate_complexity, axis=1)
            df['FrequencyCount'] = df.groupby('EntityText')['EntityText'].transform('count')
            return df
        except Exception as e:
            logging.error(f"Error adding quality metrics: {e}")
            raise
            
    def calculate_complexity(self, row: pd.Series) -> float:
        """Calculate symptom complexity score"""
        text = row['EntityText']
        return (
            len(text) * 0.3 +
            len(text.split()) * 0.4 +
            (text.count('/') + text.count(',')) * 0.3
        )
            
    def remove_similar_symptoms(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Removing similar symptoms")
        try:
            # Convert symptoms to TF-IDF vectors
            symptoms = df['EntityText'].str.lower().tolist()
            tfidf_matrix = self.vectorizer.fit_transform(symptoms)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find groups of similar symptoms
            similar_groups = []
            processed = set()
            
            for i in range(len(symptoms)):
                if i in processed:
                    continue
                    
                similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
                if len(similar_indices) > 1:  # If there are similar symptoms
                    group = list(similar_indices)
                    similar_groups.append(group)
                    processed.update(group)
            
            # Keep most frequent symptom from each group
            keep_indices = set(range(len(symptoms)))
            for group in similar_groups:
                group_df = df.iloc[group]
                keep_idx = group_df['FrequencyCount'].idxmax()
                group_indices = set(group)
                group_indices.remove(keep_idx)
                keep_indices -= group_indices
            
            return df.iloc[list(keep_indices)].copy()
            
        except Exception as e:
            logging.error(f"Error removing similar symptoms: {e}")
            raise
            
    def print_statistics(self, df: pd.DataFrame):
        stats = {
            "Total Symptoms": len(df),
            "Unique UMLS Concepts": df['UMLS_ID'].nunique(),
            "Average Symptom Length": df['SymptomLength'].mean(),
            "Semantic Type Distribution": df['SemanticTypeDesc'].value_counts().to_dict()
        }
        
        logging.info("\nDataset Statistics:")
        for key, value in stats.items():
            logging.info(f"{key}: {value}")
            
        logging.info("\nTop 10 Most Complex Symptoms:")
        logging.info(df.nlargest(10, 'Complexity')[['EntityText', 'Complexity', 'SemanticTypeDesc']])

def main():
    try:
        processor = AdvancedSymptomProcessor()
        processor.process_symptoms(
            input_file= r'C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/Temp/high_quality_symptoms.csv',
            output_file='final_symptoms.csv'
        )
    except Exception as e:
        logging.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()