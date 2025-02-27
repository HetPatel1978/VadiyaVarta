# import pandas as pd
# import numpy as np
# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# import joblib
# from tqdm import tqdm

# def load_model_components(model_path="./trained_disease_model"):
#     """Load the trained model, tokenizer, and label encoder"""
#     tokenizer = DistilBertTokenizer.from_pretrained(model_path)
#     model = DistilBertForSequenceClassification.from_pretrained(model_path)
#     label_encoder = joblib.load(f"{model_path}/label_encoder.joblib")
#     return model, tokenizer, label_encoder

# def predict_with_confidence(text, model, tokenizer):
#     """Get prediction and confidence score"""
#     inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
#     outputs = model(**inputs)
#     probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     predicted_label = outputs.logits.argmax(-1).item()
#     confidence = probabilities[0][predicted_label].item()
#     return predicted_label, confidence

# def evaluate_model_performance(test_df, model_path="./trained_disease_model"):
#     """Comprehensive model evaluation"""
#     # Load model components
#     model, tokenizer, label_encoder = load_model_components(model_path)
#     model.eval()  # Set model to evaluation mode

#     # Lists to store results
#     true_labels = []
#     predicted_labels = []
#     confidences = []
#     misclassified_examples = []

#     # Make predictions
#     print("Making predictions...")
#     for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
#         true_label = label_encoder.transform([row['diseases']])[0]
#         pred_label, confidence = predict_with_confidence(row['symptoms'], model, tokenizer)
        
#         true_labels.append(true_label)
#         predicted_labels.append(pred_label)
#         confidences.append(confidence)

#         # Store misclassified examples
#         if true_label != pred_label:
#             misclassified_examples.append({
#                 'True Disease': row['diseases'],
#                 'Predicted Disease': label_encoder.inverse_transform([pred_label])[0],
#                 'Symptoms': row['symptoms'],
#                 'Confidence': confidence
#             })

#     # Calculate metrics
#     print("\nCalculating metrics...")
#     accuracy = accuracy_score(true_labels, predicted_labels)
#     class_report = classification_report(
#         true_labels, 
#         predicted_labels, 
#         target_names=label_encoder.classes_,
#         output_dict=True
#     )
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)

#     # Create detailed results dictionary
#     results = {
#         'accuracy': accuracy,
#         'classification_report': class_report,
#         'confusion_matrix': conf_matrix,
#         'misclassified_examples': misclassified_examples,
#         'confidence_scores': confidences,
#         'true_labels': true_labels,
#         'predicted_labels': predicted_labels,
#         'label_names': label_encoder.classes_
#     }

#     return results

# def visualize_results(results):
#     """Create visualizations of model performance"""
#     # 1. Confusion Matrix Heatmap
#     plt.figure(figsize=(15, 10))
#     sns.heatmap(results['confusion_matrix'], cmap='YlOrRd')
#     plt.title('Confusion Matrix')
#     plt.savefig('confusion_matrix.png')
#     plt.close()

#     # 2. Confidence Distribution
#     plt.figure(figsize=(10, 6))
#     sns.histplot(results['confidence_scores'], bins=50)
#     plt.title('Distribution of Prediction Confidence Scores')
#     plt.xlabel('Confidence')
#     plt.ylabel('Count')
#     plt.savefig('confidence_distribution.png')
#     plt.close()

#     # 3. Class-wise Performance
#     class_metrics = pd.DataFrame(results['classification_report']).T
#     plt.figure(figsize=(15, 8))
#     sns.barplot(data=class_metrics.iloc[:-3], y=class_metrics.iloc[:-3].index, x='f1-score')
#     plt.title('F1 Score by Disease Class')
#     plt.xlabel('F1 Score')
#     plt.ylabel('Disease')
#     plt.tight_layout()
#     plt.savefig('class_performance.png')
#     plt.close()

# def generate_report(results):
#     """Generate a detailed performance report"""
#     # Overall metrics
#     print("\n=== Model Performance Report ===")
#     print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    
#     # Classification report
#     print("\nDetailed Classification Report:")
#     print(pd.DataFrame(results['classification_report']).T)
    
#     # Misclassification analysis
#     print("\nMisclassification Analysis:")
#     misclassified_df = pd.DataFrame(results['misclassified_examples'])
#     print(f"\nTotal misclassified examples: {len(misclassified_df)}")
    
#     # Most common misclassifications
#     if len(misclassified_df) > 0:
#         print("\nTop 10 Most Frequently Misclassified Diseases:")
#         print(misclassified_df['True Disease'].value_counts().head(10))
        
#         print("\nExamples of Misclassified Cases:")
#         print(misclassified_df.head().to_string())
    
#     # Confidence analysis
#     print("\nConfidence Score Analysis:")
#     confidences = results['confidence_scores']
#     print(f"Mean confidence: {np.mean(confidences):.4f}")
#     print(f"Median confidence: {np.median(confidences):.4f}")
#     print(f"Min confidence: {np.min(confidences):.4f}")
#     print(f"Max confidence: {np.max(confidences):.4f}")

# def main():
#     # Load your test dataset
#     test_df = pd.read_csv(r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/dataset2.csv")
    
#     # Preprocess symptoms (using the same preprocessing as in training)
#     symptom_columns = [col for col in test_df.columns if col != 'diseases']
#     test_df['symptoms'] = test_df[symptom_columns].apply(
#         lambda row: ' '.join(col for col in row.index if row[col] == 1), 
#         axis=1
#     )
    
#     # Evaluate model
#     print("Evaluating model...")
#     results = evaluate_model_performance(test_df)
    
#     # Generate visualizations
#     print("Generating visualizations...")
#     visualize_results(results)
    
#     # Generate and save detailed report
#     print("Generating report...")
#     generate_report(results)
    
#     print("\nEvaluation complete! Check the generated files for visualizations.")

# if __name__ == "__main__":
#     main()






# Prediction for the disease model:

import torch
import joblib
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load trained model and tokenizer
model_path = r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/Temp/trained_disease_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load label encoder to map predictions back to disease names
label_encoder = joblib.load(f"{model_path}/label_encoder.joblib")

def predict_disease_with_confidence(symptoms_list):
    """
    Predicts the disease based on given symptoms and returns the confidence score.
    :param symptoms_list: List of symptom names as strings
    :return: Predicted disease with confidence score
    """
    # Convert symptoms into a single string
    symptoms_text = " ".join(symptoms_list)

    # Tokenize input
    encoding = tokenizer(symptoms_text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(**encoding)
        logits = output.logits

        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)

        # Get the predicted class and confidence score
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Decode label back to disease name
    predicted_disease = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_disease, confidence

# Example Test Case
example_symptoms = ["fatigue", "headache", "pain"]
predicted_disease, confidence_score = predict_disease_with_confidence(example_symptoms)

print(f"Predicted Disease: {predicted_disease}")
print(f"Confidence Score: {confidence_score:.4f} (Accuracy for this prediction)")
