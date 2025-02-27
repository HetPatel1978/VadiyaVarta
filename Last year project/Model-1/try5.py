

#MAIN TRAINING MODEL


# import pandas as pd
# import torch
# import logging
# import numpy as np
# from sklearn.model_selection import train_test_split
# from transformers import (
#     DistilBertTokenizer, 
#     DistilBertForSequenceClassification, 
#     Trainer, 
#     TrainingArguments
# )
# from datasets import Dataset
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# def compute_metrics(pred):
#     """
#     Compute evaluation metrics
#     """
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
    
#     # Precision, Recall, F1, Support
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
#     # Accuracy
#     accuracy = accuracy_score(labels, preds)
    
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1
#     }

# def preprocess_symptoms(df):
#     """
#     Combine multiple symptom columns into a single text column
#     """
#     symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
#     df['All_Symptoms'] = df[symptom_cols].apply(lambda row: ' '.join(str(val) for val in row if pd.notna(val)), axis=1)
#     return df

# def load_and_prepare_dataset(file_path):
#     """
#     Load and preprocess the main dataset
#     """
#     df = pd.read_csv(file_path)
#     df = preprocess_symptoms(df)
    
#     return df

# def prepare_datasets(df, test_size=0.2, random_state=42):
#     """
#     Prepare tokenized datasets for training
#     """
#     # Encode labels
#     df['label'] = df['Disease'].astype('category').cat.codes
    
#     # Split data
#     train_texts, val_texts, train_labels, val_labels = train_test_split(
#         df["All_Symptoms"].tolist(), 
#         df["label"].tolist(), 
#         test_size=test_size, 
#         random_state=random_state
#     )
    
#     # Initialize tokenizer
#     tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
#     # Tokenization function
#     def tokenize_function(texts):
#         return tokenizer(texts, padding=True, truncation=True, max_length=512)
    
#     # Prepare encodings
#     train_encodings = tokenize_function(train_texts)
#     val_encodings = tokenize_function(val_texts)
    
#     # Create datasets
#     train_dataset = Dataset.from_dict({
#         "input_ids": train_encodings["input_ids"],
#         "attention_mask": train_encodings["attention_mask"],
#         "labels": train_labels
#     })
    
#     val_dataset = Dataset.from_dict({
#         "input_ids": val_encodings["input_ids"],
#         "attention_mask": val_encodings["attention_mask"],
#         "labels": val_labels
#     })
    
#     return train_dataset, val_dataset, tokenizer

# def train_model(train_dataset, val_dataset, tokenizer):
#     """
#     Train and save the DistilBERT model
#     """
#     # Load model
#     model = DistilBertForSequenceClassification.from_pretrained(
#         "distilbert-base-uncased", 
#         num_labels=len(set(train_dataset['labels']))
#     )
    
#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir="./disease_classification_results",
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=16,
#         per_device_eval_batch_size=16,
#         num_train_epochs=5,
#         weight_decay=0.01,
#         logging_dir="./disease_classification_logs",
#         logging_steps=10,
#         load_best_model_at_end=True,
#         metric_for_best_model="accuracy"
#     )
    
#     # Initialize Trainer with metrics computation
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         compute_metrics=compute_metrics
#     )
    
#     # Train model
#     trainer.train()
    
#     # Evaluate model
#     eval_results = trainer.evaluate()
#     print("Evaluation Results:", eval_results)
    
#     # Save model and tokenizer
#     model.save_pretrained("./trained_disease_model")
#     tokenizer.save_pretrained("./trained_disease_model")

# def main():
#     # Path to your main dataset
#     dataset_path = r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/dataset.csv"
    
#     # Load and prepare dataset
#     df = load_and_prepare_dataset(dataset_path)
    
#     # Prepare datasets
#     train_dataset, val_dataset, tokenizer = prepare_datasets(df)
    
#     # Train model
#     train_model(train_dataset, val_dataset, tokenizer)

# if __name__ == "__main__":
#     main()
    
    
    










#THIS CODE FROM HERE IS FOR DATA TRAINING WITH THE HELP OF MULTIPLE FOLDS.
    
    
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
# from transformers import (
#     DistilBertTokenizer, 
#     DistilBertForSequenceClassification, 
#     Trainer, 
#     TrainingArguments
# )
# from datasets import Dataset

# def preprocess_symptoms(df):
#     """Combine symptom columns into single text column"""
#     symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
#     df['All_Symptoms'] = df[symptom_cols].apply(lambda row: ' '.join(str(val) for val in row if pd.notna(val)), axis=1)
#     return df

# def tokenize_dataset(texts, labels, tokenizer, max_length=512):
#     """Tokenize input texts"""
#     encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_length)
#     return Dataset.from_dict({
#         "input_ids": encodings["input_ids"],
#         "attention_mask": encodings["attention_mask"],
#         "labels": labels
#     })

# def cross_validate_model(df, n_splits=5):
#     """Perform cross-validation"""
#     # Prepare data
#     df = preprocess_symptoms(df)
    
#     # Encode labels
#     df['label'] = df['Disease'].astype('category').cat.codes
#     label_map = dict(enumerate(df['Disease'].astype('category').cat.categories))
    
#     # Stratified K-Fold
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
#     # Tokenizer
#     tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
#     # Results storage
#     cv_results = []
    
#     # Cross-validation loop
#     for fold, (train_idx, val_idx) in enumerate(skf.split(df['All_Symptoms'], df['label']), 1):
#         print(f"\nFold {fold}")
        
#         # Split data
#         train_data = df.loc[train_idx]
#         val_data = df.loc[val_idx]
        
#         # Prepare datasets
#         train_dataset = tokenize_dataset(
#             train_data['All_Symptoms'].tolist(), 
#             train_data['label'].tolist(), 
#             tokenizer
#         )
#         val_dataset = tokenize_dataset(
#             val_data['All_Symptoms'].tolist(), 
#             val_data['label'].tolist(), 
#             tokenizer
#         )
        
#         # Model configuration
#         model = DistilBertForSequenceClassification.from_pretrained(
#             "distilbert-base-uncased", 
#             num_labels=len(label_map)
#         )
        
#         # Training arguments
#         training_args = TrainingArguments(
#             output_dir=f"./results_fold_{fold}",
#             evaluation_strategy="epoch",
#             save_strategy="no",
#             learning_rate=2e-5,
#             per_device_train_batch_size=16,
#             per_device_eval_batch_size=16,
#             num_train_epochs=3,
#             weight_decay=0.01,
#             logging_steps=10,
#             load_best_model_at_end=False
#         )
        
#         # Trainer
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=val_dataset
#         )
        
#         # Train
#         trainer.train()
        
#         # Evaluate
#         eval_results = trainer.evaluate()
#         print("Evaluation Results:", eval_results)
        
#         # Predictions for detailed metrics
#         predictions = trainer.predict(val_dataset)
#         preds = np.argmax(predictions.predictions, axis=-1)
        
#         # Classification report
#         report = classification_report(
#             val_data['label'], 
#             preds, 
#             target_names=list(label_map.values())
#         )
#         print("\nDetailed Classification Report:")
#         print(report)
        
#         cv_results.append({
#             'fold': fold,
#             'eval_loss': eval_results['eval_loss'],
#             'predictions': preds,
#             'true_labels': val_data['label'].tolist()
#         })
    
#     return cv_results, label_map

# def main():
#     # Load dataset
#     dataset_path = r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/dataset.csv"
#     df = pd.read_csv(dataset_path)
    
#     # Perform cross-validation
#     results, label_map = cross_validate_model(df)

# if __name__ == "__main__":
#     main()











#This code from here is to see the performance


# import pandas as pd
# import numpy as np
# import torch
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     classification_report, 
#     confusion_matrix, 
#     accuracy_score
# )
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# def load_and_preprocess_data(file_path):
#     """Load and preprocess dataset"""
#     df = pd.read_csv(file_path)
    
#     # Combine symptom columns
#     symptom_cols = [col for col in df.columns if col.startswith('Symptom_')]
#     df['All_Symptoms'] = df[symptom_cols].apply(
#         lambda row: ' '.join(str(val) for val in row if pd.notna(val)), 
#         axis=1
#     )
    
#     # Encode labels
#     df['label'] = df['Disease'].astype('category').cat.codes
#     return df

# def validate_model(df, test_size=0.2):
#     """Comprehensive model validation"""
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['All_Symptoms'], 
#         df['label'], 
#         test_size=test_size, 
#         random_state=42
#     )
    
#     # Load model and tokenizer
#     tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
#     model = DistilBertForSequenceClassification.from_pretrained("./trained_disease_model")
    
#     # Tokenize test data
#     test_encodings = tokenizer(
#         X_test.tolist(), 
#         padding=True, 
#         truncation=True, 
#         return_tensors='pt'
#     )
    
#     # Prediction
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**test_encodings)
#         predictions = torch.argmax(outputs.logits, dim=1).numpy()
    
#     # Performance metrics
#     print("Classification Report:")
#     print(classification_report(
#         y_test, 
#         predictions, 
#         target_names=df['Disease'].unique()
#     ))
    
#     # Confusion Matrix
#     conf_matrix = confusion_matrix(y_test, predictions)
#     print("\nConfusion Matrix:")
#     print(conf_matrix)
    
#     # Additional metrics
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"\nOverall Accuracy: {accuracy:.4f}")
    
#     # Confidence interval
#     confidence_interval = 1.96 * np.sqrt(
#         (accuracy * (1 - accuracy)) / len(y_test)
#     )
#     print(f"95% Confidence Interval: {accuracy:.4f} ± {confidence_interval:.4f}")

# def main():
#     dataset_path = r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/dataset.csv"
#     df = load_and_preprocess_data(dataset_path)
#     validate_model(df)

# if __name__ == "__main__":
#     main()







#THIS CODE FROM HERE TRAINS A NEW DATSET.

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def preprocess_symptoms(df):
    # Combine all symptom columns into a single text column
    symptom_columns = [col for col in df.columns if col != 'diseases']
    df['symptoms'] = df[symptom_columns].apply(
        lambda row: ' '.join(
            col for col in row.index if row[col] == 1
        ), 
        axis=1
    )
    return df

def prepare_datasets(df, test_size=0.2, random_state=42):
    # Use LabelEncoder to create dense, consecutive label indices
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['diseases'])
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['symptoms'], 
        df['label'], 
        test_size=test_size, 
        random_state=random_state
    )
    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(texts):
        return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512)
    
    train_encodings = tokenize_function(train_texts)
    val_encodings = tokenize_function(val_texts)
    
    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels.tolist()
    })
    
    val_dataset = Dataset.from_dict({
        "input_ids": val_encodings["input_ids"],
        "attention_mask": val_encodings["attention_mask"],
        "labels": val_labels.tolist()
    })
    
    # Print label information for debugging
    print("Label range:", min(train_labels), "-", max(train_labels))
    print("Number of unique labels:", len(set(train_labels)))
    print("Original disease labels:", list(label_encoder.classes_))
    
    return train_dataset, val_dataset, tokenizer, label_encoder

def train_model(train_dataset, val_dataset, tokenizer, label_encoder):
    # Explicitly determine number of labels
    unique_labels = len(label_encoder.classes_)
    print(f"Number of unique labels: {unique_labels}")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=unique_labels
    )
    
    training_args = TrainingArguments(
        output_dir="./disease_classification_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./disease_classification_logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    # Save model and label encoder
    model.save_pretrained("./trained_disease_model")
    tokenizer.save_pretrained("./trained_disease_model")
    
    # Optionally save label encoder for future use
    import joblib
    joblib.dump(label_encoder, "./trained_disease_model/label_encoder.joblib")

def main():
    df = pd.read_csv(r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/dataset3.csv")
    df = preprocess_symptoms(df)
    
    train_dataset, val_dataset, tokenizer, label_encoder = prepare_datasets(df)
    train_model(train_dataset, val_dataset, tokenizer, label_encoder)

if __name__ == "__main__":
    main()