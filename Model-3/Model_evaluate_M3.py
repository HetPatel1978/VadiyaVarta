import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# ✅ Step 1: Load Processed Dataset
print("🔹 Loading processed dataset...")
dataset_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/balanced_dataset.csv"
df = pd.read_csv(dataset_path)

# ✅ Step 2: Load Label Encoder
label_encoder_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/label_encoder.joblib"
label_encoder = joblib.load(label_encoder_path)
num_labels = len(label_encoder.classes_)

# ✅ Step 3: Split Data into Test Set
test_texts = df["All_Symptoms"]
test_labels = df["label"]

# ✅ Step 4: Load BioBERT Tokenizer and Model
model_name = "monologg/biobert_v1.1_pubmed"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_path = "C:\Users\peter\Desktop\het\Vadiya Varta\Model-3\checkpoints\checkpoint-100000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

# ✅ Step 5: Tokenize Test Data
def tokenize_data(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

test_encodings = tokenize_data(list(test_texts))

# ✅ Step 6: Create Dataset for Evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"].to(device),
    "attention_mask": test_encodings["attention_mask"].to(device),
    "labels": torch.tensor(test_labels.values, dtype=torch.long).to(device)
})

# ✅ Step 7: Define Performance Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ✅ Step 8: Set Up Trainer for Evaluation
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./biobert_results",  # Specify directory for saving outputs
    per_device_eval_batch_size=16,   # Adjust based on GPU memory
    no_cuda=False,
    logging_dir='./logs',  # Directory for logs
    evaluation_strategy="epoch",  # Evaluate after every epoch
    fp16=True  # Mixed precision for faster evaluation on GPUs
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ✅ Step 9: Run Evaluation
print("🚀 Evaluating model...")
eval_results = trainer.evaluate()

# ✅ Step 10: Print Evaluation Results
print("🔹 Evaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1-Score: {eval_results['eval_f1']:.4f}")

# ✅ Step 11: Save the Evaluation Results to a File (optional)
with open("evaluation_results.txt", "w") as f:
    f.write(f"Accuracy: {eval_results['eval_accuracy']:.4f}\n")
    f.write(f"Precision: {eval_results['eval_precision']:.4f}\n")
    f.write(f"Recall: {eval_results['eval_recall']:.4f}\n")
    f.write(f"F1-Score: {eval_results['eval_f1']:.4f}\n")

print("✅ Evaluation completed and saved to `evaluation_results.txt`.")






















# import torch
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# import numpy as np

# # ✅ Step 1: Load Processed Dataset
# print("🔹 Loading processed dataset for evaluation...")
# dataset_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/balanced_dataset.csv"
# df = pd.read_csv(dataset_path)

# # ✅ Step 2: Load Label Encoder
# label_encoder_path = r"C:\Users\peter\Desktop\het\Vadiya Varta\Model-3\label_encoder.joblib"
# label_encoder = joblib.load(label_encoder_path)
# num_labels = len(label_encoder.classes_)

# # ✅ Step 3: Split Data into Train & Test Sets (Only Using Test Data Here)
# _, test_texts, _, test_labels = train_test_split(
#     df["All_Symptoms"], df["label"], test_size=0.2, random_state=42
# )

# # ✅ Step 4: Load BioBERT Tokenizer
# model_name = "monologg/biobert_v1.1_pubmed"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # ✅ Step 5: Tokenize Test Texts
# def tokenize_data(texts):
#     return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

# print("🔹 Tokenizing test data...")
# test_encodings = tokenize_data(list(test_texts))

# # ✅ Step 6: Move Data to CUDA (If Available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🚀 Using device: {device}")

# test_dataset = Dataset.from_dict({
#     "input_ids": test_encodings["input_ids"],
#     "attention_mask": test_encodings["attention_mask"],
#     "labels": torch.tensor(test_labels.values, dtype=torch.long),
# })

# # ✅ Step 7: Load **Saved Checkpoint**
# checkpoint_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/checkpoints/checkpoint-100000"
# print(f"🔹 Loading checkpoint from {checkpoint_path}...")

# model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_labels).to(device)

# # ✅ Step 8: Define Performance Metrics
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     acc = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=1)
#     return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# # ✅ Step 9: Set Up Trainer for Evaluation
# trainer = Trainer(
#     model=model,
#     eval_dataset=test_dataset,
#     compute_metrics=compute_metrics,
# )

# # ✅ Step 10: Run Evaluation
# print("🚀 Evaluating model performance...")
# metrics = trainer.evaluate()
# print("✅ Model Evaluation Complete!")
# print(metrics)

# # ✅ Step 11: Generate Graphs for Visualization
# def plot_metrics(metrics):
#     categories = ['Accuracy', 'Precision', 'Recall', 'F1-score']
#     values = [metrics['eval_accuracy'], metrics['eval_precision'], metrics['eval_recall'], metrics['eval_f1']]

#     plt.figure(figsize=(8, 5))
#     plt.bar(categories, values, color=['blue', 'green', 'red', 'purple'])
#     plt.ylim(0, 1)
#     plt.ylabel("Score")
#     plt.title("Model Performance Metrics")
#     for i, v in enumerate(values):
#         plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
#     plt.show()

# # ✅ Step 12: Plot Metrics
# plot_metrics(metrics)
