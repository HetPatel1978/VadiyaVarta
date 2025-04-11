import torch
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Step 1: Load Processed Dataset
print("🔹 Loading processed dataset...")
dataset_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/processed_dataset.csv"
df = pd.read_csv(dataset_path)

# ✅ Step 2: Load Label Encoder
label_encoder_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/label_encoder.joblib"
label_encoder = joblib.load(label_encoder_path)
num_labels = len(label_encoder.classes_)

# ✅ Step 3: Split Data into Train & Test Sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["All_Symptoms"], df["label"], test_size=0.2, random_state=42
)

# ✅ Step 4: Load **Updated** BioBERT Tokenizer
model_name = "monologg/biobert_v1.1_pubmed"  # ✅ Using the correct BioBERT version
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Step 5: Tokenize Text (Ensuring GPU Support)
def tokenize_data(texts):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_data(list(train_texts))
test_encodings = tokenize_data(list(test_texts))

# ✅ Step 6: Move Data to CUDA (If Available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"].to(device),
    "attention_mask": train_encodings["attention_mask"].to(device),
    "labels": torch.tensor(train_labels.values, dtype=torch.long).to(device),  # ✅ FIXED: Proper Tensor Conversion
})

test_dataset = Dataset.from_dict({
    "input_ids": test_encodings["input_ids"].to(device),
    "attention_mask": test_encodings["attention_mask"].to(device),
    "labels": torch.tensor(test_labels.values, dtype=torch.long).to(device),  # ✅ FIXED: Proper Tensor Conversion
})

# ✅ Step 7: Load **Updated** BioBERT Model for Classification
print("🔹 Loading BioBERT model for fine-tuning...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

# ✅ Step 8: Define Performance Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ✅ Step 9: Set Up Trainer for Fine-Tuning (With Logging)
training_args = TrainingArguments(
    output_dir="./biobert_results",
    evaluation_strategy="epoch",  # ✅ Evaluate at the end of every epoch
    save_strategy="epoch",
    logging_dir="./logs",  # ✅ Store logs here
    logging_steps=500,  # ✅ Log loss every 500 steps (Less overhead)
    per_device_train_batch_size=16,  # ✅ Increased batch size for better GPU utilization
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # ✅ Reduced epochs for faster testing
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=3,  # ✅ Keep only the best 3 models
    report_to=["tensorboard"],  # ✅ Enable TensorBoard logging
    fp16=True,  # ✅ Use Mixed Precision Training for Faster Speed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # ✅ Stop training if no improvement for 2 epochs
)

# ✅ Step 10: Train BioBERT
print("🚀 Fine-tuning BioBERT on symptom descriptions...")
trainer.train()

# ✅ Step 11: Save Fine-Tuned Model
model.save_pretrained("biobert_disease_model")
tokenizer.save_pretrained("biobert_disease_model")

print("✅ Training complete! Model saved to `biobert_disease_model`.")
