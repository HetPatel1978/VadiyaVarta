import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Step 1: Load and Prepare Data
data_path = r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/dataset.csv"
df = pd.read_csv(data_path)

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["All_Symptoms"].tolist(),
    df["Disease"].astype("category").cat.codes.tolist(),  # Encode labels
    test_size=0.2,
    random_state=42
)

# Step 2: Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True)

# Tokenize the text data using the tokenizer
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Step 3: Convert to Hugging Face Datasets format
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

# Step 4: Load Pretrained Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(set(train_labels))
)

# Step 5: Set Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,  # Save the best model based on evaluation
    metric_for_best_model="accuracy",  # Use accuracy for model selection
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=None,  # Optional, if needed, use data_collator for batching
)

# Step 6: Train and Evaluate
trainer.train()

# Save the Model
model.save_pretrained("./trained_disease_model")
tokenizer.save_pretrained("./trained_disease_model")
