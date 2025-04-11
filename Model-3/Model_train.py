# import torch
# import pandas as pd
# import joblib
# import torch.multiprocessing
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
# from datasets import Dataset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# # ✅ Fix for Multiprocessing Issues on Windows
# torch.multiprocessing.set_start_method("spawn", force=True)

# # ✅ Step 1: Load Processed Dataset (Batch Processing for Large Datasets)
# print("🔹 Loading processed dataset...")
# dataset_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/balanced_dataset.csv"
# df = pd.read_csv(dataset_path, nrows=500000)  # ✅ Prevents Memory Overflow

# # ✅ Step 2: Load Label Encoder
# label_encoder_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/label_encoder.joblib"
# label_encoder = joblib.load(label_encoder_path)
# num_labels = len(label_encoder.classes_)

# # ✅ Step 3: Split Data into Train & Test Sets
# train_texts, test_texts, train_labels, test_labels = train_test_split(
#     df["All_Symptoms"], df["label"], test_size=0.2, random_state=42
# )

# # ✅ Step 4: Load Tokenizer
# model_name = "monologg/biobert_v1.1_pubmed"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # ✅ Step 5: Tokenize Data in Batches (Prevents Memory Overload)
# def tokenize_batch(texts, batch_size=5000):
#     all_encodings = []
#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i : i + batch_size]
#         encodings = tokenizer(
#             batch_texts.tolist(), 
#             padding="max_length", 
#             truncation=True, 
#             max_length=128, 
#             return_tensors="pt"
#         )
#         all_encodings.append(encodings)
#     return all_encodings

# print("🔹 Tokenizing training data...")
# train_encodings = tokenize_batch(train_texts)
# print("🔹 Tokenizing test data...")
# test_encodings = tokenize_batch(test_texts)

# # ✅ Step 6: Move Data to GPU (If Available)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🚀 Using device: {device}")

# # ✅ Convert Data to Hugging Face Dataset Format
# def create_hf_dataset(encodings_list, labels):
#     input_ids, attention_masks = [], []
#     for encodings in encodings_list:
#         input_ids.append(encodings["input_ids"])
#         attention_masks.append(encodings["attention_mask"])
#     return Dataset.from_dict({
#         "input_ids": torch.cat(input_ids),
#         "attention_mask": torch.cat(attention_masks),
#         "labels": torch.tensor(labels.values, dtype=torch.long),
#     })

# train_dataset = create_hf_dataset(train_encodings, train_labels)
# test_dataset = create_hf_dataset(test_encodings, test_labels)

# # ✅ Step 7: Load BioBERT Model
# print("🔹 Loading BioBERT model for fine-tuning...")
# model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

# # ✅ Step 8: Define Performance Metrics
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     acc = accuracy_score(labels, preds)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=1)
#     return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# # ✅ Step 9: Set Training Arguments (Optimized for Speed & Stability)
# training_args = TrainingArguments(
#     output_dir="C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/checkpoints",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     greater_is_better=True,
    
#     per_device_train_batch_size=4,  # ✅ Reduced to prevent CUDA OOM
#     per_device_eval_batch_size=4,
#     num_train_epochs=3,
#     weight_decay=0.05,
#     save_total_limit=2,
#     report_to=["tensorboard"],

#     fp16=True,
#     optim="adamw_torch",
    
#     logging_steps=500,
#     dataloader_num_workers=0,  # ✅ Fix for Windows Multiprocessing
#     gradient_checkpointing=True,
# )

# # ✅ Step 10: Train the Model (Wrapped in `if __name__ == "__main__":`)
# if __name__ == "__main__":
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         compute_metrics=compute_metrics,
#         callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
#     )

#     print("🚀 Fine-tuning BioBERT on symptom descriptions...")
#     trainer.train()

#     # ✅ Step 11: Save Model
#     model_save_path = "C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/biobert_disease_model"
#     model.save_pretrained(model_save_path)
#     tokenizer.save_pretrained(model_save_path)

#     print(f"✅ Training complete! Model saved to `{model_save_path}`.")













#Resume Model training 

import torch
import pandas as pd
import joblib
import torch.multiprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ✅ Fix multiprocessing issue on Windows
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # ✅ Step 1: Load Processed Dataset
    print("🔹 Loading processed dataset...")
    dataset_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/balanced_dataset.csv"
    df = pd.read_csv(dataset_path)

    # ✅ Step 2: Load Label Encoder
    label_encoder_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/label_encoder.joblib"
    label_encoder = joblib.load(label_encoder_path)
    num_labels = len(label_encoder.classes_)

    # ✅ Step 3: Split Data into Train & Test Sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["All_Symptoms"], df["label"], test_size=0.2, random_state=42
    )

    # ✅ Step 4: Load BioBERT Tokenizer
    model_name = "monologg/biobert_v1.1_pubmed"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ✅ Step 5: Tokenize Text
    def tokenize_data(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    print("🔹 Tokenizing training data...")
    train_encodings = tokenize_data(list(train_texts))
    print("🔹 Tokenizing test data...")
    test_encodings = tokenize_data(list(test_texts))

    # ✅ Step 6: Move Data to CUDA (If Available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": torch.tensor(train_labels.values, dtype=torch.long),
    })

    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": torch.tensor(test_labels.values, dtype=torch.long),
    })

    # ✅ Step 7: Load **Saved Checkpoint**
    checkpoint_path = r"C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/checkpoints/checkpoint-100000"
    print(f"🔹 Loading checkpoint from {checkpoint_path}")

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_labels).to(device)

    # ✅ Step 8: Define Performance Metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=1)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # ✅ Step 9: Training Arguments (Optimized for Windows & CUDA)
    training_args = TrainingArguments(
        output_dir="C:/Users/peter/Desktop/het/Vadiya Varta/Model-3",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/logs",
        logging_steps=500,
        per_device_train_batch_size=8,  # ✅ Reduced to prevent OOM errors
        per_device_eval_batch_size=8,
        num_train_epochs=3,  # ✅ Resume remaining epochs
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,  # ✅ Keep only best 2 checkpoints
        report_to=["tensorboard"],
        fp16=True,  # ✅ Mixed precision for speed
        dataloader_num_workers=0,  # ✅ Prevent multiprocessing issues on Windows
        resume_from_checkpoint=checkpoint_path,  # ✅ Resume training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],  
    )

    # ✅ Step 10: Resume Training
    print("🚀 Resuming fine-tuning from checkpoint...")
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # ✅ Step 11: Save Final Model
    model.save_pretrained("C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/biobert_disease_model")
    tokenizer.save_pretrained("C:/Users/peter/Desktop/het/Vadiya Varta/Model-3/biobert_disease_model")

    print("✅ Training resumed and completed! Model saved.")
