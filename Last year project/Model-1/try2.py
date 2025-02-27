import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import accuracy_score
from datasets import Dataset

# Load the trained model and tokenizer
model_path = "./trained_disease_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load your dataset (you can use the validation dataset or new data)
data_path = r"C:\Users\hetpa\OneDrive\Desktop\AIML\Last year project\dataset\cleaned_dataset.csv"
df = pd.read_csv(data_path)

# Split dataset (for example, use validation set)
val_texts = df["All_Symptoms"].tolist()  # or use your val dataset
val_labels = df["Disease"].astype("category").cat.codes.tolist()

# Tokenize the validation data
def tokenize_input(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

val_encodings = tokenize_input(val_texts)

# Prepare dataset in Hugging Face format
val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

# Evaluate the model
model.eval()  # Set model to evaluation mode

predictions = []
true_labels = []

# Iterate through the validation dataset and make predictions
with torch.no_grad():
    for i in range(len(val_dataset)):
        input_ids = torch.tensor(val_dataset[i]["input_ids"]).unsqueeze(0)  # Convert to tensor and add batch dimension
        attention_mask = torch.tensor(val_dataset[i]["attention_mask"]).unsqueeze(0)
        labels = val_dataset[i]["labels"]
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        
        # Store predictions and true labels
        predictions.append(predicted_class)
        true_labels.append(labels)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Accuracy on validation set: {accuracy * 100:.2f}%")
