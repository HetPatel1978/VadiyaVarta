import pandas as pd
import joblib
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# ✅ Step 1: Load Dataset
print("🔹 Loading dataset...")
df = pd.read_csv(r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/dataset/dataset2.csv")  # Change this to the correct file path

# ✅ Step 2: Standardize Column Names
df.columns = df.columns.str.lower().str.strip()  # Ensure uniform column naming
symptom_cols = df.columns[1:]  # All columns except 'diseases' are symptoms

# ✅ Step 3: Define Multiple Symptom Sentence Formats
symptom_templates = [
    "I have been feeling {}.",
    "I am suffering from {}.",
    "I have been experiencing {} lately.",
    "Lately, I’ve noticed {}.",
    "For the past few days, I’ve had {}.",
    "I’ve been dealing with {}.",
    "I am struggling with {}.",
    "Symptoms I’m facing include {}.",
    "I feel {}.",
    "I have symptoms like {}.",
    "Recently, I started having {}.",
    "I'm having trouble with {}.",
]

# ✅ Step 4: Convert Symptoms into a Natural Language Sentence
print("🔹 Converting symptoms into a readable text format...")

def format_symptoms(row):
    symptoms = [symptom.replace("_", " ") for symptom in symptom_cols if row[symptom] == 1]
    if not symptoms:
        return "No symptoms provided"
    template = random.choice(symptom_templates)  # Randomly select a sentence format
    return template.format(", ".join(symptoms))

df["All_Symptoms"] = df.apply(format_symptoms, axis=1)

# ✅ Step 5: Remove Unnecessary Columns
df = df[["diseases", "All_Symptoms"]]

# ✅ Step 6: Remove Highly Rare Diseases (appearing fewer than 10 times)
disease_counts = df["diseases"].value_counts()
df = df[df["diseases"].isin(disease_counts[disease_counts >= 10].index)]

print(f"🔹 Removed rare diseases. Remaining unique diseases: {df['diseases'].nunique()}")

# ✅ Step 7: Convert Disease Labels to Numbers
print("🔹 Encoding disease labels...")
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["diseases"])

# ✅ Save Label Encoder for Future Use
joblib.dump(label_encoder, "label_encoder.joblib")

# ✅ Step 8: Handle Missing & Duplicate Data
print("🔹 Removing duplicate and empty symptom entries...")
df.drop_duplicates(subset=["All_Symptoms"], inplace=True)
df = df[df["All_Symptoms"].str.strip() != ""]

# ✅ Step 9: Balance the Dataset (Oversampling)
print("🔹 Balancing dataset using oversampling...")

# Find majority & minority disease classes
disease_counts = df["diseases"].value_counts()
max_samples = disease_counts.max()  # Number of samples in the largest class

# Oversample all minority classes to match the majority class
balanced_df_list = []
for disease in disease_counts.index:
    disease_df = df[df["diseases"] == disease]
    disease_df_upsampled = resample(disease_df, 
                                    replace=True,  
                                    n_samples=max_samples, 
                                    random_state=42)
    balanced_df_list.append(disease_df_upsampled)

# Combine all balanced disease samples
df_balanced = pd.concat(balanced_df_list)

print(f"🔹 Final dataset shape: {df_balanced.shape}")

# ✅ Step 10: Save Processed Dataset
df_balanced.to_csv(r"C:/Users/hetpa/OneDrive/Desktop/AIML/Last year project/Model-2/processed_dataset.csv", index=False)
print("✅ Preprocessing complete! Saved as `processed_dataset.csv`")
