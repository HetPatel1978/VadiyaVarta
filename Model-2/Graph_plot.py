# import torch
# import pandas as pd
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from tqdm import tqdm

# # Load the label encoder and dataset
# label_encoder = joblib.load(r"C:\Users\peter\Desktop\het\Vadiya Varta\Model-3\label_encoder.joblib")
# df = pd.read_csv(r"C:\Users\peter\Desktop\het\Vadiya Varta\Model-3\balanced_dataset.csv")

# # Select a sample of the dataset for evaluation
# df_sample = df.sample(n=500, random_state=42).reset_index(drop=True)

# # Load tokenizer and model from checkpoint
# model_path = r"C:\Users\peter\Desktop\het\Vadiya Varta\Model-2\biobert_disease_model"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# # Predict
# preds = []
# true = []

# for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
#     inputs = tokenizer(row["All_Symptoms"], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         pred = torch.argmax(outputs.logits, dim=1).cpu().item()
#     preds.append(pred)
#     true.append(row["label"])

# # Decode to disease names
# true_names = label_encoder.inverse_transform(true)
# pred_names = label_encoder.inverse_transform(preds)

# # Classification report
# print("\n📊 Classification Report:")
# print(classification_report(true_names, pred_names, zero_division=0))

# # Confusion matrix (Top 10 diseases only for readability)
# top_diseases = pd.Series(true_names).value_counts().head(10).index
# mask = [d in top_diseases for d in true_names]

# conf_matrix = confusion_matrix(
#     pd.Series(true_names)[mask],
#     pd.Series(pred_names)[mask],
#     labels=top_diseases
# )

# # Plot confusion matrix
# plt.figure(figsize=(12, 8))
# sns.heatmap(conf_matrix, xticklabels=top_diseases, yticklabels=top_diseases, annot=True, fmt="d", cmap="Blues")
# plt.title("🔍 Confusion Matrix - Top 10 Diseases")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.tight_layout()
# plt.show()









#Axis	    Meaning
# X-axis	Precision – how many of the predicted diseases were actually correct.
# Y-axis	Recall – how many actual cases the model correctly identified.
# Z-axis	F1 Score – harmonic mean of precision and recall (overall performance).


# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D

# # Simulated performance metrics for top 10 diseases (replace with actual data if available)
# diseases = ['common cold', 'flu', 'anemia', 'pneumonia', 'depression', 'migraine', 
#             'muscle spasm', 'kidney failure', 'sialoadenitis', 'hirsutism']
# precision = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# recall = [0.98, 0.95, 1.0, 0.82, 0.59, 1.0, 1.0, 1.0, 1.0, 1.0]
# f1_score = [0.99, 0.98, 1.0, 0.90, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# x = precision
# y = recall
# z = f1_score

# ax.scatter(x, y, z, c='blue', s=100)

# # Annotate each point
# for i in range(len(diseases)):
#     ax.text(x[i], y[i], z[i], diseases[i], size=10, zorder=1)

# # Set axis labels
# ax.set_xlabel('Precision')
# ax.set_ylabel('Recall')
# ax.set_zlabel('F1 Score')
# ax.set_title('🧠 Model Performance Metrics (3D View)')

# plt.tight_layout()
# plt.show()












import torch
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load data
df = pd.read_csv("C:\Users\peter\Desktop\het\Vadiya Varta\Model-2\processed_dataset.csv").sample(n=300, random_state=42)
texts = df["All_Symptoms"].tolist()
labels = df["label"].tolist()

# Load tokenizer and model
model_path = "C:\Users\peter\Desktop\het\Vadiya Varta\Model-2\biobert_disease_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Extract embeddings
cls_embeddings = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model.bert(**inputs)
        cls = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        cls_embeddings.append(cls)

# Run t-SNE
print("Running t-SNE...")
tsne_3d = TSNE(n_components=3, random_state=42).fit_transform(cls_embeddings)

# Create DataFrame for plotting
vis_df = pd.DataFrame(tsne_3d, columns=["x", "y", "z"])
vis_df["Disease"] = labels

# Plot
fig = px.scatter_3d(
    vis_df, x="x", y="y", z="z", color="Disease",
    title="3D Visualization of Predicted Disease Embeddings",
    width=1000, height=800
)
fig.show()
