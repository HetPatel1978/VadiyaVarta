# evaluate.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# ✅ Create output directory
# -----------------------------

output_dir = "evaluation_results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# ✅ Load true & predicted labels
# -----------------------------

true_df = pd.read_csv(r"C:\Users\peter\Desktop\het\Vadiya Varta\true_labels.csv")
pred_df = pd.read_csv(r"C:\Users\peter\Desktop\het\Vadiya Varta\pred_labels.csv")

true_col = true_df.columns[0]
pred_col = pred_df.columns[0]

true_labels = true_df[true_col]
pred_labels = pred_df[pred_col]

# -----------------------------
# ✅ Compute Evaluation Metrics
# -----------------------------

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

# -----------------------------
# ✅ Save Metrics Summary
# -----------------------------

summary_path = os.path.join(output_dir, "evaluation_summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Evaluation Summary ===\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n")

print(f"✅ Saved summary to {summary_path}")

# -----------------------------
# ✅ Save Full Classification Report
# -----------------------------

report = classification_report(true_labels, pred_labels, zero_division=0)
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("=== Classification Report ===\n\n")
    f.write(report)

print(f"✅ Saved classification report to {report_path}")

# -----------------------------
# ✅ Plot & Save Full Confusion Matrix
# -----------------------------

conf_matrix = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, cmap="Blues", annot=False, fmt="d")
plt.title("Full Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
full_cm_path = os.path.join(output_dir, "confusion_matrix_full.png")
plt.savefig(full_cm_path)
plt.close()
print(f"✅ Saved full confusion matrix to {full_cm_path}")

# -----------------------------
# ✅ Optional: Plot & Save Top-10 Confusion Matrix
# -----------------------------

top10_labels = pd.Series(true_labels).value_counts().head(10).index.tolist()

mask_true = true_labels.isin(top10_labels)
mask_pred = pred_labels.isin(top10_labels)
mask = mask_true & mask_pred

top10_conf_matrix = confusion_matrix(true_labels[mask], pred_labels[mask], labels=top10_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(top10_conf_matrix, cmap="Blues", annot=True, fmt="d", xticklabels=top10_labels, yticklabels=top10_labels)
plt.title("Top-10 Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
top10_cm_path = os.path.join(output_dir, "confusion_matrix_top10.png")
plt.savefig(top10_cm_path)
plt.close()
print(f"✅ Saved Top-10 confusion matrix to {top10_cm_path}")

# -----------------------------
# ✅ Done
# -----------------------------

print("\n🎉 Evaluation complete! All results are saved in the 'evaluation_results' folder.")
