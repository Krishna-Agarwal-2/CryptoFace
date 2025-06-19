import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score

CSV_PATH = "data/similarity_scores1.csv"

# Load similarity results
df = pd.read_csv(CSV_PATH)
df["label"] = df["match"].map({"yes": 1, "no": 0})
scores = df["similarity"].values
labels = df["label"].values

# Sweep thresholds from 0 to 1
thresholds = np.linspace(0, 1, 1000)
accuracies = []

for thresh in thresholds:
    preds = (scores >= thresh).astype(int)
    acc = accuracy_score(labels, preds)
    accuracies.append(acc)

# Find best threshold
best_idx = np.argmax(accuracies)
best_threshold = thresholds[best_idx]
best_accuracy = accuracies[best_idx]

# ROC Curve
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Final metrics at best threshold
final_preds = (scores >= best_threshold).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(labels, final_preds, average='binary')
acc = accuracy_score(labels, final_preds)

print(f"🎯 Best threshold: {best_threshold:.4f} | Accuracy: {acc:.4f}")
print(f"📈 Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
print(f"🏁 ROC AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/roc_curve1.png")
plt.show()
