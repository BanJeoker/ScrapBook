import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

# Assume you have your model and data ready
# model = ... (your trained model)
# X_test = ... (your test features)
# y_test = ... (your test labels)

# Get predicted probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# Find the optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold}")

# Compute Precision-Recall curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_probs)

# Get predicted classes for the optimal threshold
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)

# Compute confusion matrix
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
print(f"Confusion Matrix for Optimal Threshold:\n{cm_optimal}")

# Plot ROC Curve and Precision-Recall Curve
plt.figure(figsize=(14, 8))

# ROC Curve
plt.subplot(2, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Precision-Recall Curve
plt.subplot(2, 2, 2)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

# Confusion Matrix
plt.subplot(2, 2, 3)
ConfusionMatrixDisplay(confusion_matrix=cm_optimal, display_labels=['Class 0', 'Class 1']).plot(ax=plt.gca(), cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix for Optimal Threshold')

plt.tight_layout()
plt.show()
