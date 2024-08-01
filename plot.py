import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss
import xgboost as xgb

# Example DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'target': [0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  # Imbalanced target
}

df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Compute Sample Weights
class_weights = {0: 1, 1: 5}
y_train_weights = y_train.map(class_weights)

# Define XGBoost model
model = xgb.XGBClassifier(
    eval_metric='logloss',  # Log loss for evaluation
    use_label_encoder=False,
    alpha=1,
    lambda_=1,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    learning_rate=0.05
)

# Train model
model.fit(X_train, y_train, sample_weight=y_train_weights)

# Predictions
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
y_test_pred = model.predict(X_test)

# Metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc_roc = auc(*roc_curve(y_test, y_test_pred_proba)[:2])
test_f1 = f1_score(y_test, y_test_pred)
test_logloss = log_loss(y_test, y_test_pred_proba)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_test_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

plt.figure(figsize=(8, 6))
disp.plot(ax=plt.gca(), cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print metrics
print("Testing Metrics:")
print(f'Accuracy: {test_accuracy:.4f}')
print(f'AUC-ROC: {test_auc_roc:.4f}')
print(f'F1 Score: {test_f1:.4f}')
print(f'Log Loss: {test_logloss:.4f}')
