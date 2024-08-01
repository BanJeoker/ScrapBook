import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from sklearn.utils import resample
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# Assuming you have a DataFrame 'df' and 'target' is your target column

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# 1. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. Normalization
scaler = StandardScaler()

# 3. Upsampling the Minority Class (Manually)
# Combine the features and target into a single DataFrame for resampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority = train_data[train_data['target'] == 0]
minority = train_data[train_data['target'] == 1]

# Upsample minority class
minority_upsampled = resample(minority,
                              replace=True,  # Sample with replacement
                              n_samples=len(majority),  # Match majority class size
                              random_state=42)  # For reproducibility

# Combine upsampled minority class with majority class
upsampled = pd.concat([majority, minority_upsampled])

# Separate features and target again
X_train_upsampled = upsampled.drop(columns=['target'])
y_train_upsampled = upsampled['target']

# Define the pipeline
pipeline = Pipeline(steps=[
    ('scaler', scaler),
    ('classifier', XGBClassifier(eval_metric='logloss'))
])

# 4. Apply XGBoost
pipeline.fit(X_train_upsampled, y_train_upsampled)

# Predict on the training set
y_train_pred_proba = pipeline.predict_proba(X_train_upsampled)[:, 1]  # Probability of the positive class
y_train_pred = pipeline.predict(X_train_upsampled)

# Predict on the test set
y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of the positive class
y_test_pred = pipeline.predict(X_test)

# 5. Produce Metrics
# Training metrics
train_accuracy = accuracy_score(y_train_upsampled, y_train_pred)
train_auc_roc = roc_auc_score(y_train_upsampled, y_train_pred_proba)
train_f1 = f1_score(y_train_upsampled, y_train_pred)
train_logloss = log_loss(y_train_upsampled, y_train_pred_proba)

# Testing metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc_roc = roc_auc_score(y_test, y_test_pred_proba)
test_f1 = f1_score(y_test, y_test_pred)
test_logloss = log_loss(y_test, y_test_pred_proba)

# Print results
print("Training Metrics:")
print(f'Accuracy: {train_accuracy:.4f}')
print(f'AUC-ROC: {train_auc_roc:.4f}')
print(f'F1 Score: {train_f1:.4f}')
print(f'Log Loss: {train_logloss:.4f}')

print("\nTesting Metrics:")
print(f'Accuracy: {test_accuracy:.4f}')
print(f'AUC-ROC: {test_auc_roc:.4f}')
print(f'F1 Score: {test_f1:.4f}')
print(f'Log Loss: {test_logloss:.4f}')
