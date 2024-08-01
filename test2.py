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

# 3. Downsampling the Majority Class (Manually)
# Combine the features and target into a single DataFrame for resampling
train_data = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
majority = train_data[train_data['target'] == 0]
minority = train_data[train_data['target'] == 1]

# Downsample majority class
majority_downsampled = resample(majority,
                                replace=False,  # Sample without replacement
                                n_samples=len(minority),  # Match minority class size
                                random_state=42)  # For reproducibility

# Combine minority class with downsampled majority class
downsampled = pd.concat([majority_downsampled, minority])

# Separate features and target again
X_train_resampled = downsampled.drop(columns=['target'])
y_train_resampled = downsampled['target']

# Define the pipeline
pipeline = Pipeline(steps=[
    ('scaler', scaler),
    ('classifier', XGBClassifier(eval_metric='logloss'))
])

# 4. Apply XGBoost
pipeline.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of the positive class
y_pred = pipeline.predict(X_test)

# 5. Produce Metrics
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'AUC-ROC: {auc_roc:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Log Loss: {logloss:.4f}')
