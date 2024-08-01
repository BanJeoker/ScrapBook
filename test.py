import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from sklearn.metrics import classification_report

# Example DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'target': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=y_train.unique(), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Train the XGBoost model with sample weights
model = xgb.XGBClassifier(scale_pos_weight=class_weight_dict[1],  # Adjust for imbalanced classes
                          eval_metric='logloss',
                          use_label_encoder=False,
                          random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate on validation set
y_val_pred = model.predict(X_val_scaled)
print("Validation Performance:")
print(classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = model.predict(X_test_scaled)
print("Test Performance:")
print(classification_report(y_test, y_test_pred))


# sampling
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import xgboost as xgb
from sklearn.metrics import classification_report

# Example DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
    'target': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Combine training data
X_train_combined = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_combined['target'] = y_train.values

# Separate majority and minority classes
majority_class = X_train_combined[X_train_combined['target'] == 0]
minority_class = X_train_combined[X_train_combined['target'] == 1]

# Apply random oversampling
minority_oversampled = resample(minority_class,
                                replace=True,  # Sample with replacement
                                n_samples=len(majority_class),  # Match majority class
                                random_state=42)  # For reproducibility

# Combine majority class with oversampled minority class
X_train_oversampled = pd.concat([majority_class, minority_oversampled])

# Separate features and target
X_train_resampled = X_train_oversampled.drop(columns=['target'])
y_train_resampled = X_train_oversampled['target']

# Train the XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate on validation set
y_val_pred = model.predict(X_val_scaled)
print("Validation Performance:")
print(classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = model.predict(X_test_scaled)
print("Test Performance:")
print(classification_report(y_test, y_test_pred))





# with parameter tuning
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import xgboost as xgb
from sklearn.metrics import classification_report

# Example DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
    'target': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Combine training data
X_train_combined = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_combined['target'] = y_train.values

# Separate majority and minority classes
majority_class = X_train_combined[X_train_combined['target'] == 0]
minority_class = X_train_combined[X_train_combined['target'] == 1]

# Apply random oversampling
minority_oversampled = resample(minority_class,
                                replace=True,  # Sample with replacement
                                n_samples=len(majority_class),  # Match majority class
                                random_state=42)  # For reproducibility

# Combine majority class with oversampled minority class
X_train_oversampled = pd.concat([majority_class, minority_oversampled])

# Separate features and target
X_train_resampled = X_train_oversampled.drop(columns=['target'])
y_train_resampled = X_train_oversampled['target']

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='accuracy',  # or another appropriate metric
                           cv=3,  # Number of cross-validation folds
                           n_jobs=-1,  # Use all available cores
                           verbose=1)  # Print progress

# Fit GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate on validation set with the best model
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val_scaled)
print("Validation Performance:")
print(classification_report(y_val, y_val_pred))

# Evaluate on test set with the best model
y_test_pred = best_model.predict(X_test_scaled)
print("Test Performance:")
print(classification_report(y_test, y_test_pred))

-------------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import xgboost as xgb
from sklearn.metrics import classification_report

# Example DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
    'target': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Combine training data
X_train_combined = pd.DataFrame(X_train_scaled, columns=X.columns)
X_train_combined['target'] = y_train.values

# Separate majority and minority classes
majority_class = X_train_combined[X_train_combined['target'] == 0]
minority_class = X_train_combined[X_train_combined['target'] == 1]

# Apply random oversampling
minority_oversampled = resample(minority_class,
                                replace=True,  # Sample with replacement
                                n_samples=len(majority_class),  # Match majority class
                                random_state=42)  # For reproducibility

# Combine majority class with oversampled minority class
X_train_oversampled = pd.concat([majority_class, minority_oversampled])

# Separate features and target
X_train_resampled = X_train_oversampled.drop(columns=['target'])
y_train_resampled = X_train_oversampled['target']

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='accuracy',  # or another appropriate metric
                           cv=3,  # Number of cross-validation folds
                           n_jobs=-1,  # Use all available cores
                           verbose=1)  # Print progress

# Fit GridSearchCV
grid_search.fit(X_train_resampled, y_train_resampled)

# Best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate on test set with the best model
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test_scaled)
print("Test Performance:")
print(classification_report(y_test, y_test_pred))

########################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Example DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
    'target': [1, 0, 1, 0, 1, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)

# Fit model
model.fit(X_train_scaled, y_train)

# Predict probabilities
y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]  # Probabilities for the positive class
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]    # Probabilities for the positive class

# Predict classes
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Compute metrics
def compute_metrics(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred_proba)
    return accuracy, auc_roc, f1, logloss

# Metrics for training set
train_accuracy, train_auc_roc, train_f1, train_logloss = compute_metrics(y_train, y_train_pred, y_train_pred_proba)
print("Training Metrics:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"AUC-ROC: {train_auc_roc:.4f}")
print(f"F1 Score: {train_f1:.4f}")
print(f"Log Loss: {train_logloss:.4f}")

# Metrics for test set
test_accuracy, test_auc_roc, test_f1, test_logloss = compute_metrics(y_test, y_test_pred, y_test_pred_proba)
print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"AUC-ROC: {test_auc_roc:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"Log Loss: {test_logloss:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', title='Feature Importance', xlabel='Feature Importance', ylabel='Features')
plt.show()
