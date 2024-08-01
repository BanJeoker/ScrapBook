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
