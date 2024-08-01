import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
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

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Apply resampling techniques to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the XGBoost model
import xgboost as xgb

# Initialize the XGBoost model with class weights
model = xgb.XGBClassifier(scale_pos_weight=len(negative_class) / len(positive_class),
                          eval_metric='logloss',
                          use_label_encoder=False,
                          random_state=42)
model.fit(X_train, y_train)


# Evaluate on validation set
y_val_pred = model.predict(X_val)
print("Validation Performance:")
print(classification_report(y_val, y_val_pred))

# Evaluate on test set
y_test_pred = model.predict(X_test)
print("Test Performance:")
print(classification_report(y_test, y_test_pred))

