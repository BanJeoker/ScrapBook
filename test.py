import pandas as pd
from sklearn.model_selection import train_test_split

# Example DataFrame
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [5, 4, 3, 2, 1],
    'target': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df.drop(columns=['target'])
y = df['target']

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training features:")
print(X_train)
print("\nTest features:")
print(X_test)
print("\nTraining target:")
print(y_train)
print("\nTest target:")
print(y_test)
