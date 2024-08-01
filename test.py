import pandas as pd

# Example DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [1, 2, 1, 2, 1],
    'D': [2, 4, 6, 8, 10]
}
df = pd.DataFrame(data)

# Calculate correlation matrix
corr_matrix = df.corr().abs()

# Set the threshold for high correlation
threshold = 0.8

# List to keep track of features to drop
to_drop = []

# Loop through each feature
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        # Compare each pair of features
        if corr_matrix.iloc[i, j] > threshold:
            colname = corr_matrix.columns[i]
            if colname not in to_drop:
                to_drop.append(colname)

# Drop the features
df_reduced = df.drop(columns=to_drop)

print("Original DataFrame:")
print(df)
print("\nFeatures to Drop:")
print(to_drop)
print("\nReduced DataFrame:")
print(df_reduced)
