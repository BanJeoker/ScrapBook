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




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame
data = pd.DataFrame({
    'property_type': ['House', 'Apartment', 'House', 'Apartment', 'House', 'House', 'Apartment'],
    'rating': ['A', 'B', 'A', 'A', 'B', 'B', 'C']
})

# Count the occurrences of each property_type and rating combination
counts = data.groupby(['property_type', 'rating']).size().unstack(fill_value=0)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the grouped bar chart
counts.plot(kind='bar', ax=ax)

# Customize the plot
ax.set_xlabel('Property Type')
ax.set_ylabel('Count')
ax.set_title('Histogram of Property Types Separated by Rating')
ax.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame
data = pd.DataFrame({
    'rating': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'C'],
    'pre_rating': ['X', 'X', 'Y', 'X', 'Z', 'Y', 'Y', 'Z', 'X']
})

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Use seaborn's histplot to plot the histogram with different pre_ratings for each rating
sns.histplot(data=data, x='pre_rating', hue='rating', multiple='dodge', discrete=True)

# Customize the plot
plt.xlabel('Pre Rating')
plt.ylabel('Count')
plt.title('Distribution of Pre Ratings for Each Rating')
plt.legend(title='Rating')

# Show plot
plt.tight_layout()
plt.show()

