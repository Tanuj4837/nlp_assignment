import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# Step 1: Load data directly from the extracted folder
with open('rt-polaritydata/rt-polarity.pos', 'r', encoding='ISO-8859-1') as pos_file:
    pos_reviews = pos_file.readlines()

with open('rt-polaritydata/rt-polarity.neg', 'r', encoding='ISO-8859-1') as neg_file:
    neg_reviews = neg_file.readlines()

# Step 2: Split the data into training, validation, and test sets
train_pos, val_pos, test_pos = pos_reviews[:4000], pos_reviews[4000:4500], pos_reviews[4500:]
train_neg, val_neg, test_neg = neg_reviews[:4000], neg_reviews[4000:4500], neg_reviews[4500:]

X_train = train_pos + train_neg
y_train = [1] * 4000 + [0] * 4000
X_val = val_pos + val_neg
y_val = [1] * 500 + [0] * 500
X_test = test_pos + test_neg
y_test = [1] * 831 + [0] * 831

# Step 3: Vectorize the text data (converting text to numeric features)
vectorizer = CountVectorizer(binary=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 5: Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Step 6: Calculate the confusion matrix and performance metrics
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Step 7: Output the confusion matrix and metrics
print("Confusion Matrix:")
print(conf_matrix)

print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 8: Create a table of results for the slide
data = {'Metric': ['TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score'],
        'Value': [tp, tn, fp, fn, precision, recall, f1]}

df = pd.DataFrame(data)
print("\nTable for Slide:")
print(df)

# Optional: Save the table as a CSV (for use in your report or presentation)
df.to_csv('evaluation_metrics.csv', index=False)
