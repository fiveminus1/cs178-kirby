import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Load IMDB
dataset = load_dataset("stanfordnlp/imdb")

RANDOM_STATE = 42

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# smaller subset first
train_df = train_df.sample(n=20000, random_state=RANDOM_STATE)
test_df = test_df.sample(n=25000, random_state=RANDOM_STATE)

X_train_full = train_df["text"].values
y_train_full = train_df["label"].values

X_test = test_df["text"].values
y_test = test_df["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_train_full
)

print("Train size:", len(X_train))
print("Validation size:", len(X_val))
print("Test size:", len(X_test))

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("TF-IDF train shape:", X_train_tfidf.shape) # rows = number of training examples, columns = number of TF-IDF features

k_values = [3, 5, 11, 21, 31, 41, 51, 71, 91]
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(X_train_tfidf, y_train)
    y_val_pred = knn.predict(X_val_tfidf)

    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    results.append((k, acc, f1))

    print(f"k={k} | val_acc={acc:.4f} | val_f1={f1:.4f}")

best_k = max(results, key=lambda x: x[2])[0]  # choose best F1
print("Best k:", best_k)

vectorizer_final = TfidfVectorizer(max_features=5000, stop_words="english")

X_train_final_tfidf = vectorizer_final.fit_transform(X_train_full)

# Final kNN model
knn_final = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="cosine",
    algorithm="brute"
)
knn_final.fit(X_train_final_tfidf, y_train_full)

# Predict test set in batches
batch_size = 5000
y_test_pred_batches = []

for start in range(0, len(X_test), batch_size):
    end = min(start + batch_size, len(X_test))
    X_batch = X_test[start:end]

    X_batch_tfidf = vectorizer_final.transform(X_batch)
    y_batch_pred = knn_final.predict(X_batch_tfidf)

    y_test_pred_batches.append(y_batch_pred)
    print(f"Predicted test examples {start} to {end}")

y_test_pred = np.concatenate(y_test_pred_batches)

print("\nFinal Test Results")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall   :", recall_score(y_test, y_test_pred))
print("F1       :", f1_score(y_test, y_test_pred))