import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load IMDB
dataset = load_dataset("stanfordnlp/imdb")

RANDOM_STATE = 42

train_df = dataset["train"].to_pandas()
test_df = dataset["test"].to_pandas()

# smaller subset first
train_df_small = train_df.sample(n=4000, random_state=RANDOM_STATE)
test_df_small = test_df.sample(n=2000, random_state=RANDOM_STATE)

X_train_full = train_df_small["text"].values
y_train_full = train_df_small["label"].values

X_test = test_df_small["text"].values
y_test = test_df_small["label"].values

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

vectorizer = TfidfVectorizer(max_features=2000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

print("TF-IDF train shape:", X_train_tfidf.shape)

k_values = [3, 5, 11]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(X_train_tfidf, y_train)
    y_val_pred = knn.predict(X_val_tfidf)

    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    print(f"k={k} | val_acc={acc:.4f} | val_f1={f1:.4f}")