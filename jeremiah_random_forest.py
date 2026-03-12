import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

splits = {
  'train': 'plain_text/train-00000-of-00001.parquet', 
  'test': 'plain_text/test-00000-of-00001.parquet', 
  'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'
}

df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])

if __name__ == '__main__':
  X = df["text"]
  y = df["label"]

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

  vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

  X_train_vec = vectorizer.fit_transform(X_train)
  X_val_vec = vectorizer.transform(X_val)

  rf = RandomForestClassifier(n_estimators=100, random_state=42)

  rf.fit(X_train_vec, y_train)

  val_preds = rf.predict(X_val_vec)

  print("Validation Accuracy:", accuracy_score(y_val, val_preds))