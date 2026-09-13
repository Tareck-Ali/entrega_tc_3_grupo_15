from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import settings


def train():
    df = pd.read_csv(settings.DATA)

    X = df["medical_abstract"].fillna("")
    y = df["condition_label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20_000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    #print(classification_report(y_val, predictions))

    Path(settings.MODELS).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, settings.BASELINE_MODEL)

    #print(f"Model saved to {settings.BASELINE_MODEL}")


if __name__ == "__main__":
    train()