from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import settings


def train():
    df = pd.read_csv(settings.TRAIN_DATA)

    X = df["medical_abstract"].fillna("")
    y = df["condition_label"]

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

    model.fit(X, y)

    Path(settings.MODELS).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, settings.BASELINE_MODEL)

    print(f"Model saved to {settings.BASELINE_MODEL}")


if __name__ == "__main__":
    train()
