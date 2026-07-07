"""
model_training.py
------------------
Trains several classification algorithms on the OptiCrop dataset,
compares their performance, and serializes the champion model
(together with its scaler and label encoder) to model/model.pkl.

Run:
    python model_training.py
"""

import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from preprocessing import prepare_data, FEATURE_COLUMNS

MODEL_PATH = "model/model.pkl"
METRICS_PATH = "model/metrics.json"


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }


def train_and_evaluate(data):
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    results = {}
    fitted_models = {}

    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }
        fitted_models[name] = model

        print(f"{name:22s} | Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    return results, fitted_models


def select_champion(results, fitted_models):
    champion_name = max(results, key=lambda k: results[k]["f1_score"])
    print(f"\nChampion model: {champion_name} "
          f"(F1-Score = {results[champion_name]['f1_score']})")
    return champion_name, fitted_models[champion_name]


def main():
    print("Loading and preprocessing dataset...")
    data = prepare_data()

    print(f"Training on {data['X_train'].shape[0]} samples, "
          f"testing on {data['X_test'].shape[0]} samples, "
          f"{len(data['encoder'].classes_)} crop classes.\n")

    results, fitted_models = train_and_evaluate(data)
    champion_name, champion_model = select_champion(results, fitted_models)

    # Detailed report for the champion
    y_pred = champion_model.predict(data["X_test"])
    print("\nClassification report (champion model):")
    print(classification_report(
        data["y_test"], y_pred,
        target_names=data["encoder"].classes_,
        zero_division=0,
    ))

    # Bundle model + scaler + encoder + feature order together so the
    # Flask app only needs to load a single artifact.
    bundle = {
        "model": champion_model,
        "scaler": data["scaler"],
        "encoder": data["encoder"],
        "feature_columns": FEATURE_COLUMNS,
        "champion_name": champion_name,
        "metrics": results,
        "all_models": fitted_models,
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved champion bundle to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
