"""
Predictive Credit Risk Engine
-----------------------------
Trains and compares two classifiers (Logistic Regression and Decision Tree)
to predict credit risk, handling class imbalance with SMOTE.

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

Usage:
    python credit_risk_model.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

try:
    from imblearn.over_sampling import SMOTE
except ModuleNotFoundError:
    sys.exit(
        "Missing package 'imbalanced-learn'.\n"
        "Install it with: pip install imbalanced-learn"
    )

# Always look for the CSV next to this script file, no matter what folder
# the terminal happens to be pointed at when you run it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "credit_risk_dataset.csv")
TARGET_COL = "Credit_Risk"
RANDOM_STATE = 42


def load_data(path):
    """Load the dataset and print a quick summary so we know what we're working with."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(
            f"Could not find '{path}'. Make sure the CSV is in the same "
            "folder as this script, or update DATA_PATH at the top of the file."
        )

    print("First 5 rows:")
    print(df.head().to_string(index=False))

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns:", list(df.columns))

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("\nNo missing values found.")
    else:
        print("\nMissing values:")
        print(missing)

    return df


def resolve_target_column(df, target_col):
    """
    Match the target column name case-insensitively / ignoring underscores,
    so small naming differences (e.g. 'credit risk' vs 'Credit_Risk')
    don't crash the script.
    """
    if target_col in df.columns:
        return target_col

    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    key = target_col.lower().replace(" ", "").replace("_", "")

    if key in normalized:
        return normalized[key]

    sys.exit(
        f"Target column '{target_col}' not found. Available columns: {list(df.columns)}\n"
        "Update TARGET_COL at the top of the script to match your dataset."
    )


def encode_categoricals(df):
    """Label-encode any text/object columns so the models can use them."""
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include="object").columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

    return df, encoders


def split_and_balance(df, target_col, test_size=0.2, random_state=RANDOM_STATE):
    """Split into train/test, then balance the training set with SMOTE."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("\nClass distribution after SMOTE:")
    print(y_train_bal.value_counts())

    return X_train_bal, X_test, y_train_bal, y_test


def train_models(X_train, y_train):
    """Train Logistic Regression and Decision Tree models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def evaluate_model(name, model, X_test, y_test):
    """Print accuracy, F1, and ROC-AUC for a single model, and return them."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"ROC AUC  : {roc_auc:.3f}")

    return {
        "Model": name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "predictions": preds,
        "probabilities": probs,
    }


def plot_confusion_matrix(y_test, preds, title):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(y_test, model_results):
    plt.figure(figsize=(8, 6))

    for result in model_results:
        fpr, tpr, _ = roc_curve(y_test, result["probabilities"])
        plt.plot(fpr, tpr, label=f"{result['Model']} (AUC={result['ROC-AUC']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_decision_tree_importance(model, feature_names):
    """Feature importance based on the Decision Tree's split gains."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="bar", color="steelblue")
    plt.title("Feature Importance — Decision Tree")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_logistic_importance(model, feature_names):
    """
    Logistic Regression has no .feature_importances_, but the absolute size
    of its coefficients tells us which features push the prediction most.
    """
    coefs = pd.Series(model.coef_[0], index=feature_names)
    coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 6))
    colors = ["crimson" if c < 0 else "seagreen" for c in coefs]
    coefs.plot(kind="bar", color=colors)
    plt.title("Feature Importance — Logistic Regression (coefficient weight)")
    plt.xlabel("Features")
    plt.ylabel("Coefficient value")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()


def main():
    df = load_data(DATA_PATH)
    target_col = resolve_target_column(df, TARGET_COL)

    df, _ = encode_categoricals(df)

    X_train, X_test, y_train, y_test = split_and_balance(df, target_col)

    models = train_models(X_train, y_train)

    results = [
        evaluate_model(name, model, X_test, y_test)
        for name, model in models.items()
    ]

    # Visualizations
    dt_result = next(r for r in results if r["Model"] == "Decision Tree")
    plot_confusion_matrix(y_test, dt_result["predictions"], "Decision Tree Confusion Matrix")
    plot_roc_curves(y_test, results)
    plot_decision_tree_importance(models["Decision Tree"], X_train.columns)
    plot_logistic_importance(models["Logistic Regression"], X_train.columns)

    # Final comparison table
    summary = pd.DataFrame(
        [{k: v for k, v in r.items() if k in ("Model", "Accuracy", "F1 Score", "ROC-AUC")} for r in results]
    )
    print("\n========== Final Comparison ==========")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()