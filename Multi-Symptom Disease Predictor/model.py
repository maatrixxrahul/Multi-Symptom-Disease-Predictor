"""
model.py
Model training, evaluation, and prediction logic for the
Multi-Symptom Disease Predictor project.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """Splits features/labels into train and test sets (stratified by label)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_model(n_estimators=200, max_depth=None, random_state=42):
    """
    Builds a Random Forest multi-class classifier for disease prediction.

    Parameters:
        n_estimators (int): number of trees
        max_depth (int, optional): max tree depth
        random_state (int): random seed for reproducibility

    Returns:
        RandomForestClassifier: an unfitted model
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


def train_model(model, X_train, y_train):
    """Fits the model on training data and returns the fitted model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on a held-out test set.

    Returns:
        dict: {
            "accuracy": float,
            "report": str (classification report),
            "confusion_matrix": np.ndarray,
            "labels": list[str] (class order used in the confusion matrix)
        }
    """
    y_pred = model.predict(X_test)
    labels = sorted(y_test.unique())

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
    }


def predict_top_n(model, symptom_vector, top_n=3):
    """
    Predicts the top-N most likely diseases for a given symptom vector.

    Parameters:
        model: trained classifier with predict_proba support
        symptom_vector (pd.DataFrame): single-row binary feature vector
        top_n (int): number of top predictions to return

    Returns:
        list[tuple[str, float]]: (disease, probability) pairs, sorted descending
    """
    probs = model.predict_proba(symptom_vector)[0]
    classes = model.classes_

    ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def save_trained_model(model, filepath="disease_predictor_model.joblib"):
    """Saves the trained model to disk."""
    joblib.dump(model, filepath)


def load_trained_model(filepath="disease_predictor_model.joblib"):
    """Loads a previously trained model from disk."""
    return joblib.load(filepath)