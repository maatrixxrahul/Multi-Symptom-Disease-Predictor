"""
data_preprocessing.py
Data loading (chunked, for large files up to 3GB), symptom encoding, and
synthetic data generation for the Multi-Symptom Disease Predictor project.
"""

import numpy as np
import pandas as pd


DEFAULT_SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "sore_throat", "shortness_of_breath",
    "nausea", "vomiting", "diarrhea", "muscle_pain", "chills", "loss_of_appetite",
    "rash", "joint_pain", "chest_pain", "dizziness", "runny_nose", "loss_of_smell",
]

DEFAULT_DISEASES = [
    "Common Cold", "Influenza", "COVID-19", "Migraine", "Gastroenteritis",
    "Pneumonia", "Allergic Reaction", "Dengue Fever",
]


def load_large_csv(file_obj, chunksize=200_000, progress_callback=None):
    """
    Reads a (potentially very large, up to ~3GB) CSV file in chunks to avoid
    spiking memory usage, then concatenates the chunks into a single DataFrame.

    Parameters:
        file_obj: path or file-like object (e.g. Streamlit's UploadedFile)
        chunksize (int): number of rows to read per chunk
        progress_callback (callable, optional): called with (rows_read_so_far)
            after each chunk, useful for driving a progress bar in the UI

    Returns:
        pd.DataFrame: the full concatenated dataframe
    """
    chunks = []
    rows_read = 0

    for chunk in pd.read_csv(file_obj, chunksize=chunksize):
        chunks.append(chunk)
        rows_read += len(chunk)
        if progress_callback is not None:
            progress_callback(rows_read)

    df = pd.concat(chunks, ignore_index=True)
    return df


def generate_synthetic_data(n_samples=3000, symptoms=None, diseases=None, seed=42):
    """
    Generates a synthetic symptom-disease dataset for demo/testing purposes,
    where each disease has a characteristic (but noisy) symptom pattern.

    Parameters:
        n_samples (int): number of records to generate
        symptoms (list[str], optional): symptom column names
        diseases (list[str], optional): disease label options
        seed (int): random seed for reproducibility

    Returns:
        pd.DataFrame: binary symptom columns + a 'disease' label column
    """
    rng = np.random.default_rng(seed)
    symptoms = symptoms or DEFAULT_SYMPTOMS
    diseases = diseases or DEFAULT_DISEASES

    # each disease gets a random "core" set of symptoms it's likely to trigger
    disease_profiles = {
        disease: rng.choice(symptoms, size=rng.integers(3, 6), replace=False)
        for disease in diseases
    }

    records = []
    for _ in range(n_samples):
        disease = rng.choice(diseases)
        core_symptoms = disease_profiles[disease]

        row = {}
        for symptom in symptoms:
            if symptom in core_symptoms:
                # core symptoms appear most of the time
                row[symptom] = int(rng.random() < 0.85)
            else:
                # non-core symptoms appear rarely (noise)
                row[symptom] = int(rng.random() < 0.08)

        row["disease"] = disease
        records.append(row)

    return pd.DataFrame(records)


def encode_symptoms(df, symptom_cols, label_col="disease"):
    """
    Splits a dataframe into feature matrix (X) and label vector (y).

    Parameters:
        df (pd.DataFrame): dataframe with binary symptom columns and a label column
        symptom_cols (list[str]): names of the symptom feature columns
        label_col (str): name of the disease label column

    Returns:
        X (pd.DataFrame): binary symptom feature matrix
        y (pd.Series): disease labels
    """
    X = df[symptom_cols].fillna(0).astype(int)
    y = df[label_col]
    return X, y


def symptoms_to_vector(selected_symptoms, all_symptoms):
    """
    Converts a list of selected symptoms (e.g. from checkboxes) into a
    single-row binary feature vector matching the model's expected input.

    Parameters:
        selected_symptoms (list[str]): symptoms the patient is experiencing
        all_symptoms (list[str]): full ordered list of symptom columns used in training

    Returns:
        pd.DataFrame: single-row dataframe of 0/1 values, one column per symptom
    """
    vector = {s: int(s in selected_symptoms) for s in all_symptoms}
    return pd.DataFrame([vector])