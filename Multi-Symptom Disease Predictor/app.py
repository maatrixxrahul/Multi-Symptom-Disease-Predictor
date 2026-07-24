"""
app.py
Interactive Streamlit interface for the Multi-Symptom Disease Predictor project.
Supports training-data uploads up to 3 GB (see .streamlit/config.toml).
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_preprocessing import (
    load_large_csv,
    generate_synthetic_data,
    encode_symptoms,
    symptoms_to_vector,
    DEFAULT_SYMPTOMS,
)
from model import (
    train_test_split_data,
    build_model,
    train_model,
    evaluate_model,
    predict_top_n,
)

#  Page Config 
st.set_page_config(page_title="Multi-Symptom Disease Predictor", layout="wide")

st.title("🩺 Multi-Symptom Disease Predictor")
st.markdown(
    "Predict the most likely disease from reported symptoms. "
    "Supports training-data uploads of **up to 3 GB**."
)

#  Session State 
if "model" not in st.session_state:
    st.session_state.model = None
if "symptom_cols" not in st.session_state:
    st.session_state.symptom_cols = DEFAULT_SYMPTOMS

tab_train, tab_predict = st.tabs(["🧠 Train Model", "🔍 Predict"])


# TAB 1 — TRAIN MODEL
with tab_train:
    st.header("Train on Symptom–Disease Data")

    uploaded_file = st.file_uploader(
        "Upload CSV (binary symptom columns + a 'disease' label column, up to 3 GB)",
        type=["csv"],
    )
    use_demo_data = st.checkbox("Use synthetic demo data instead", value=uploaded_file is None)

    df = None

    if use_demo_data:
        n_samples = st.slider("Number of synthetic records", 500, 10_000, 3000, step=500)
        if st.button("Generate Synthetic Data"):
            df = generate_synthetic_data(n_samples=n_samples)
            st.session_state.raw_df = df

    elif uploaded_file is not None:
        st.info(
            f"File size: {uploaded_file.size / (1024 ** 2):.1f} MB. "
            "Reading in chunks to keep memory usage manageable..."
        )
        progress_bar = st.progress(0, text="Reading file...")
        rows_seen = {"count": 0}

        def _progress(rows_read):
            rows_seen["count"] = rows_read
            # progress bar is indicative (unknown total rows ahead of time)
            progress_bar.progress(min(rows_read / 1_000_000, 1.0),
                                    text=f"Rows read so far: {rows_read:,}")

        with st.spinner("Loading large CSV in chunks..."):
            df = load_large_csv(uploaded_file, chunksize=200_000, progress_callback=_progress)
        progress_bar.progress(1.0, text=f"Done — {len(df):,} rows loaded")
        st.session_state.raw_df = df

    if "raw_df" in st.session_state and st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.success(f"Loaded {len(df):,} records.")
        st.dataframe(df.head(10), use_container_width=True)

        symptom_cols = [c for c in df.columns if c != "disease"]
        st.session_state.symptom_cols = symptom_cols

        if st.button("Train Model", type="primary"):
            with st.spinner("Preparing data..."):
                X, y = encode_symptoms(df, symptom_cols, label_col="disease")
                X_train, X_test, y_train, y_test = train_test_split_data(X, y)

            with st.spinner("Training Random Forest model..."):
                model = build_model()
                model = train_model(model, X_train, y_train)
                st.session_state.model = model

            with st.spinner("Evaluating model..."):
                results = evaluate_model(model, X_test, y_test)

            st.metric("Test Accuracy", f"{results['accuracy'] * 100:.2f}%")

            with st.expander("Classification Report"):
                st.text(results["report"])

            with st.expander("Confusion Matrix"):
                cm_fig = px.imshow(
                    results["confusion_matrix"],
                    x=results["labels"], y=results["labels"],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    text_auto=True, color_continuous_scale="Blues",
                )
                st.plotly_chart(cm_fig, use_container_width=True)

            st.success("Model trained and ready — switch to the **Predict** tab.")


# TAB 2 — PREDICT
with tab_predict:
    st.header("Predict Disease from Symptoms")

    if st.session_state.model is None:
        st.warning("Train a model in the **Train Model** tab first.")
    else:
        symptom_cols = st.session_state.symptom_cols
        st.write("Select the symptoms the patient is experiencing:")

        cols = st.columns(3)
        selected_symptoms = []
        for i, symptom in enumerate(symptom_cols):
            with cols[i % 3]:
                if st.checkbox(symptom.replace("_", " ").title(), key=f"sym_{symptom}"):
                    selected_symptoms.append(symptom)

        if st.button("Predict", type="primary"):
            if not selected_symptoms:
                st.warning("Select at least one symptom.")
            else:
                vector = symptoms_to_vector(selected_symptoms, symptom_cols)
                top_predictions = predict_top_n(st.session_state.model, vector, top_n=3)

                st.subheader("Top Predicted Diseases")
                for disease, prob in top_predictions:
                    st.write(f"**{disease}** — {prob * 100:.1f}% confidence")
                    st.progress(float(prob))

st.markdown("---")
st.caption("Built by Rahul Sah · github.com/maatrixxrahul")