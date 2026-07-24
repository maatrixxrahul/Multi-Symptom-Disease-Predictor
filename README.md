# Multi-Symptom Disease Predictor

A machine learning tool that predicts the most likely disease from a set of reported symptoms, trained on a multi-class symptom–disease dataset — supporting large training-data uploads of **up to 3 GB**.

---

## 📖 Overview

Given a patient's reported symptoms (fever, cough, fatigue, headache, etc.), this project predicts the most probable underlying disease using a multi-class classification model. It's designed to work with large real-world symptom datasets, supporting CSV uploads up to **3 GB** by raising Streamlit's default upload limit and reading large files in memory-safe chunks.

---

## ✨ Features

- Multi-symptom → multi-disease classification (~98% accuracy on the reference dataset)
- Large file upload support — **up to 3 GB** CSV training data
- Chunked CSV reading to avoid loading the entire file into memory at once
- Interactive symptom checklist for live predictions
- Top-N most likely diseases with confidence scores
- Model evaluation: accuracy, classification report, confusion matrix
- Upload progress feedback for large files

---

## 🧠 Method

1. **Data Preparation** — a symptom–disease dataset (binary symptom columns + a disease label column) is loaded, optionally in chunks for large files.
2. **Symptom Encoding** — symptoms are represented as a multi-hot feature vector (1 = symptom present, 0 = absent).
3. **Model Training** — a multi-class classifier (Random Forest by default) is trained to map symptom vectors to disease labels.
4. **Prediction** — for a new set of reported symptoms, the model outputs the most likely disease(s) with associated probabilities.
5. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion matrix summarize model performance on a held-out test set.

---

## 🛠️ Tech Stack

- **Python**
- **Scikit-learn** — Random Forest classifier, train/test split, evaluation metrics
- **Pandas / NumPy** — data handling and chunked large-file reading
- **Plotly / Matplotlib** — confusion matrix and prediction visualization
- **Streamlit** — interactive web interface, configured for large file uploads

---

## 📂 Project Structure

```
multi-symptom-disease-predictor/
│
├── app.py                    # Main application / entry point
├── model.py                   # Model training, evaluation, and prediction logic
├── data_preprocessing.py      # Data loading (chunked), symptom encoding
├── requirements.txt           # Project dependencies
├── README.md                   # Project documentation
├── .streamlit/
│   └── config.toml              # Streamlit config — raises upload limit to 3 GB
├── data/                        # Sample symptom-disease datasets
└── assets/                      # Screenshots, sample plots, etc.
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maatrixxrahul/multi-symptom-disease-predictor.git
   cd multi-symptom-disease-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

The included `.streamlit/config.toml` raises Streamlit's upload limit from the 200 MB default to **3072 MB (3 GB)** — no extra flags needed, it's picked up automatically.

---

## 🚀 Usage

### Train / Upload Data
1. Go to the **Train Model** tab.
2. Upload a symptom–disease CSV (binary symptom columns + a `disease` label column) — up to 3 GB.
3. Large files are read in chunks with progress feedback, then combined for training.
4. Review accuracy, classification report, and confusion matrix after training.

### Predict
1. Go to the **Predict** tab.
2. Check off the symptoms the patient is experiencing.
3. View the top predicted disease(s) with confidence scores.

---

## 📊 Example Output

- Model accuracy and classification report (precision/recall/F1 per disease)
- Confusion matrix heatmap
- Top-3 predicted diseases with probability bars for a given symptom set



---

## ⚠️ Notes on Large File Uploads (3 GB)

- Streamlit's default upload cap is 200 MB; this project overrides it via `.streamlit/config.toml` (`maxUploadSize = 3072`).
- Files are read in **chunks** (`pandas.read_csv(..., chunksize=...)`) rather than all at once, to reduce peak memory usage.
- Actual capacity still depends on your machine's available RAM — a 3 GB CSV can expand significantly once loaded into a DataFrame. For datasets that don't fit in memory even when chunked, consider a library like **Dask** or sampling the data before training.
- Browser upload of very large files can be slow depending on connection speed; a progress indicator is shown during upload/processing.

---

## 🔮 Future Improvements

- Support out-of-core / incremental training (`partial_fit`) for datasets larger than available RAM
- Add SHAP explainability for individual predictions
- Support additional model types (XGBoost, LightGBM) with a model-selection dropdown
- Symptom severity (not just presence/absence) as an input feature

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Rahul Sah**
GitHub: [maatrixxrahul](https://github.com/maatrixxrahul)
Portfolio: [maatrixxrahul.netlify.app](https://maatrixxrahul.netlify.app)
