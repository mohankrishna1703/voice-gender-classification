# Voice Gender Classification
This is a machine learning project that classifies human voice samples as **male** or **female** using extracted audio features.
The project includes:
- Data loading and preprocessing
- Clustering using K-Means
- Classification using Random Forest
- A simple Streamlit web app for predictions

---

## 📁 Project Files

| File Name         | Description |
|------------------|-------------|
| `load_data.py`    | Loads and displays dataset preview |
| `preprocess.py`   | Normalizes and scales features |
| `eda.py` | Shows basic charts like gender count and heatmap |
| `cluster.py`      | Groups voice data using KMeans clustering |
| `train_model.py`  | Trains a Random Forest classifier |
| `save_model.py`   | Saves the trained model as `rf_model.pkl` |
| `app.py`          | Streamlit app to predict gender using uploaded CSV |

---

## 🔍 Features Used

The model uses various audio features like:
- Spectral centroid
- Bandwidth
- Pitch (mean, min, max)
- MFCCs (mel-frequency cepstral coefficients)
- Energy, contrast, flatness, skew, and more

---
