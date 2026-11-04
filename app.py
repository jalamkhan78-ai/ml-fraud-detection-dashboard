import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Credit Card Fraud Classifier", page_icon="💳", layout="wide")
st.title("Credit Card Fraud Detection")
st.write(
    "Compare five trained classification models on an uploaded labelled CSV. "
    "The app evaluates the saved models only; it does not retrain them."
)

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

models = load_models()
model_name = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))
uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

st.caption("The CSV must contain the 30 model input columns and a target column named Class (0 = genuine, 1 = fraud).")
if uploaded is not None:
    data = pd.read_csv(uploaded)
    if "Class" not in data.columns:
        st.error("The uploaded CSV must include the target column 'Class'.")
        st.stop()

    expected_features = list(models[model_name].feature_names_in_) if hasattr(models[model_name], "feature_names_in_") else None
    if expected_features is None:
        # All models were fitted using the same 30-column feature order.
        expected_features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    missing = [c for c in expected_features if c not in data.columns]
    extra = [c for c in data.columns if c not in expected_features and c != "Class"]
    if missing:
        st.error("Missing required feature columns: " + ", ".join(missing))
        st.stop()
    if extra:
        st.warning("Extra columns will be ignored: " + ", ".join(extra))

    X = data[expected_features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data["Class"], errors="coerce")
    if X.isna().any().any() or y.isna().any():
        st.error("Input features and Class must contain numeric values without missing entries.")
        st.stop()
    if not set(y.unique()).issubset({0, 1}):
        st.error("Class must contain only 0 and 1.")
        st.stop()

    model = models[model_name]
    predictions = model.predict(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = predictions

    accuracy = accuracy_score(y, predictions)
    auc = roc_auc_score(y, scores) if y.nunique() == 2 else float("nan")
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)
    mcc = matthews_corrcoef(y, predictions)

    st.subheader(f"Evaluation: {model_name}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("AUC", f"{auc:.4f}" if pd.notna(auc) else "N/A")
    c3.metric("Precision", f"{precision:.4f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", f"{recall:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    st.dataframe(pd.DataFrame(cm, index=["Actual genuine", "Actual fraud"],
                              columns=["Predicted genuine", "Predicted fraud"]))
    st.subheader("Classification report")
    st.code(classification_report(y, predictions, target_names=["Genuine (0)", "Fraud (1)"], zero_division=0))
    preview = data.copy()
    preview["Predicted_Class"] = predictions
    st.subheader("Predictions preview")
    st.dataframe(preview.head(100), use_container_width=True)
    st.download_button("Download predictions CSV", preview.to_csv(index=False).encode("utf-8"),
                       file_name="creditcard_predictions.csv", mime="text/csv")
else:
    st.info("Upload the included test_data.csv to view evaluation metrics and the confusion matrix.")
    st.write("Available models:", ", ".join(MODEL_FILES.keys()))
