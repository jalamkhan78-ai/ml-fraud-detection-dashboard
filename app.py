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

MODEL_ICONS = {
    "Logistic Regression": "📉",
    "Decision Tree": "🌳",
    "K-Nearest Neighbors": "👥",
    "Gaussian Naive Bayes": "🧠",
    "Support Vector Machine": "🛡️",
}

st.set_page_config(page_title="Credit Card Fraud Classifier", page_icon="💳", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: linear-gradient(120deg, #0f2929, #0f766e);
        border: 1px solid #14b8a6;
        padding: 14px;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: white;
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0f2929, #0f766e);
        color: white;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>💳 Credit Card Fraud Detection</h1>
        <p>Compare five trained classifiers and explore their fraud-detection performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}

models = load_models()

with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox(
        "Choose a classification model",
        list(MODEL_FILES.keys()),
        format_func=lambda name: f"{MODEL_ICONS[name]} {name}",
    )
    uploaded = st.file_uploader("Upload labelled test data", type=["csv"])
    use_sample = st.checkbox("Use included test_data.csv", value=uploaded is None)
    st.divider()
    st.caption("Input Requirements")
    st.info("30 numeric feature columns plus `Class` (0 = genuine, 1 = fraud).")

if use_sample and uploaded is None:
    sample_path = BASE_DIR / "test_data.csv"
    if sample_path.exists():
        uploaded = sample_path
    else:
        st.warning("The included test_data.csv file was not found. Upload a CSV to continue.")

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

    st.subheader(f"Evaluation: {MODEL_ICONS[model_name]} {model_name}")
    genuine_count = int((y == 0).sum())
    fraud_count = int((y == 1).sum())
    st.caption(f"Analysed {len(data):,} transactions • {fraud_count:,} fraud • {genuine_count:,} genuine")

    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Accuracy", f"{accuracy:.4f}")
    c2.metric("📈 AUC Score", f"{auc:.4f}" if pd.notna(auc) else "N/A")
    c3.metric("🔍 Precision", f"{precision:.4f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("🚨 Recall", f"{recall:.4f}")
    c5.metric("⚖️ F1 Score", f"{f1:.4f}")
    c6.metric("🧮 MCC Score", f"{mcc:.4f}")

    preview = data.copy()
    preview["Predicted_Class"] = predictions
    preview["Prediction_Label"] = preview["Predicted_Class"].map({0: "Genuine", 1: "Fraud"})

    tab_summary, tab_matrix, tab_predictions = st.tabs(
        ["📊 Summary", "🔲 Confusion matrix", "🔎 Predictions"]
    )
    with tab_summary:
        st.subheader("Classification report")
        report = pd.DataFrame(
            classification_report(
                y, predictions, target_names=["Genuine (0)", "Fraud (1)"],
                zero_division=0, output_dict=True
            )
        ).T
        st.dataframe(report.style.format("{:.4f}"), use_container_width=True)
        st.progress(float(recall), text=f"Fraud recall: {recall:.1%}")
        st.progress(float(precision), text=f"Fraud precision: {precision:.1%}")

    with tab_matrix:
        cm = confusion_matrix(y, predictions, labels=[0, 1])
        matrix = pd.DataFrame(
            cm, index=["Actual genuine", "Actual fraud"],
            columns=["Predicted genuine", "Predicted fraud"],
        )
        st.dataframe(matrix, use_container_width=True)
        st.caption("Rows represent actual classes; columns represent model predictions.")

    with tab_predictions:
        prediction_filter = st.multiselect(
            "Show prediction classes",
            options=["Genuine", "Fraud"],
            default=["Genuine", "Fraud"],
        )
        filtered_preview = preview[preview["Prediction_Label"].isin(prediction_filter)]
        st.caption(f"Showing {len(filtered_preview):,} of {len(preview):,} transactions")
        st.dataframe(filtered_preview.head(100), use_container_width=True)
        st.download_button(
            "⬇️ Download predictions CSV",
            preview.to_csv(index=False).encode("utf-8"),
            file_name="creditcard_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a labelled CSV or select the included test_data.csv to view evaluation metrics.")
    st.subheader("Available models")
    st.write(" • ".join(f"{MODEL_ICONS[name]} {name}" for name in MODEL_FILES))
