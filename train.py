"""Reproducible training script for the Credit Card Fraud Detection assignment."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "creditcard.csv"  # Put the original dataset here before running.
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
X, y = df.drop(columns=["Class"]), df["Class"].astype(int)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# A stratified 60,000-row training sample keeps KNN/SVM practical on modest hardware.
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full, train_size=60000, random_state=42, stratify=y_train_full
)
X_test, _, y_test, _ = train_test_split(
    X_test_full, y_test_full, train_size=20000, random_state=42, stratify=y_test_full
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
    ]),
    "decision_tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1))
    ]),
    "gaussian_nb": Pipeline([("scaler", StandardScaler()), ("classifier", GaussianNB())]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42, max_iter=5000))
    ]),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")
test = X_test.copy()
test["Class"] = y_test.values
test.to_csv(ROOT / "test_data.csv", index=False)
print("Saved five models and test_data.csv.")
