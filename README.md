# Credit Card Fraud Detection — ML Assignment 2

## a. Problem statement
Build and compare five supervised classification models to identify fraudulent credit-card transactions (`Class = 1`) and genuine transactions (`Class = 0`). Because fraud cases are rare, accuracy alone is not sufficient; AUC, precision, recall, F1 and MCC are also reported.

## b. Dataset description
- Dataset: Credit Card Fraud Detection (`creditcard.csv`) supplied with the assignment.
- Shape: 284,807 rows × 31 columns.
- Inputs: 30 numeric columns (`Time`, `V1`–`V28`, `Amount`).
- Target: `Class` (0 = genuine, 1 = fraud).
- Class counts: 284,315 genuine and 492 fraud transactions.
- Missing values: none.
- The `V1`–`V28` columns are anonymized features in the supplied data.
- A stratified 80/20 split was made. To keep training and the interactive demo practical, the saved models were fitted on a stratified 60,000-row subset of the training partition and evaluated on a stratified 20,000-row holdout sample. The same holdout file is used for all models.

## c. GitHub Repository Link
Add your repository URL here after creating and pushing this project to your own GitHub account:
`<YOUR_GITHUB_REPOSITORY_URL>`

## d. Models used and evaluation
All five models use the same input features and the same held-out test rows. StandardScaler is included in the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and Linear SVM. The Decision Tree is trained without scaling. Class weighting is used for Logistic Regression, Decision Tree and Linear SVM to address class imbalance.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| Logistic Regression | 0.9829 | 0.9506 | 0.0815 | 0.8824 | 0.1493 | 0.2653 |\n| Decision Tree | 0.9984 | 0.8968 | 0.5294 | 0.7941 | 0.6353 | 0.6477 |\n| KNN | 0.9992 | 0.9116 | 0.8065 | 0.7353 | 0.7692 | 0.7697 |\n| Gaussian Naive Bayes | 0.9801 | 0.9613 | 0.0687 | 0.8529 | 0.1272 | 0.2388 |\n| Support Vector Machine | 0.9821 | 0.9501 | 0.0761 | 0.8529 | 0.1398 | 0.2517 |\n
### Observations
- **Logistic Regression:** High recall, identifying most fraud cases, but it generates many false positives. Its precision and F1 are therefore low.
- **Decision Tree:** Strong accuracy and a more balanced precision/recall trade-off than the linear models. It misses some fraud cases but produces fewer false alarms.
- **KNN:** Best overall balance in this run, with the highest F1 and MCC. It has high precision and reasonable recall, although it misses some fraud transactions.
- **Gaussian Naive Bayes:** Highest AUC in this run and high recall, but many genuine transactions are flagged as fraud, resulting in low precision.
- **Support Vector Machine (LinearSVC):** High recall and AUC, but false positives reduce precision and F1.
- **Overall winner:** KNN based on the highest F1 (0.7692) and MCC (0.7697) on this particular holdout sample. If the priority is catching as many fraud transactions as possible, Logistic Regression, Gaussian Naive Bayes or SVM have higher recall in this run. The preferred model depends on the operational cost of false negatives versus false positives.

### Confusion matrices
Confusion matrix order is `[[TN, FP], [FN, TP]]`.
- **Logistic Regression:** `[[19628, 338], [4, 30]]`\n- **Decision Tree:** `[[19942, 24], [7, 27]]`\n- **KNN:** `[[19960, 6], [9, 25]]`\n- **Gaussian Naive Bayes:** `[[19573, 393], [5, 29]]`\n- **Support Vector Machine:** `[[19614, 352], [5, 29]]`\n
## Running locally
1. Install Python 3.10 or newer.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Upload `test_data.csv` in the app and select a model.

## Repository contents
- `app.py` — Streamlit interface; loads saved models, accepts labelled CSV test data, calculates metrics, displays confusion matrix/report and allows prediction download.
- `train.py` — training script (requires the original `creditcard.csv` in this folder).
- `model/` — five saved classifiers/pipelines.
- `test_data.csv` — common labelled holdout sample for app evaluation.
- `metrics.json` — metrics from the generated holdout experiment.
- `requirements.txt` — deployment dependencies.

## Deployment on Streamlit Community Cloud
Push this folder to GitHub. In Streamlit Community Cloud, create a new app, select the repository and branch, set the main file to `app.py`, and deploy. After deployment, add the live app URL to your final submission PDF.

## Important submission items you must complete yourself
- Run the project on BITS Virtual Lab and capture the required screenshot.
- Create/push the repository to your GitHub account and replace the placeholder URL above.
- Deploy the app and add its actual live URL.
- Include the final README content and links in the single PDF submission.
