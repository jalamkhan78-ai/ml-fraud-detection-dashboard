# Credit Card Fraud Detection — ML Assignment 2

## a. Problem Statement
Build and compare five supervised classification models to identify fraudulent credit-card transactions (`Class = 1`) and genuine transactions (`Class = 0`). Because fraud cases are rare, accuracy alone is not sufficient; AUC, precision, recall, F1 and MCC are also reported.

## b. Dataset Description
- Dataset: Credit Card Fraud Detection (`creditcard.csv`) supplied with the assignment.
- Shape: 284,807 rows × 31 columns.
- Inputs: 30 numeric columns (`Time`, `V1`–`V28`, `Amount`).
- Target: `Class` (0 = genuine, 1 = fraud).
- Class counts: 284,315 genuine and 492 fraud transactions.
- Missing values: none.
- The `V1`–`V28` columns are anonymized features in the supplied data.
- A stratified 80/20 split was made. To keep training and the interactive demo practical, the saved models were fitted on a stratified 60,000-row subset of the training partition and evaluated on a stratified 20,000-row holdout sample. The same holdout file is used for all models.

## c. GitHub Repository Link
https://github.com/jalamkhan78-ai/ml-fraud-detection-dashboard

## d. Models Used and Evaluation
All five models use the same input features and the same held-out test rows. StandardScaler is included in the pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and Linear SVM. The Decision Tree is trained without scaling. Class weighting is used for Logistic Regression, Decision Tree and Linear SVM to address class imbalance.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9829 | 0.9506 | 0.0815 | 0.8824 | 0.1493 | 0.2653 |
| Decision Tree | 0.9984 | 0.8968 | 0.5294 | 0.7941 | 0.6353 | 0.6477 |
| KNN | 0.9992 | 0.9116 | 0.8065 | 0.7353 | 0.7692 | 0.7697 |
| Gaussian Naive Bayes | 0.9801 | 0.9613 | 0.0687 | 0.8529 | 0.1272 | 0.2388 |
| Support Vector Machine | 0.9821 | 0.9501 | 0.0761 | 0.8529 | 0.1398 | 0.2517 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High recall, identifying most fraud cases, but it generates many false positives. Its precision and F1 are therefore low. |
| Decision Tree | Strong accuracy and a more balanced precision/recall trade-off than the linear models. It misses some fraud cases but produces fewer false alarms. |
| KNN | Best overall balance in this run, with the highest F1 and MCC. It has high precision and reasonable recall, although it misses some fraud transactions. |
| Gaussian Naive Bayes | Highest AUC in this run and high recall, but many genuine transactions are flagged as fraud, resulting in low precision. |
| Support Vector Machine (LinearSVC) | High recall and AUC, but false positives reduce precision and F1. |
| Overall winner | KNN based on the highest F1 (0.7692) and MCC (0.7697) on this particular holdout sample. If the priority is catching as many fraud transactions as possible, Logistic Regression, Gaussian Naive Bayes or SVM have higher recall in this run. The preferred model depends on the operational cost of false negatives versus false positives. |

