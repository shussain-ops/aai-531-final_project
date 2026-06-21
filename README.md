# Credit Card Fraud Detection Final Project

## Business Use Case

This project solves a credit card fraud detection problem using supervised machine learning. The goal is to identify fraudulent transactions from highly imbalanced transaction data so that a financial institution can reduce fraud losses while minimizing false alarms on genuine customers.

## Dataset

Kaggle dataset: `mlg-ulb/creditcardfraud`

Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains credit card transactions with these fields:

- `Time`: seconds elapsed from the first transaction in the dataset.
- `Amount`: transaction amount.
- `V1` to `V28`: anonymized PCA-transformed numerical features.
- `Class`: target variable where `1 = fraud` and `0 = genuine`.

## Project Structure

```text
credit_card_fraud_final_project/
├── notebooks/
│   └── Final Project SectionX-Team 1 - Credit Card Fraud.ipynb
├── src/
│   ├── data_download.py
│   ├── data_audit.py
│   ├── features.py
│   ├── train_models.py
│   └── evaluate.py
├── data/
│   └── raw/
├── artifacts/
├── slides/
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download `creditcard.csv` from Kaggle and place it in:

```text
data/raw/creditcard.csv
```

3. Or download using Kaggle API:

```bash
set KAGGLE_USERNAME=codetrickle
set KAGGLE_KEY=your_kaggle_api_key
python src/data_download.py
```

4. Open the notebook and run cells top to bottom.

## Security Notes

Do not commit API keys, `kaggle.json`, `.env`, raw data files, trained model files, or generated artifacts to GitHub.

## Modeling Approach

The notebook compares Dummy baseline, Logistic Regression, Decision Tree, Random Forest, and Histogram Gradient Boosting.

Main metrics: fraud precision, fraud recall, F1-score, ROC-AUC, PR-AUC, confusion matrix, threshold tuning, and business review queue impact.
