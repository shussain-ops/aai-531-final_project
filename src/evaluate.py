"""Model evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def get_fraud_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return model.predict(X)


def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_score = get_fraud_scores(model, X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_fraud": precision_score(y_test, y_pred, zero_division=0),
        "recall_fraud": recall_score(y_test, y_pred, zero_division=0),
        "f1_fraud": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc_avg_precision": average_precision_score(y_test, y_score),
    }


def build_review_queue_scenarios(X_test, y_test, scores, review_rates=None):
    if review_rates is None:
        review_rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    test_results = X_test.copy()
    test_results["actual_class"] = y_test.values
    test_results["fraud_score"] = scores
    test_results = test_results.sort_values("fraud_score", ascending=False).reset_index(drop=True)
    total_fraud = int((test_results["actual_class"] == 1).sum())
    rows = []
    for rate in review_rates:
        n_review = max(1, int(len(test_results) * rate))
        reviewed = test_results.head(n_review)
        captured_fraud = int((reviewed["actual_class"] == 1).sum())
        rows.append({
            "review_rate": rate,
            "transactions_reviewed": n_review,
            "fraud_captured": captured_fraud,
            "fraud_capture_rate": captured_fraud / total_fraud if total_fraud else 0,
            "alert_precision": captured_fraud / n_review,
        })
    return pd.DataFrame(rows)
