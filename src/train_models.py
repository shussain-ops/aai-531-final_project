"""Train credit card fraud detection models."""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from features import add_features
from evaluate import evaluate_model


def build_models(preprocessor):
    return {
        "Dummy Baseline": Pipeline(steps=[("preprocess", preprocessor), ("model", DummyClassifier(strategy="most_frequent"))]),
        "Logistic Regression": Pipeline(steps=[("preprocess", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42))]),
        "Decision Tree": Pipeline(steps=[("preprocess", preprocessor), ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42))]),
        "Random Forest": Pipeline(steps=[("preprocess", preprocessor), ("model", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42))]),
        "Hist Gradient Boosting": Pipeline(steps=[("preprocess", preprocessor), ("model", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42))]),
    }


def main():
    csv_path = Path("data/raw/creditcard.csv")
    df = pd.read_csv(csv_path)
    model_df = add_features(df)
    X = model_df.drop(columns=["Class"])
    y = model_df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scale_cols = ["Time", "Amount", "LogAmount", "Hour"]
    passthrough_cols = [c for c in X_train.columns if c not in scale_cols]
    preprocessor = ColumnTransformer(transformers=[("scale", StandardScaler(), scale_cols), ("pass", "passthrough", passthrough_cols)], remainder="drop")
    models = build_models(preprocessor)
    results = []
    trained_models = {}
    for name, model in models.items():
        print("Training:", name)
        if name == "Hist Gradient Boosting":
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))
        trained_models[name] = model
    metrics_df = pd.DataFrame(results).sort_values(by=["pr_auc_avg_precision", "recall_fraud"], ascending=False)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(artifacts_dir / "credit_card_fraud_model_metrics.csv", index=False)
    best_model_name = metrics_df.iloc[0]["model"]
    joblib.dump(trained_models[best_model_name], artifacts_dir / "credit_card_fraud_best_model.joblib")
    print()
    print("Best model:", best_model_name)
    print(metrics_df)


if __name__ == "__main__":
    main()
