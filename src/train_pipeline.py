"""Train and evaluate multiple models on the IPL dataset.

Usage: python -m src.train_pipeline  (from project root)
"""
import os
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from src.data_loader import load_ipl_data
from src.preprocess import fit_transform, transform_with_artifacts, load_artifacts
from src.models import get_models, evaluate_regression, save_model


def run(output_report: str = "results.json"):
    print("Loading data...")
    df = load_ipl_data()

    print("Splitting data into train/test...")
    # We'll split dataframe-level to enable proper K-fold target-encoding without leakage
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    print("Preprocessing and fitting encoders/scaler with K-fold target encoding...")
    X_train, X_test, y_train, y_test, artifacts = fit_transform(train_df, test_df)

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        print(f"Evaluating {name}...")
        preds = model.predict(X_test)
        metrics = evaluate_regression(y_test, preds)
        print(f"{name} metrics: {metrics}")
        results[name] = metrics
        save_model(model, name)

    # Save preprocessing artifacts already saved by fit_transform (preprocess_artifacts.joblib)
    print("Preprocessing artifacts saved to artifacts/preprocess_artifacts.joblib")

    # Save results
    out = Path(output_report)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Finished. Models and artifacts saved to artifacts/. Results written to {out}")


if __name__ == '__main__':
    run()
