# HOWZAT — IPL Score Prediction

Compact training and evaluation pipeline for predicting IPL match totals. The repo includes data-loading and preprocessing utilities, model training and evaluation scripts (scikit-learn), a simple web UI, and a small RAG (retrieval-augmented generation) helper.

**Highlights**
- **Data & preprocessing:** scripts to load and prepare the IPL dataset, including label-encoding and scaling.
- **Training pipeline:** several regression baselines (Linear, RandomForest, HistGB, MLP) with utilities to evaluate and persist models.
- **Web UI & RAG:** lightweight UI templates in `templates/` and a Retrieval-Augmented-Generation helper under `ragfeature/`.

Repository layout
- `app.py`: top-level web application entrypoint (simple demo UI).
- `ragfeature/`: tools to build/query a FAISS index for RAG use; contains `app.py` and `build_ipl_rag.py`.
- `artifacts/`: saved preprocessing objects and trained models (joblib files).
- `content/`: primary data `ipl_data.csv` and original player stats folder.
- `data/`: other prepared CSVs and notebooks used during preprocessing.
- `src/`: core training pipeline and helpers
  - `data_loader.py` — dataset loader
  - `preprocess.py` — encoder/scaler fitting & saving
  - `models.py` — model registry and evaluation helpers
  - `train_pipeline.py` — orchestrates training and saving models
  - `run_experiments.py` — convenience wrapper for experiments
- `templates/` and `static/`: minimal web UI (HTML + CSS)

Getting started (Windows PowerShell)

1) Using the included PowerShell helper (recommended):

```powershell
# From project root
.\create_env.ps1 -EnvName .venv
# Activate the created venv
.\.venv\Scripts\Activate.ps1
```

2) Or with conda:

```powershell
conda env create -f environment.yml
conda activate ipl_env
```

Install (if not using the helper):

```powershell
python -m pip install -r requirements.txt
```

Training & experiments

- Fit the preprocessors and train models with the main pipeline:

```powershell
python -m src.train_pipeline
# or run the convenience script
python -m src.run_experiments
```

- Output artifacts:
  - Trained models and preprocessing artifacts are written to `artifacts/` (joblib files).
  - Evaluation metrics are written to `results.json` in the project root.

Running the web UI (demo)

- A minimal web UI is available; run the top-level `app.py`:

```powershell
# From project root
python app.py
```

- Open your browser at the address printed by the script (typically `http://127.0.0.1:5000`). The templates live in `templates/` and CSS in `static/`.

RAG (retrieval) feature

- The `ragfeature/` folder contains utilities to build a FAISS index and a small helper app for retrieval-based features.
- To rebuild the FAISS index (if you'd like to update the retriever):

```powershell
python ragfeature/build_ipl_rag.py
```

Artifacts and model files

- Look in `artifacts/` for files like `RandomForest.joblib`, `RandomForest_tuned.joblib`, `HistGB.joblib`, `MLP.joblib`, and `preprocess_artifacts.joblib`.

Development notes & suggestions

- Preprocessing uses `LabelEncoder` with a fallback mapping for unseen categories (encoded as `-1`). For robustness consider:
  - One-hot with a fixed category list
  - Target encoding or learned embeddings for high-cardinality fields
- You can add LightGBM/XGBoost or a full cross-validated hyperparameter search using scikit-learn's `GridSearchCV` / `RandomizedSearchCV` or Optuna.
- If you want, I can add:
  - a short demo notebook that trains on a small subset
  - a CI target to run a smoke training during PRs

Contact / Next steps

If you'd like, I can:
- run a quick training pass and attach the results
- add hyperparameter tuning with cross-validation
- add a small demo showing how to use the RAG index in a notebook

Enjoy! — The HOWZAT tooling is intentionally compact, so it can be a quick starting point for further experimentation.
