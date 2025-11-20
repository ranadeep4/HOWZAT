# IPL Score Prediction - Training & Evaluation Pipeline

This repository contains a small pipeline to load the provided IPL dataset, preprocess categorical features (label encoding + scaling), and train/evaluate multiple regression models (scikit-learn) for predicting the total score.

What I added
- `requirements.txt` – Python packages used.
- `environment.yml` – optional conda environment file.
- `create_env.ps1` – PowerShell script to create a venv and install packages (Windows friendly).
- `src/` – pipeline source code:
  - `data_loader.py` – loads `content/ipl_data.csv` by default.
  - `preprocess.py` – fits LabelEncoders and MinMaxScaler and saves them to `artifacts/`.
  - `models.py` – registry of models to try and evaluation helpers.
  - `train_pipeline.py` – orchestrates training/evaluation and saves models + results.
  - `run_experiments.py` – convenience wrapper to run the pipeline and print results.
- `.gitignore` and `README.md`.

How to create an isolated environment (Windows PowerShell)

1) Using the included PowerShell helper (recommended for Windows):

```powershell
# From project root
.\create_env.ps1 -EnvName .venv
# Activate
.\.venv\Scripts\Activate.ps1
```

2) Or using conda:

```powershell
conda env create -f environment.yml
conda activate ipl_env
```

How to run the pipeline

1) Activate your environment (see above).
2) From project root run:

```powershell
python -m src.train_pipeline
# or to run the wrapper
python -m src.run_experiments
```

Artifacts
- Trained models and preprocessing artifacts are saved into `artifacts/` as joblib files.
- A `results.json` file with metrics is written to project root.

Notes & next steps
- The preprocessing uses LabelEncoder with a simple fallback for unseen categories (maps to -1). You may want to replace this with more robust handling (target encoding, one-hot with fixed categories, or embedding approaches for NN).
- I added a few scikit-learn regressors (Linear, RandomForest, HistGradientBoosting, MLP). You can add XGBoost, LightGBM or tune hyperparameters.
- If you want, I can run a quick training on a small subset or add a hyperparameter search and cross-validation step.
