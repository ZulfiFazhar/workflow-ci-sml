# MLflow CI/CD Project - AI Coding Agent Instructions

## Project Overview

This is a Dicoding MSML submission project demonstrating MLflow integration with GitHub Actions CI/CD. The project trains a Random Forest classifier for credit score classification and tracks experiments using MLflow with local tracking URI.

## Architecture & Key Components

### MLflow Project Structure

- **MLProject/MLproject**: MLflow project definition with entry points and parameters
- **MLProject/modelling.py**: Training script with manual MLflow logging (autolog disabled)
- **MLProject/conda.yaml**: Environment specification (Python 3.12.7, MLflow 2.19.0)
- **.github/workflows/mlflow-ci.yml**: CI/CD pipeline that trains model on every push to `main`

### Data Flow

1. GitHub Actions triggers on push to `MLProject/**` or `.github/workflows/**`
2. Workflow runs `mlflow run .` with local environment manager
3. Model trained with RandomForestClassifier, logs metrics/artifacts to local `mlruns/`
4. Python script extracts latest run ID from MLflow tracking server
5. Artifacts (confusion matrix, feature importance, classification report) generated and can be committed back to repo

## Critical Workflows

### Running MLflow Project Locally

```bash
cd MLProject
mlflow run . \
  --experiment-name Credit_Scoring_CI \
  -P data_path=Credit_Score_Classification_Dataset_preprocessing.csv \
  -P target_col="Credit Score" \
  --env-manager=local
```

### Retrieving Latest Run ID

The workflow uses a Python one-liner to get the most recent run:

```python
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
exp = mlflow.get_experiment_by_name('Credit_Scoring_CI')
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                          order_by=['start_time DESC'], max_results=1)
print(runs.iloc[0]['run_id'])
```

### Environment Variables

- `MLFLOW_TRACKING_URI`: `http://127.0.0.1:5000` (local tracking)
- `EXPERIMENT_NAME`: `Credit_Scoring_CI`
- `PYTHON_VERSION`: `3.12.7`

## Project-Specific Conventions

### MLflow Logging Pattern

**Manual logging is used instead of autolog** - see [modelling.py](MLProject/modelling.py#L77):

```python
mlflow.sklearn.autolog(disable=True)
```

This ensures explicit control over:

- Parameters: `n_estimators`, `max_depth`, `random_state`
- Metrics: `train_accuracy`, `test_accuracy`, `test_precision`, `test_recall`, `test_f1_score`, `overfitting_diff`
- Artifacts: `confusion_matrix.png`, `feature_importance.png`, `classification_report.json`
- Model registration: `CreditScoringCI`

### Artifact Generation

All plots use `matplotlib.use('Agg')` for headless rendering in CI environment. Artifacts are saved locally before logging:

- Confusion matrix: `plot_confusion_matrix()` → `confusion_matrix.png`
- Feature importance: `plot_feature_importance()` → `feature_importance.png` (top 20 features)
- Classification report: JSON format with per-class metrics

### CI/CD Specific Patterns

- Workflow uses `--env-manager=local` instead of conda (faster in CI)
- Dependencies installed via pip before MLflow run
- Conditional steps check `if: steps.get_run_id.outputs.RUN_ID != 'no-run-found'`
- Artifact paths use relative paths from `MLProject/` directory

## Integration Points

### GitHub Actions Triggers

- Push to `main` branch affecting `MLProject/**` or `.github/workflows/**`
- Manual trigger via `workflow_dispatch`

### MLflow Tracking

- Local tracking server at `http://127.0.0.1:5000` (ephemeral in CI)
- Experiment name: `Credit_Scoring_CI`
- Model registered as: `CreditScoringCI`

### Dependencies

Core stack (must match across [conda.yaml](MLProject/conda.yaml) and [workflow](MLProject/.github/workflows/mlflow-ci.yml#L41)):

- MLflow: `2.19.0` (pinned version)
- Python: `3.12.7`
- scikit-learn, pandas, numpy, matplotlib, seaborn

## Key Files Reference

- [MLProject/modelling.py](MLProject/modelling.py): Training logic, manual MLflow logging
- [MLProject/MLproject](MLProject/MLproject): Entry point parameters and defaults
- [.github/workflows/mlflow-ci.yml](.github/workflows/mlflow-ci.yml): CI/CD pipeline
- [MLProject/conda.yaml](MLProject/conda.yaml): Environment specification
