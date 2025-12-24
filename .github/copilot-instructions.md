# MLflow CI/CD Project - AI Coding Agent Instructions

## Project Overview

This is an MLflow-based credit scoring classification project with automated CI/CD via GitHub Actions. The workflow trains a RandomForest model on preprocessed credit score data and tracks experiments using MLflow's file-based backend.

## Architecture & Key Components

### Directory Structure

- **`MLProject/`**: MLflow project root containing all ML code and data
  - `MLproject`: MLflow project configuration defining entry points and parameters
  - `conda.yaml`: Python 3.12.7 environment with MLflow 2.19.0, scikit-learn, pandas, numpy, matplotlib, seaborn
  - `modelling.py`: Main training script with manual MLflow logging (autolog disabled)
  - `Credit_Score_Classification_Dataset_preprocessing.csv`: Preprocessed dataset (included in repo)
- **`.github/workflows/mlflow-ci.yml`**: CI/CD pipeline triggered on pushes to main or manual dispatch

### Data Flow

1. CI workflow triggers on changes to `MLProject/**` or `.github/workflows/**`
2. GitHub Actions runs `mlflow run .` with local env manager (pip-based, not conda)
3. `modelling.py` loads CSV, splits data (80/20 stratified), trains RandomForest
4. Manual MLflow logging: params, metrics (accuracy/precision/recall/F1), artifacts (confusion matrix, feature importance plots, classification report JSON)
5. Model registered as "CreditScoringCI" in MLflow model registry
6. Artifacts stored in `mlruns/` (file-based tracking, gitignored)

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

**Important**: Use `--env-manager=local` (not conda) to match CI behavior. Dependencies must be pre-installed via pip.

### Modifying Model Parameters

Edit [MLProject/MLproject](MLProject/MLproject) entry point parameters:

- `n_estimators` (default: 100)
- `max_depth` (default: 20)
- `test_size` (default: 0.2)

Pass via CLI: `-P n_estimators=200 -P max_depth=30`

### Accessing MLflow UI

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then navigate to `http://localhost:5000`

## Project-Specific Conventions

### MLflow Logging Pattern

- **Autolog is DISABLED** in [modelling.py](MLProject/modelling.py#L77): `mlflow.sklearn.autolog(disable=True)`
- All logging is manual within `with mlflow.start_run()` context
- Artifacts logged: `classification_report.json`, `confusion_matrix.png`, `feature_importance.png`, model as "CreditScoringCI"
- Tracking URI adapts: uses env var `MLFLOW_TRACKING_URI` or defaults to `./mlruns`

### CI/CD Specifics

- **Environment**: Python 3.12.7 hardcoded in [workflow](../.github/workflows/mlflow-ci.yml#L15)
- **Dependencies**: Installed via pip in CI (not conda), despite `conda.yaml` existing
- **Run ID Extraction**: [Post-training script](../.github/workflows/mlflow-ci.yml#L55-L70) queries MLflow for latest run ID using pandas-based search
- **Trigger Paths**: Only `MLProject/**` and `.github/workflows/**` changes trigger builds
- **⚠️ CRITICAL**: Do NOT set `MLFLOW_TRACKING_URI` to `http://127.0.0.1:5000` in workflow env vars—this causes "Connection refused" errors in CI. The workflow relies on modelling.py defaulting to file-based `./mlruns` when the env var is unset.

### Matplotlib Backend

Uses `Agg` backend (`matplotlib.use('Agg')`) for headless plotting in CI—no display required

### Target Column Convention

- Target column name: `"Credit Score"` (with space, not snake_case)
- Features: All columns except target
- Labels derived from `y.unique()` for confusion matrix

## Integration Points

### GitHub Actions

- Workflow file: [.github/workflows/mlflow-ci.yml](../.github/workflows/mlflow-ci.yml)
- No Docker deployment configured (env var `DOCKER_IMAGE_NAME` defined but unused)
- **File-based MLflow backend only**—no remote tracking server. Must NOT set `MLFLOW_TRACKING_URI` to HTTP server URL in workflow
- `permissions: contents: write` for potential artifact commits (not currently used)

### MLflow Experiment Naming

- Experiment name: `Credit_Scoring_CI` (defined in workflow env vars)
- Run name: `ci_random_forest` (hardcoded in [modelling.py](MLProject/modelling.py#L79))

## Common Tasks

### Adding New Metrics

1. Calculate metric after prediction in [train_model()](MLProject/modelling.py#L115-L119)
2. Log with `mlflow.log_metric("metric_name", value)`
3. Prints to console for CI logs visibility

### Changing Dataset

1. Replace `Credit_Score_Classification_Dataset_preprocessing.csv` in `MLProject/`
2. Ensure target column matches or update `-P target_col=...` in workflow
3. If feature names change, verify feature importance plotting handles new count

### Debugging CI Failures

- Check Python version compatibility (locked to 3.12.7)
- Verify MLflow 2.19.0 compatibility with dependencies
- Review `mlflow run` output in Actions logs—tracks params, metrics, artifacts in real-time
- Common issue: local conda env vs CI's pip-based `--env-manager=local`
- **"Connection refused" to localhost:5000**: `MLFLOW_TRACKING_URI` env var in workflow is set to HTTP server URL. Remove it or set to `file:./mlruns`. The code in modelling.py defaults to file-based tracking when env var is absent.

## Anti-Patterns to Avoid

- Don't re-enable MLflow autolog—breaks current manual logging workflow
- Don't use conda in CI—workflow bypasses conda.yaml and uses pip directly
- Don't move MLProject files out of `MLProject/` directory—workflow paths are hardcoded
- Don't set `MLFLOW_TRACKING_URI` to HTTP server URL (`http://127.0.0.1:5000`) in CI workflow—use file-based tracking or omit the env var entirely
