# Workflow CI - MLflow Project

Repository untuk Kriteria 3 Submission Dicoding MSML

## 🎯 Penilaian: ADVANCE (4 pts)

✅ **Basic (2 pts):**

- Folder MLProject
- Workflow CI untuk training model

✅ **Skilled (3 pts):**

- Menyimpan artifacts ke GitHub

✅ **Advance (4 pts):**

- Build Docker image dengan `mlflow build-docker`
- Push ke Docker Hub

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/workflows/
│   └── mlflow-ci.yml          # GitHub Actions workflow
├── MLProject/
│   ├── MLproject              # MLflow project config
│   ├── conda.yaml             # Conda environment
│   ├── modelling.py           # Training script
│   ├── Credit_Score_Classification_Dataset_preprocessing.csv
│   └── docker_hub_url.txt     # Link Docker Hub (generated)
├── .gitignore
└── README.md
```

## 🚀 Setup Instructions

### 1. Copy Dataset Preprocessing

```bash
# Copy dari repo Eksperimen_SML
copy ..\eksperimen-sml\preprocessing\"Credit Score Classification Dataset_preprocessing.csv" MLProject\
```

### 2. Setup GitHub Secrets

Buka: Repository Settings → Secrets and variables → Actions → New repository secret

Tambahkan 2 secrets:

- **DOCKER_USERNAME**: Username Docker Hub Anda
- **DOCKER_PASSWORD**: Password/Token Docker Hub Anda

### 3. Initialize Git & Push

```bash
cd Workflow-CI

# Initialize repo
git init
git add .
git commit -m "Initial commit: MLflow Project CI"

# Create GitHub repo (via web atau CLI)
gh repo create Workflow-CI --public --source=. --remote=origin --push

# Atau manual:
git remote add origin https://github.com/<username>/Workflow-CI.git
git branch -M main
git push -u origin main
```

### 4. Trigger Workflow

Workflow akan otomatis running setelah push. Atau trigger manual:

1. Buka tab "Actions" di GitHub
2. Pilih workflow "CI/CD MLflow Project"
3. Click "Run workflow"

## 🔧 Workflow Steps

1. ✅ **Set up job** - Prepare runner
2. ✅ **Checkout repo** - Clone repository
3. ✅ **Set up Python 3.12.7** - Install Python
4. ✅ **Check Env** - Verify environment
5. ✅ **Install dependencies** - Install MLflow & libraries
6. ✅ **Set MLflow Tracking URI** - Configure tracking
7. ✅ **Run mlflow project** - Train model dengan MLflow
8. ✅ **Get latest MLflow run id** - Ambil run ID hasil training
9. ✅ **Upload artifacts to GitHub** - Save mlruns & outputs
10. ✅ **Build Docker Model** - `mlflow models build-docker`
11. ✅ **Log in to Docker Hub** - Authenticate
12. ✅ **Tag Docker image** - Tag dengan SHA & latest
13. ✅ **Push Docker image** - Upload ke Docker Hub
14. ✅ **Save Docker Hub URL** - Generate link file
15. ✅ **Commit artifacts** - Push hasil ke repo

## 🐳 Docker Hub

Setelah workflow success, Docker image akan available di:

```
https://hub.docker.com/r/<DOCKER_USERNAME>/credit-scoring-model
```

### Pull & Run Docker Image

```bash
# Pull image
docker pull <DOCKER_USERNAME>/credit-scoring-model:latest

# Run model server
docker run -p 5002:8080 <DOCKER_USERNAME>/credit-scoring-model:latest

# Test prediction
curl -X POST http://localhost:5002/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

## 📊 MLflow Project Parameters

File `MLproject` mendefinisikan parameters:

- `data_path`: Path dataset preprocessing (default: Credit_Score_Classification_Dataset_preprocessing.csv)
- `target_col`: Kolom target (default: "Credit Score")
- `test_size`: Test split ratio (default: 0.2)
- `n_estimators`: RandomForest trees (default: 100)
- `max_depth`: Max tree depth (default: 20)

### Run Locally

```bash
cd MLProject

# Run dengan default parameters
mlflow run . --env-manager=local

# Run dengan custom parameters
mlflow run . \
  --env-manager=local \
  -P n_estimators=200 \
  -P max_depth=30
```

## 🔍 Artifacts Generated

Workflow akan generate:

1. **MLflow Artifacts:**

   - `confusion_matrix.png`
   - `feature_importance.png`
   - `classification_report.json`
   - Model registry

2. **Docker Image:**

   - Tagged: `latest` & `<commit-sha>`
   - Pushed to Docker Hub

3. **GitHub Artifacts:**
   - `mlruns/` - MLflow tracking
   - `docker_hub_url.txt` - Link Docker Hub

## ✅ Checklist Kriteria 3 - ADVANCE

- [ ] Folder MLProject dibuat
- [ ] File MLproject (config) ada
- [ ] File conda.yaml ada
- [ ] File modelling.py ada
- [ ] Dataset preprocessing di-copy
- [ ] GitHub Actions workflow (.github/workflows/mlflow-ci.yml)
- [ ] GitHub Secrets (DOCKER_USERNAME, DOCKER_PASSWORD) configured
- [ ] Repository GitHub public
- [ ] Workflow running & success minimal 1x
- [ ] Artifacts ter-upload ke GitHub
- [ ] Docker image ter-push ke Docker Hub
- [ ] File docker_hub_url.txt generated

## 🎓 Target: 4/4 pts (ADVANCE)

**Requirements Terpenuhi:**

- ✅ Folder MLProject
- ✅ Workflow CI
- ✅ Upload artifacts ke GitHub
- ✅ Build Docker dengan `mlflow build-docker`
- ✅ Push ke Docker Hub

## 🔧 Troubleshooting

**Error: Docker build failed**

- Check Docker Hub credentials di Secrets
- Pastikan DOCKER_USERNAME & DOCKER_PASSWORD benar

**Error: MLflow project run failed**

- Check dataset ada di MLProject/
- Verify conda.yaml dependencies

**Error: No run ID found**

- Check MLflow tracking URI
- Pastikan experiment name benar

## 📝 Notes

- Workflow menggunakan `--env-manager=local` untuk speed (skip conda env creation)
- Docker image size ~2GB (includes Python + ML libraries)
- Workflow duration: ~5-10 menit
- Docker push bisa lebih lama tergantung internet speed
