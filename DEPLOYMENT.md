# CommodiSense Deployment Guide

## Why not Streamlit Cloud?

Streamlit Cloud forces Python 3.14, which is incompatible with the `numba`/`llvmlite` ecosystem (required by SHAP, pandas-ta, etc.). We've removed these dependencies and switched to **Hugging Face Spaces**, which supports Python 3.11 natively.

## Deployment on Hugging Face Spaces

### 1. Create a Space on Hugging Face

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `commodisense`
   - **Space SDK**: `Docker`
   - **Visibility**: Public (for GitHub Actions to access)
   - **License**: MIT
3. Click **Create Space**

### 2. Connect to GitHub

In the space settings (⚙️ icon):

1. **Repository URL**: https://github.com/Yashvardhansharma112/commodisense
2. **Sync with Git repo**: Enabled
3. Add a webhook to auto-deploy on push

### 3. Set Secrets

In space settings → **Secrets**, add the following (get keys from your local `.env`):

```
GROQ_API_KEY=your_groq_key_here
EIA_API_KEY=your_eia_key_here
USDA_API_KEY=your_usda_key_here
```

See `.env.example` in the repo for where to get these keys.

### 4. Create Dockerfile

The space will auto-detect `requirements.txt` and `dashboard/app.py` (Streamlit default entry point).

If you need a custom Dockerfile, create `Dockerfile` in the repo root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

### 5. Deploy

Push any commit to `main` branch:

```bash
git push origin main
```

Hugging Face will auto-build and deploy within 2-5 minutes.

## Running Locally

For full functionality including SHAP explainability, run with Python 3.12 and the full requirements:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-pipeline.txt
python -m spacy download en_core_web_sm
streamlit run dashboard/app.py
```

## Pipeline (GitHub Actions)

GitHub Actions runs the full pipeline with `requirements-pipeline.txt` (includes torch, transformers, spacy, prophet) on Python 3.12. The Hugging Face Space runs only the dashboard (Python 3.11, dashboard-only requirements).

This keeps deployment simple and fast while preserving full ML capability in automation.
