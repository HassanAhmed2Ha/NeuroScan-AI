# NeuroScan AI - Breast Cancer Diagnosis System

**Live Backend:** [https://huggingface.co/spaces/Hassan2007/tumor-diagnosis-backend](https://huggingface.co/spaces/Hassan2007/tumor-diagnosis-backend)  
**Frontend Deployment:** [https://tumor-diagnosis-frontend.vercel.app/](https://tumor-diagnosis-frontend.vercel.app/)

---

## Introduction

This project is not just a machine learning model.  
It is an attempt to turn a simple classification task into a **complete, usable system** that combines prediction, explanation, and accessibility.

The idea was straightforward at the beginning:  
build a model that classifies breast tumors as malignant or benign.

But very quickly, it became clear that the real challenge was not the model itself —  
it was everything around it: deployment, consistency, interpretability, and usability.

So instead of stopping at a notebook, I built a full pipeline:  
model → API → explainability → frontend → deployment.

---

## Why I Built This Project

Most machine learning projects stop at one point:  
a trained model with good accuracy.

But in reality, that is not useful on its own.

A real system should:
- Accept real inputs from users  
- Return understandable results  
- Explain its decisions  
- Work reliably across environments  

While working on this, I realized a gap:  
there are many tutorials about models, but very few about turning them into **real, usable tools**.

This project is my attempt to close that gap.

---

## What This Project Does

The system allows users to:
- Input clinical features of a tumor  
- Get a real-time prediction (Malignant / Benign)  
- See confidence score  
- Understand *why* the model made that decision using SHAP  

The focus is not only prediction —  
but **making the model interpretable and usable**.

---

## Key Components

- Deep Learning model (TensorFlow / Keras)
- FastAPI backend for inference
- StandardScaler for consistent preprocessing
- SHAP for explainability
- React frontend for interaction and visualization
- HuggingFace Spaces deployment (backend)
- Vercel deployment (frontend)

---

## Challenges I Faced (And What Actually Happened)

### 1. Environment Inconsistency (Colab vs HuggingFace)

The model worked perfectly on Google Colab.  
Then it broke on HuggingFace.

Different versions of:
- TensorFlow  
- NumPy  
- SHAP  

caused unexpected errors.

**What I learned:** A model that works in one environment is not guaranteed to work in another.

**What I did:** - Fixed all dependencies manually in `requirements.txt`
- Used `tensorflow-cpu` to match HuggingFace constraints

---

### 2. TensorFlow on HuggingFace (Free Tier Limitations)

HuggingFace free spaces:
- No GPU  
- Limited resources  

This made TensorFlow heavy and sometimes unstable.

**Solution:** Forced CPU usage, reduced logs and overhead, and optimized inference flow.
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

---

### 3. The Critical Bug: Model Always Predicts "Malignant"

After deployment, something was clearly wrong.  
No matter what input I entered, the model kept predicting the same result.  
This was not a model issue — it was worse.

**Root Cause:** I trained the model using StandardScaler, but I did NOT apply the same scaler during inference.  
So the model was receiving completely different data distributions.

**Fix:** Saved the scaler during training:
```python
import joblib
joblib.dump(my_scaler, 'scaler.pkl')
```
Loaded it in the backend and applied it before prediction:
```python
scaler = joblib.load('scaler.pkl')
scaled_data = scaler.transform(raw_data)
```

**Lesson:** Preprocessing is not optional. If you skip it, your model becomes meaningless.

---

### 4. SHAP Was Too Slow

SHAP is powerful, but expensive.  
Running it directly caused slow responses and high computation cost.

**Solution:** - Lazy initialization (only once)
- Reduced number of samples

This made the system usable without removing explainability.

---

### 5. Prompt Engineering Built the Frontend

Instead of building the UI manually from scratch, I used structured prompt engineering to generate:
- Layout
- UX flow
- Animation logic
- SHAP visualization

Then refined it manually.  
The result: a clean, interactive interface that connects directly to the model.

---

## Model & Files

**Save Model**
```python
model.save('breast_cancer_model.keras')
```

**Save Scaler**
```python
joblib.dump(my_scaler, 'scaler.pkl')
```

---

## API

**POST `/predict`**

**Input**
```json
{
  "worst_radius": 0.0,
  "worst_texture": 0.0,
  "worst_concave_points": 0.0,
  "worst_area": 0.0,
  "worst_concavity": 0.0
}
```

**Output**
```json
{
  "probability": 0.0,
  "prediction": "Malignant | Benign",
  "shap_values": [...]
}
```

---

## Project Structure

- **Backend** → FastAPI (`APT.py`)
- **Frontend** → React (`App.jsx`)
- **Dependencies** → `requirements.txt`
- **UI Config** → `package.json`

---

## What This Project Really Taught Me

The model is the easiest part.  
The real difficulty is:
- Making everything consistent
- Handling deployment limitations
- Debugging silent errors
- Turning output into something understandable

There is a big difference between:  
"a model that works" and "a system people can actually use".

This project is about crossing that gap.

---

## Disclaimer

This system is for research and educational purposes only.  
It is not a medical diagnostic tool.

---

## Author

**Hassan Ahmed** Bioinformatics | Data Science | AI Systems  
Alexandria, Egypt

- **LinkedIn:** [https://www.linkedin.com/in/hassan-ahmed2007](https://www.linkedin.com/in/hassan-ahmed2007)
- **Portfolio:** [https://hassan-ahmed-portfolio.vercel.app](https://hassan-ahmed-portfolio.vercel.app)
- **Email:** hassanahmed07.e9@gmail.com
