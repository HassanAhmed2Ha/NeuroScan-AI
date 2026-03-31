import os
import joblib
import numpy as np
import shap
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from fastapi.middleware.cors import CORSMiddleware

# Disable GPU and reduce logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Build Model Architecture
model = Sequential([
    Input(shape=(5,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Global variables (Initialized as None)
explainer = None
scaler = None

# Helper for SHAP
def model_predict_wrapper(x):
    # Using model directly for speed
    return model(x).numpy().astype('float64')

# Fast Load (Only essentials)
try:
    model.load_weights('breast_cancer_model.keras')
    scaler = joblib.load('scaler.pkl')
    print("Core Model and Scaler Ready!")
except Exception as e:
    print(f"Loading Error: {e}")

class PatientData(BaseModel):
    worst_radius: float
    worst_texture: float
    worst_concave_points: float
    worst_area: float
    worst_concavity: float

@app.post("/predict")
def predict_cancer(data: PatientData):
    global explainer
    try:
        raw_data = np.array([[
            data.worst_radius, data.worst_texture, 
            data.worst_concave_points, data.worst_area, data.worst_concavity
        ]])
        
        scaled_data = scaler.transform(raw_data)
        
        # Standard Prediction
        prediction = model(scaled_data).numpy()
        probability = float(prediction[0][0])
        result = "Malignant" if probability < 0.5 else "Benign"
        
        # 2. LAZY SHAP INITIALIZATION
        # Only build it once, during the first request
        if explainer is None:
            print("Initializing SHAP Explainer on-the-fly...")
            background = np.zeros((1, 5))
            explainer = shap.KernelExplainer(model_predict_wrapper, background)
        
        # Calculate SHAP with low samples for speed
        shap_values = explainer.shap_values(scaled_data, nsamples=40)
        
        if isinstance(shap_values, list):
            shap_list = shap_values[0].flatten().tolist()
        else:
            shap_list = shap_values.flatten().tolist()

        return {
            "probability": probability,
            "prediction": result,
            "shap_values": shap_list
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e), "shap_values": [0.0]*5}