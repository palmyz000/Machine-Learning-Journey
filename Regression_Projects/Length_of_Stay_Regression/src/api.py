from  fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Hospital Length of Stay Prediction API")

MODEL_DIR = os.path.join(CURRENT_DIR, "..", "models")
pipeline_path = os.path.join(MODEL_DIR, "lengthofstay_final_model.joblib")
columns_path = os.path.join(MODEL_DIR, "model_columns.joblib")

pipelines = joblib.load(pipeline_path)
model_cols = joblib.load(columns_path)

class PatientData(BaseModel):
    gender: str
    facid: str
    rcount: int
    vmonth: int
    vday_of_week: int
    
    dialysisrenalendstage: int
    asthma: int
    irondef: int
    pneum: int
    substancedependence: int
    psychologicaldisordermajor: int
    depress: int
    psychother: int
    fibrosisandother: int
    malnutrition: int
    secondarydiagnosisnonicd9: int
    
    hemo: float
    hematocrit: float
    neutrophils: float
    sodium: float
    glucose: float
    bloodureanitro: float
    creatinine: float
    bmi: float
    pulse: float
    respiration: float

@app.post("/predict")
def predict_los(data: dict):

    input_df =pd.DataFrame([data])
    processed_df =pd.get_dummies(input_df)
    final_df = processed_df.reindex(columns=model_cols, fill_value=0)   
    prediction_log = pipelines.predict(final_df)[0]
    
    prediction_days = float(np.expm1(prediction_log))
    prediction_rounded = int(round(prediction_days))
    
    return {
        "predicted_days": prediction_days,
        "predicted_days_rounded": prediction_rounded
    }

