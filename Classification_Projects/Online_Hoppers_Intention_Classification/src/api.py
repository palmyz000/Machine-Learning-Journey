from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# --- 🛠️ 1. จัดการ Path ใหม่ (ชี้ไปที่โฟลเดอร์ models) ---

# 1. หาตำแหน่งของไฟล์ api.py ปัจจุบัน (ตอนนี้เราอยู่ใน .../src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. ถอยหลังออกไป 1 ก้าว เพื่อกลับไปที่โฟลเดอร์หลักของโปรเจกต์
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 3. ระบุเป้าหมายไปที่โฟลเดอร์ models
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# 4. ประกอบร่างชื่อไฟล์
model_path = os.path.join(MODELS_DIR, 'model.joblib')
columns_path = os.path.join(MODELS_DIR, 'column_names.joblib')

# 5. โหลดโมเดล
model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

# --- 🚀 2. สร้าง API ---
app = FastAPI(
    title="E-commerce Intention API",
    description="API สำหรับทำนายว่าลูกค้าจะกดซื้อสินค้าหรือไม่",
    version="1.0"
)

# สร้าง Schema สำหรับรับข้อมูลจากหน้าเว็บ
class ShopperData(BaseModel):
    Administrative: int = 0
    Administrative_Duration: float = 0.0
    Informational: int = 0
    Informational_Duration: float = 0.0
    ProductRelated: int = 10
    ProductRelated_Duration: float = 150.0
    BounceRates: float = 0.0
    ExitRates: float = 0.02
    PageValues: float = 20.5
    SpecialDay: float = 0.0
    Month: str = "Nov"
    OperatingSystems: int = 2
    Browser: int = 2
    Region: int = 1
    TrafficType: int = 2
    VisitorType: str = "Returning_Visitor"
    Weekend: bool = False

@app.post("/predict")
def predict_purchase(data: ShopperData):
    # แปลงข้อมูล JSON เป็น DataFrame
    df = pd.DataFrame([data.dict()])
    df['Weekend'] = df['Weekend'].astype(int)
    
    # ทำ One-Hot Encoding
    cols_to_ohe = ['Month', 'VisitorType', 'Region', 'Browser', 'OperatingSystems', 'TrafficType']
    df_processed = pd.get_dummies(df, columns=cols_to_ohe)
    
    # จัดเรียงคอลัมน์ให้ตรงกับตอนเทรนโมเดล
    df_final = df_processed.reindex(columns=model_columns, fill_value=0)
    
    # ทำนายผล
    prediction = model.predict(df_final)[0]
    probability = model.predict_proba(df_final)[0][1]
    
    return {
        "status": "success",
        "prediction": int(prediction),
        "probability_to_buy": float(probability),
        "action_recommended": "Offer 10% Discount" if probability > 0.3 else "Do Nothing" 
    }