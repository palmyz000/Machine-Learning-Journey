import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "..", "models")

@st.cache_resource
def load_models():
    pipeline = joblib.load(os.path.join(MODEL_DIR, 'lengthofstay_final_model.joblib'))
    cols = joblib.load(os.path.join(MODEL_DIR, 'model_columns.joblib'))
    return pipeline, cols

pipeline, model_cols = load_models()

st.set_page_config(page_title="Hospital LOS AI", layout="wide")
st.title("🏥 ระบบพยากรณ์ระยะเวลาแอดมิท")

tab1, tab2, tab3 = st.tabs(["👤 ข้อมูลทั่วไป", "🩺 โรคประจำตัว", "🧪 ผลตรวจ & สัญญาณชีพ"])

with tab1:
    col1, col2 = st.columns(2)
    gender = col1.selectbox("เพศ", ['F', 'M'])
    facid = col2.selectbox("สาขาที่แอดมิท", ['A', 'B', 'C', 'D', 'E'])
    rcount = st.number_input("ประวัติแอดมิทซ้ำ", 0, 10, 0)
    vmonth = st.slider("เดือนที่แอดมิท (1-12)", 1, 12, 1)
    vday_of_week = st.slider("วันในสัปดาห์ (0=จันทร์, 6=อาทิตย์)", 0, 6, 0)

with tab2:
    st.write("ติ๊กเลือกหากคนไข้มีภาวะเหล่านี้:")
    col3, col4 = st.columns(2)
    dialysisrenalendstage = col3.checkbox("ไตวายระยะสุดท้าย")
    asthma = col3.checkbox("หอบหืด")
    irondef = col3.checkbox("ขาดธาตุเหล็ก")
    pneum = col3.checkbox("ปอดอักเสบ")
    substancedependence = col3.checkbox("ติดสารเสพติด")
    psychologicaldisordermajor = col4.checkbox("ความผิดปกติทางจิตเวชรุนแรง")
    depress = col4.checkbox("ซึมเศร้า")
    psychother = col4.checkbox("รับการบำบัดทางจิตเวช")
    fibrosisandother = col4.checkbox("พังผืดและอื่นๆ")
    malnutrition = col4.checkbox("ขาดสารอาหาร")
    secondarydiagnosisnonicd9 = col4.checkbox("มีโรคร่วมอื่นๆ")

with tab3:
    col5, col6, col7 = st.columns(3)
    hemo = col5.number_input("Hemoglobin", value=12.0)
    hematocrit = col5.number_input("Hematocrit", value=40.0)
    neutrophils = col5.number_input("Neutrophils", value=10.0)
    sodium = col6.number_input("Sodium", value=140.0)
    glucose = col6.number_input("Glucose", value=100.0)
    bloodureanitro = col6.number_input("BUN", value=15.0)
    creatinine = col7.number_input("Creatinine", value=1.0)
    bmi = col7.number_input("BMI", value=25.0)
    pulse = col7.number_input("Pulse", value=80.0)
    respiration = col7.number_input("Respiration", value=18.0)

st.markdown("---")
if st.button("🚀 ทำนายจำนวนวันนอน", type="primary", use_container_width=True):
    
    with st.spinner("AI กำลังวิเคราะห์ข้อมูล..."):
        input_data = {
            "gender": gender, "facid": facid, "rcount": rcount, "vmonth": vmonth, "vday_of_week": vday_of_week,
            "dialysisrenalendstage": int(dialysisrenalendstage), "asthma": int(asthma), "irondef": int(irondef),
            "pneum": int(pneum), "substancedependence": int(substancedependence), 
            "psychologicaldisordermajor": int(psychologicaldisordermajor), "depress": int(depress), 
            "psychother": int(psychother), "fibrosisandother": int(fibrosisandother), 
            "malnutrition": int(malnutrition), "secondarydiagnosisnonicd9": int(secondarydiagnosisnonicd9),
            "hemo": hemo, "hematocrit": hematocrit, "neutrophils": neutrophils, "sodium": sodium,
            "glucose": glucose, "bloodureanitro": bloodureanitro, "creatinine": creatinine,
            "bmi": bmi, "pulse": pulse, "respiration": respiration
        }
        
        input_df = pd.DataFrame([input_data])
        processed_df = pd.get_dummies(input_df)
        final_df = processed_df.reindex(columns=model_cols, fill_value=0)
        
        prediction_log = pipeline.predict(final_df)[0]
        prediction_days = int(round(np.expm1(prediction_log)))
        
        if prediction_days <= 3:
            st.success(f"🟢 คาดว่าจะต้องนอนโรงพยาบาลประมาณ: {prediction_days} วัน (เคสปกติ)")
        elif prediction_days <= 7:
            st.warning(f"🟡 คาดว่าจะต้องนอนโรงพยาบาลประมาณ: {prediction_days} วัน (เคสปานกลาง)")
        else:
            st.error(f"🔴 คาดว่าจะต้องนอนโรงพยาบาลประมาณ: {prediction_days} วัน (เคสเฝ้าระวัง/นอนนาน)")