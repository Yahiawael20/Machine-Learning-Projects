import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("random_forest_laptop_model.pkl")

st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻" 
)

st.title("💻 Laptop Price Predictor")

# ---------------- Inputs ----------------
Company = st.selectbox("Company", ['Dell','Lenovo','HP','Asus','Acer','MSI','Toshiba','Apple','Samsung','Razer','Mediacom','Microsoft','Xiaomi','Vero','Chuwi','Google','LG','Huawei','Fujitsu'])
TypeName = st.selectbox("Type", ['Notebook','Gaming','Ultrabook','2 in 1 Convertible','Workstation','Netbook'])
Inches = st.number_input("Inches")

Ram = st.selectbox("RAM", [4,8,16,32])
Weight = st.number_input("Weight")

OpSys = st.selectbox("operating system", ['Windows','No OS','Linux','Chrome OS','macOS','Android'])

SSD = st.number_input("SSD", value=0)
HDD = st.number_input("HDD", value=0)
Flash = st.number_input("Flash", value=0)
Hybrid = st.number_input("Hybrid", value=0)

Cpu_brand = st.selectbox("CPU Brand", ['Intel','AMD','Samsung'])     
Cpu_family = st.selectbox("CPU Family", ["Core i7", "Core i5", "Core i3", "Celeron Dual", "Pentium Quad",
    "Core M", "A9-Series 9420", "Celeron Quad", "A6-Series 9220",
    "A12-Series 9720P", "Atom x5-Z8350", "A8-Series 7410",
    "A9-Series 9410", "Atom x5-Z8550", "Pentium Dual", "Ryzen 1700",
    "E-Series E2-9000e", "Atom X5-Z8350", "Xeon E3-1535M",
    "A9-Series A9-9420", "A10-Series 9620P", "E-Series 7110",
    "A10-Series A10-9620P", "A10-Series 9600P", "A6-Series A6-9220",
    "Xeon E3-1505M", "A4-Series 7210", "E-Series 9000",
    "Cortex A72&A53", "E-Series 9000e", "FX 8800P",
    "Ryzen 1600", "A12-Series 9700P", "Atom x5-Z8300",
    "Atom Z8350", "A6-Series 7310", "E-Series 6110",
    "E-Series E2-6110", "FX 9830P"])   

Cpu_speed = st.number_input("CPU Speed")

Gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD", "ARM"])

Touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
IPS = st.selectbox("IPS Display", ["No", "Yes"])


Touchscreen = 1 if Touchscreen == "Yes" else 0
IPS = 1 if IPS == "Yes" else 0

X_res = st.number_input("X resolution")
Y_res = st.number_input("Y resolution")

PPI = ((X_res**2 + Y_res**2)**0.5) / Inches if Inches != 0 else 0

# ---------------- Prediction ----------------
if st.button("Predict"):

    df = pd.DataFrame({
        'Company':[Company],
        'TypeName':[TypeName],
        'Inches':[Inches],
        'Ram (GB)':[Ram],
        'OpSys':[OpSys],
        'Weight (KG)':[Weight],
        'SSD':[SSD],
        'HDD':[HDD],
        'Flash':[Flash],
        'Hybrid':[Hybrid],
        'Cpu_brand':[Cpu_brand],
        'Cpu_family':[Cpu_family],
        'Cpu_speed':[Cpu_speed],
        'Gpu_brand':[Gpu_brand],
        'Touchscreen':[Touchscreen],
        'IPS':[IPS],
        'X_res':[X_res],
        'Y_res':[Y_res],
        'PPI':[PPI]
    })

    
    df = pd.get_dummies(df)

    
    model_cols = model.feature_names_in_
    df = df.reindex(columns=model_cols, fill_value=0)

    # predict
    pred_log = model.predict(df)[0]
    pred = np.expm1(pred_log)

    st.success(f" Price: {int(pred)}")
