import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cardio AI Predictor", 
    page_icon="❤️", 
    layout="centered"
)

# --- 2. TRAIN MODEL (Background Process) ---
@st.cache_resource
def build_model():
    # Synthetic dataset: Age, Duration(mins), Intensity(1-10) -> Heart Rate
    data = {
        'Age':             [20, 25, 30, 35, 40, 45, 50, 55, 60, 22, 28, 48, 19, 65, 33],
        'Duration_Mins':   [10, 20, 30, 15, 45, 20, 60, 10, 30, 25, 40, 15, 10, 20, 50],
        'Intensity_Level': [3,  7,  8,  4,  6,  5,  5,  3,  6,  9,  8,  4,  2,  4,  7],
        # Target Variable: BPM (Beats Per Minute)
        'Heart_Rate_BPM':  [95, 145, 160, 110, 135, 125, 130, 100, 128, 175, 165, 115, 85, 105, 155]
    }
    
    df = pd.DataFrame(data)
    X = df[['Age', 'Duration_Mins', 'Intensity_Level']]
    y = df['Heart_Rate_BPM']
    
    regressor = LinearRegression()
    regressor.fit(X, y)
    
    return regressor

model = build_model()

# --- 3. UI LAYOUT ---
st.title("❤️ Exercise Heart Rate Predictor")
st.markdown("Use this AI tool to estimate your heart rate (BPM) based on your workout details.")

# Disclaimer
st.info("ℹ️ **Note:** This model assumes a healthy individual. Consult a doctor for medical advice.")

st.write("---")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Profile")
    age = st.slider("Select Age", 15, 80, 25)
    
with col2:
    st.subheader("Workout Details")
    duration = st.number_input("Duration (minutes)", min_value=5, max_value=120, value=30)
    intensity = st.slider("Intensity Level (1=Relaxed, 10=Max Effort)", 1, 10, 5)

# --- 4. PREDICTION ENGINE ---
if st.button("Calculate Heart Rate", type="primary"):
    # Prepare input for the model
    input_features = np.array([[age, duration, intensity]])
    
    # Predict
    prediction = model.predict(input_features)[0]
    result_bpm = int(prediction)
    
    # Calculate Max Heart Rate (Standard Formula: 220 - Age)
    max_hr = 220 - age
    
    st.write("---")
    st.subheader("Results")
    
    # Display Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated BPM", f"{result_bpm} ❤️")
    m2.metric("Max Safe HR", f"{max_hr} BPM")
    
    # Determine Zone
    percentage = (result_bpm / max_hr) * 100
    
    if percentage < 50:
        st.success(f"Zone: **Warm Up / Recovery** ({percentage:.1f}%)")
    elif 50 <= percentage < 70:
        st.info(f"Zone: **Fat Burn** ({percentage:.1f}%)")
    elif 70 <= percentage < 85:
        st.warning(f"Zone: **Cardio Endurance** ({percentage:.1f}%)")
    else:
        st.error(f"Zone: **Peak Performance / Anaerobic** ({percentage:.1f}%)")
        if result_bpm > max_hr:
            st.error("⚠️ **WARNING:** Predicted rate exceeds generic Max Heart Rate!")

# --- 5. SIDEBAR INFO ---
with st.sidebar:
    st.header("How it works")
    st.write("We use **Linear Regression** to analyze relationships between:")
    st.write("1. **Age:** HR typically decreases with age.")
    st.write("2. **Duration:** Longer workouts drift HR higher.")
    st.write("3. **Intensity:** The biggest driver of HR.")