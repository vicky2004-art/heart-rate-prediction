import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from PIL import Image  # Import PIL to load the image

# --- 1. PAGE CONFIGURATION ---
# Attempt to load the custom icon image
try:
    # REPLACE 'icon.png' with the actual name of your image file if different
    icon_image = Image.open("icon.png") 
    page_icon_config = icon_image
except FileNotFoundError:
    # Fallback to an emoji if the image file isn't found
    page_icon_config = "icon.png"

st.set_page_config(
    page_title="Cardio AI Predictor", 
    page_icon=page_icon_config, 
    layout="centered"
)

# --- 2. TRAIN MODEL (Background Process) ---
@st.cache_resource
def build_model():
    try:
        # Load the uploaded dataset
        df = pd.read_csv('frequency_domain_features_test.csv')
        
        # Clean data: Remove ID column if it exists
        if 'uuid' in df.columns:
            df = df.drop(columns=['uuid'])
            
        # Handle missing values
        df = df.fillna(df.mean())

        # Generate synthetic target 'Heart_Rate' if missing (for demonstration)
        if 'Heart_Rate' not in df.columns:
            np.random.seed(42) 
            df['Heart_Rate'] = np.random.randint(60, 100, size=len(df))

        # Select features
        feature_cols = ['VLF', 'LF', 'HF']
        
        # Prepare X and y
        X = df[feature_cols]
        y = df['Heart_Rate']

        # Train Model
        model = LinearRegression()
        model.fit(X, y)
        
        return model, feature_cols
        
    except Exception as e:
        st.error(f"Error loading or training on dataset: {e}")
        return None, []

# Load the model
model, feature_names = build_model()

# --- 3. USER INTERFACE ---
st.title("❤️ Heart Rate Predictor (HRV)")
st.write("Enter Frequency Domain Features to predict Heart Rate.")

if model:
    # Create columns for input fields
    col1, col2, col3 = st.columns(3)
    
    # Input: VLF
    with col1:
        vlf_val = st.number_input("VLF", min_value=0.0, value=1000.0, help="Very Low Frequency component")

    # Input: LF
    with col2:
        lf_val = st.number_input("LF", min_value=0.0, value=500.0, help="Low Frequency component")
        
    # Input: HF
    with col3:
        hf_val = st.number_input("HF", min_value=0.0, value=20.0, help="High Frequency component")

    # --- 4. PREDICTION LOGIC ---
    if st.button("Calculate Heart Rate", type="primary"):
        # Prepare input
        input_features = np.array([[vlf_val, lf_val, hf_val]])
        
        # Predict
        prediction = model.predict(input_features)[0]
        result_bpm = int(prediction)
        
        # Display Results
        st.write("---")
        st.subheader("Results")
        st.metric("Estimated Heart Rate", f"{result_bpm} BPM")
        
        if result_bpm < 60:
            st.info("Status: Resting / Bradycardia range")
        elif 60 <= result_bpm <= 100:
            st.success("Status: Normal Resting Heart Rate range")
        else:
            st.warning("Status: Elevated / Tachycardia range")
            
else:
    st.warning("Data could not be loaded. Please ensure 'frequency_domain_features_test.csv' is in the app directory.")