import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import cv2
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="ERIS Earthquake Digital Twin", layout="wide", page_icon="🌍")

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    if os.path.exists("earthquake_model.pkl"):
        with open("earthquake_model.pkl", "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# --- SIDEBAR ---
st.sidebar.title("🛠️ ERIS Control Panel")
st.sidebar.markdown("### 1. Adjust Precursors")
seismic = st.sidebar.slider("Seismic Noise", 0, 100, 35)
radon = st.sidebar.slider("Radon Gas Levels", 0, 100, 75)
animal = st.sidebar.slider("Animal Anomaly", 0, 100, 65)
emf = st.sidebar.slider("EMF Disturbance", 0, 100, 20)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Judge's Guide")
st.sidebar.info("""
1. **Adjust sliders** to see the Risk Meter change live.
2. **Click 'Run Simulation'** for the Digital Twin report.
3. **Switch Tabs** to see Explainability and CV Video Analysis.
""")

# --- HEADER ---
st.title("🧠 ERIS: Earthquake Risk Intelligence System")
st.caption("BSc-IT National Award-Winning Research | Developed by Srushti, Shrutika, Parnika, Helly")

# --- LOGIC ---
if model:
    features = np.array([[seismic, radon, animal, emf]])
    risk_proba = model.predict_proba(features)[0, 1]
    risk_percent = int(risk_proba * 100)
else:
    risk_percent = 0
    st.error("Model file not found! Run train_model.py first.")

# --- MAIN TABS (THIS PREVENTS EXCESSIVE SCROLLING) ---
tab_main, tab_xai, tab_cv = st.tabs(["📊 Live Risk Dashboard", "🤖 Explainable AI (XAI)", "📹 Visual Seismograph"])

with tab_main:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Earthquake Risk Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percent,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "crimson" if risk_percent > 60 else "gold" if risk_percent > 30 else "limegreen"},
                'steps': [
                    {'range': [0, 30], 'color': "#e8f5e9"},
                    {'range': [30, 60], 'color': "#fffde7"},
                    {'range': [60, 100], 'color': "#ffebee"}
                ]
            }
        ))
        fig.update_layout(height=350, margin=dict(t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Mumbai Vulnerability Map")
        map_data = pd.DataFrame({
            'lat': np.random.normal(19.0760, 0.05, 50),
            'lon': np.random.normal(72.8777, 0.05, 50),
        })
        st.map(map_data)

    if st.button("🚀 Run Digital Twin Simulation"):
        st.success(f"Simulation Analysis: {risk_percent}% instability detected for Mumbai Suburban.")

with tab_xai:
    st.markdown("### 🔍 Why is the Risk High?")
    if model:
        importances = model.feature_importances_
        feature_names = ['Seismic', 'Radon', 'Animal', 'EMF']
        contributions = np.array(importances) * np.array([seismic, radon, animal, emf])
        
        xai_fig = go.Figure(data=[go.Bar(
            x=feature_names, y=(contributions/np.sum(contributions)*100),
            marker_color=['#4CC9F0', '#4361EE', '#3A0CA3', '#7209B7']
        )])
        xai_fig.update_layout(title="Feature Contribution to Current Prediction", yaxis_title="% Influence")
        st.plotly_chart(xai_fig, use_container_width=True)
        st.info("Explainability (XAI) ensures that AI decisions are transparent and trustable for emergency responders.")

with tab_cv:
    st.markdown("### 📹 Visual Seismograph (Innovation)")
    uploaded_video = st.file_uploader("Upload tremor footage (MP4/MOV)", type=["mp4", "mov"])
    if uploaded_video:
        with open("temp.mp4", "wb") as f: f.write(uploaded_video.read())
        cap = cv2.VideoCapture("temp.mp4")
        vibrations = []
        ret, prev = cap.read()
        if ret:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            for _ in range(50):
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vibrations.append(np.sum(cv2.absdiff(prev_gray, gray)) / (gray.shape[0]*gray.shape[1]))
                prev_gray = gray
            st.line_chart(vibrations)
            st.write(f"**Mean Vibration Intensity:** {np.mean(vibrations):.2f}")
        cap.release()

st.markdown("---")
st.write("© 2026 ERIS Project | Scaling AI for Social Good")