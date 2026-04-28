import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import cv2
import os
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(page_title="ERIS Earthquake Digital Twin", layout="wide", page_icon="🌍")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #f0f2f6; border-radius: 4px; padding: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- LIVE MODEL TRAINING (Avoids AttributeError/Pickle Errors) ---
@st.cache_resource
def train_model_live():
    # Synthetic data based on your research logic
    data = {
        'seismic': np.random.randint(0, 100, 2000),
        'radon': np.random.randint(0, 100, 2000),
        'animal': np.random.randint(0, 100, 2000),
        'emf': np.random.randint(0, 100, 2000),
    }
    df = pd.DataFrame(data)
    # Your Research Logic: If Radon & Animal are high OR Seismic is very high
    df['risk'] = ((df['radon'] > 70) & (df['animal'] > 70) | (df['seismic'] > 85)).astype(int)
    
    X = df[['seismic', 'radon', 'animal', 'emf']]
    y = df['risk']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

model = train_model_live()

# --- SIDEBAR ---
st.sidebar.title("🛠️ ERIS Control Panel")
seismic = st.sidebar.slider("Seismic Noise", 0, 100, 35)
radon = st.sidebar.slider("Radon Gas Levels", 0, 100, 75)
animal = st.sidebar.slider("Animal Anomaly", 0, 100, 65)
emf = st.sidebar.slider("EMF Disturbance", 0, 100, 20)

st.sidebar.markdown("---")
st.sidebar.info("Developed by: Srushti, Shrutika, Parnika, Helly")

# --- HEADER ---
st.title("🧠 ERIS: Earthquake Risk Intelligence System")
st.caption("BSc-IT National Award-Winning Research | Mumbai Digital Twin")

# --- PREDICTION LOGIC ---
features = np.array([[seismic, radon, animal, emf]])
risk_proba = model.predict_proba(features)[0, 1]
risk_percent = int(risk_proba * 100)

# --- TABS ---
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
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Mumbai Vulnerability Map")
        map_data = pd.DataFrame({
            'lat': np.random.normal(19.0760, 0.05, 50),
            'lon': np.random.normal(72.8777, 0.05, 50),
        })
        st.map(map_data)

with tab_xai:
    st.markdown("### 🔍 Why is the Risk High?")
    importances = model.feature_importances_
    feature_names = ['Seismic', 'Radon', 'Animal', 'EMF']
    contributions = np.array(importances) * np.array([seismic, radon, animal, emf])
    xai_fig = go.Figure(data=[go.Bar(
        x=feature_names, y=(contributions/max(np.sum(contributions), 1)*100),
        marker_color=['#4CC9F0', '#4361EE', '#3A0CA3', '#7209B7']
    )])
    st.plotly_chart(xai_fig, use_container_width=True)

with tab_cv:
    st.markdown("### 📹 Visual Seismograph (Innovation)")
    uploaded_video = st.file_uploader("Upload tremor footage", type=["mp4", "mov"])
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
        cap.release()