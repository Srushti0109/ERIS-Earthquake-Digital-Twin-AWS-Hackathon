ERIS: Earthquake Risk Intelligence System 🌍🏛️
A Digital Twin for Mumbai’s Seismic Resilience
https://eris-earthquake-digital-twin-aws-hackathon-dyt58hhtldddvd7dji2.streamlit.app/

📌 Project Overview
ERIS (Earthquake Risk Intelligence System) is a cutting-edge Digital Twin and AI-driven forecasting platform developed for the Google/AWS Hackathon. Focused on the high-density urban landscape of Mumbai, ERIS integrates multi-modal geophysical precursors to predict seismic risks and provides Explainable AI (XAI) insights for urban planners and disaster management authorities.

By leveraging real-time data and predictive modeling, ERIS transitions earthquake management from reactive to proactive.

🚀 Key Features
Multi-Modal Data Fusion: Combines seismic history, tectonic plate movement (GNSS), groundwater fluctuations, and atmospheric ionospheric data.

Explainable AI (XAI): Uses SHAP and LIME to demystify "black-box" models, explaining why a specific risk level was assigned to a ward in Mumbai.

Digital Twin Visualization: A high-fidelity 3D representation of Mumbai’s infrastructure to simulate the impact of varying earthquake magnitudes.

Real-time Risk Mapping: Ward-by-ward vulnerability indexing based on soil quality, building age, and population density.

ERIS: Tech Strategy
Data: Multi-modal fusion of seismic, satellite, and groundwater data via AWS IoT.

Model: ConvLSTM for spatial-temporal risk forecasting (capturing where and when).

Twin: Graph-based Digital Twin (Amazon Neptune) to simulate impact on Mumbai’s critical infrastructure.

XAI: SHAP integration to explain "why" a risk score is high, building stakeholder trust.

Cloud: Serverless architecture (Lambda/Step Functions) for instant scaling and disaster resilience.

🏗️ Architecture
The system follows a modular pipeline designed for low latency and high reliability:

Data Ingestion: Automated scraping and API calls to geological and meteorological databases.

Preprocessing: Feature engineering on non-linear precursors (e.g., Radon gas emissions, Thermal anomalies).

Inference Engine: Hybrid LSTM-CNN models predict the probability of seismic events.

Digital Twin Overlay: Data is projected onto a 3D map of Mumbai to visualize "Hot Zones."

📈 Why Mumbai?
Despite not being on a major plate boundary, Mumbai sits near the Panvel Flexure and has experienced increasing low-intensity tremors. With its extreme population density and aging colonial-era infrastructure, a significant seismic event could be catastrophic. ERIS provides the intelligence needed to reinforce the right areas at the right time.

🚦 Getting Started
Prerequisites
Python 3.9+

AWS CLI configured

Mapbox API Token
