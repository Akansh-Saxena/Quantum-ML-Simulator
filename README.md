# 🌍 Multimodal Weather & Physics Intelligence Hub

**Lead AI Engineer:** Akansh Saxena  
**Institution:** Final-Year B.Tech CSE, J.K. Institute of Applied Physics & Technology, Allahabad University  
**Project:** Multimodal Capstone Portfolio Project

## 📖 Overview
This repository contains a production-grade, state-of-the-art **Multimodal AI Dashboard** that simultaneously models complex physics phenomena and analyzes real-world geospatial/visual data. Evolving from a pure physics simulation, this project now seamlessly marries:
- **Dual-Mode Location Engine**: Auto-detects user environments or performs Global City/Country searches via Geocoding APIs.
- **Chronological & NWS Intelligence**: Tracks localized timezones, 24-hour meteorological trends, and displays real-time simulated National Weather Service (NWS) style emergency alerts.
- **Multimodal Visual Intelligence (CV)**: Ingests localized environmental input via live Webcam capturing or File uploading, inferenced on lightweight Convolutional Neural Networks (MobileNetV2).
- **Quantum Physics Synthesis**: Employs an advanced Keras LSTM Network tailored to process 100-step time-series datasets, mapping the environment's pressure and temperature dynamically onto a 3D Gravitational Disruption Manifold.

## 🛠️ Cutting-Edge Tech Stack
- **Frontend / Data App:** Streamlit
- **Multimodal AI / Deep Learning:** TensorFlow, Keras (MobileNetV2), OpenCV 
- **Explainable AI (XAI):** SHAP, Scikit-Learn
- **Geospatial & Timezones:** Geocoder, Geopy, TimezoneFinder, pytz, Requests (OpenWeatherMap)
- **Data & Vizards:** Pandas, NumPy, Plotly (3D Surfaces), Matplotlib

## 🚀 How to Run Locally

### Prerequisites
Make sure Python 3.9+ is installed and accessible via PATH.

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Akansh-Saxena/Quantum-ML-Simulator.git
   cd Quantum-ML-Simulator
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Provide API Secrets:**
   Create a directory `.streamlit` and add `secrets.toml`:
   ```toml
   OPENWEATHER_API_KEY = "your_actual_api_key_here"
   ```

4. **Launch the Multimodal App:**
   ```bash
   streamlit run streamlit_app.py
   ```
   *The system will launch locally at `http://localhost:8501`, syncing real-time weather alerts and physics telemetry continuously.*
