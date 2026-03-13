# 🌍 Multimodal Weather & Physics Intelligence Hub

**Lead AI Engineer:** Akansh Saxena  
**Institution:** Final-Year B.Tech CSE, J.K. Institute of Applied Physics & Technology, Allahabad University  
**Project:** Multimodal Capstone Portfolio Project

## 📖 Overview
This repository contains a production-grade, state-of-the-art **Multimodal AI Dashboard** that simultaneously models complex physics phenomena and analyzes real-world geospatial/visual data. Evolving from a pure physics simulation, this project now seamlessly marries:
- Real-Time Geospatial Environment Mapping (Live IP geolocation & Weather API Integration).
- Computer Vision processing (via integrated Webcam capturing & File uploading inferenced on lightweight Neural Networks).
- Deep Sequence Learning via an LSTM architecture for complex Target Disruption extraction.

## 🛠️ Cutting-Edge Tech Stack
- **Frontend / Data App:** Streamlit
- **Multimodal AI / Deep Learning:** TensorFlow, Keras (MobileNetV2), OpenCV 
- **Explainable AI (XAI):** SHAP, Scikit-Learn
- **Geospatial & Networking:** Geocoder, Requests (OpenWeatherMap RESTful API)
- **Data & Vizards:** Pandas, NumPy, Plotly (3D Surfaces), Matplotlib

## 🧠 Multimodal Architecture & Pipeline
1. **Live Weather Mode & Sensor Integration**: Automatically captures the environment's telemetry via geolocation or webcam feed to establish the base engine parameters (simulating open-world influences).
2. **Visual Intelligence Engine**: Features an active Webcam tab capable of ingesting localized environmental input and driving rapid inference predictions through pre-trained deep convolutional neural networks.
3. **Sequence Target Engine**: Employs an advanced Keras LSTM Network tailored to process synthetic 100-step subatomic time-series datasets combined with atmospheric telemetry mapping.
4. **Surrogate XAI Layer**: Instantly breaks down the multidimensional influence mapped onto 3D Topology matrices, verifying adherence to simulated physical constraints.

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
   *The system will launch locally at `http://localhost:8501`*
