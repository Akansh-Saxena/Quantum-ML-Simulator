# 🌌 Advanced Machine Learning Physics Simulation

**Author:** Akansh Saxena  
**Institution:** Final-Year B.Tech CSE, J.K. Institute of Applied Physics & Technology, Allahabad University  
**Project:** Capstone Portfolio Project

## 📖 Overview
This repository contains a production-grade, interactive machine learning dashboard that simulates complex electromagnetic and quantum physics. It models localized gravitational disruption (Antigravity behavior) using a custom synthetic data generation engine and deep learning. 

The application utilizes a **Long Short-Term Memory (LSTM)** neural network built with TensorFlow/Keras to perform sequence prediction on 100-timestep engine burns. Additionally, it implements **Explainable AI (XAI)** using SHAP to interpret the theoretical physics phenomena driving the neural network's predictions.

## 🛠️ Tech Stack
- **Frontend / Data App:** Streamlit
- **Deep Learning / AI:** TensorFlow, Keras
- **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations), Scikit-Learn (Surrogate modeling for real-time inference)
- **Data Manipulation:** Pandas, NumPy
- **Interactive Visualization:** Plotly (3D Surfaces, Time-Series Sequence rendering), Matplotlib

## 🧠 Machine Learning Architecture
1. **Synthetic Data Engine:** Generates highly complex, non-linear time-series physics data with 4 multidimensional parameters: Fluctuating EM Field Strength, Exotic Matter Density, Subatomic Resonance, and Energy Input.
2. **LSTM Backend:** A Keras Sequential LSTM Neural Network trained heavily to recognize sequential quantum fluctuations and predict the final localized "Gravitational Disruption Index" target.
3. **Surrogate XAI Engine:** Employs a Random Forest Regressor over temporal averages to compute SHAP values instantly and explain the underlying dynamics in real-time on the Streamlit UI.

## 🚀 How to Run Locally

### Prerequisites
Ensure that Python 3.9+ is installed on your machine.

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-username/antigravity-simulation.git
   cd antigravity-simulation
   ```

2. **Install Dependencies:**
   It is recommended to use a virtual environment. Install the packages using the provided robust requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit App:**
   ```bash
   streamlit run streamlit_app.py
   ```
   *The stunning dark-mode application will automatically open in your default browser at `http://localhost:8501`.*
