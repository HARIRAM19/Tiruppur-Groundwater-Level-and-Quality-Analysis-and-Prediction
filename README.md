# 🌍 Real-Time IoT-Based Spatio-Temporal Deep Learning Framework for Groundwater Prediction

## 📌 Overview

This repository contains the complete implementation of **Phase I** and **Phase II** of a Master's thesis project focused on **groundwater level and quality prediction** using **IoT, Deep Learning, and Spatio-Temporal Analysis**.

The project evolves from a **hybrid AI-based forecasting system (Phase I)** to a **fully integrated real-time IoT-enabled intelligent monitoring system (Phase II)**.

📄 **Thesis References:**

* Phase I Thesis
* Phase II Thesis 

---

## 🚀 Project Highlights

* 🔄 **End-to-End Pipeline**: From data acquisition → preprocessing → modeling → deployment
* 🌐 **Real-Time IoT Integration** (Phase II)
* 🧠 **Advanced Deep Learning Models**:

  * LSTM
  * CNN-LSTM
  * Ensemble Learning
* 🌍 **Spatio-Temporal Intelligence** using geographic features
* 📊 **Interactive Dashboard & Alert System**
* ⚡ **Scenario Simulation for Decision Support**
* 📉 **High Prediction Accuracy** (Low MAE & RMSE)

---

## 🧠 Problem Statement

Groundwater resources are under severe stress due to:

* Over-extraction
* Industrial contamination
* Climate variability

Traditional monitoring systems:

* ❌ Manual and periodic
* ❌ Lack predictive intelligence
* ❌ Do not integrate spatial-temporal dynamics

This project addresses these gaps through a **real-time intelligent prediction framework**.

---

## 🏗️ System Architecture

The system follows a **multi-layer architecture**:

1. **Sensing Layer** – IoT sensors (Water level, pH, TDS, EC, Turbidity, Temperature)
2. **Edge Layer** – Signal conditioning & ADC conversion
3. **Communication Layer** – Secure transmission (GSM / LoRa)
4. **Cloud & Storage Layer** – Scalable data storage
5. **Preprocessing Layer** – Cleaning, normalization, windowing
6. **Spatial Layer** – Latitude, longitude, geo-features
7. **Model Layer** – Deep learning models
8. **Application Layer** – Dashboard + Alerts

---

## 🔬 Methodology

### 📊 Data Sources

* Groundwater Level Data (1994–2024)
* Water Quality Parameters (pH, TDS, EC, etc.)
* Meteorological Data (Rainfall, Temperature)
* Climate Projections (CMIP6 scenarios)

### ⚙️ Data Processing

* Missing value handling
* Outlier removal
* Temporal windowing
* Feature engineering (lag, rolling, seasonal)

### 🌍 Spatial Intelligence

* Latitude & Longitude encoding
* Zone-based feature engineering

---

## 🤖 Models Used

### Phase I (Hybrid AI Framework)

* ICEEMDAN + VMD + SMA + LSTM
* CNN-LSTM
* Adaptive Weighting Model (AWM)

### Phase II (Real-Time Deep Learning System)

* LSTM
* CNN-LSTM
* Ensemble Models
* Anomaly Detection

---

## 📈 Results & Performance

### Key Outcomes:

* ✅ Accurate prediction of groundwater level & quality
* ✅ Real-time monitoring capability
* ✅ Improved robustness using ensemble learning

### Performance Metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Square Error)
* R² Score
* NSE (Nash–Sutcliffe Efficiency)

📌 Phase II demonstrates:

* Improved scalability
* Higher prediction accuracy
* Real-time applicability

---

## 📊 Features

* 📡 Real-time IoT data streaming
* 📉 Time-series analysis & decomposition
* 🌡️ Multi-parameter groundwater monitoring
* 🗺️ Spatio-temporal heatmaps
* 📊 Dashboard visualization
* 🚨 Alert system for anomalies
* 🔮 Scenario simulation (rainfall, pumping)

---

## 🖥️ Tech Stack

### 🔧 Software

* Python 3.x
* TensorFlow / Keras
* NumPy, Pandas
* Scikit-learn
* Matplotlib / Seaborn / Plotly

### ⚙️ Hardware

* ESP32 / Arduino
* IoT Sensors (Water level, pH, TDS, EC, etc.)

### ☁️ Infrastructure

* Cloud storage
* Web-based dashboard

---

## 🔄 Phase Evolution

| Feature       | Phase I              | Phase II                 |
| ------------- | -------------------- | ------------------------ |
| Data Type     | Historical + Climate | Real-time IoT            |
| Models        | Hybrid AI            | Deep Learning + Ensemble |
| Deployment    | Offline              | Real-Time                |
| Visualization | Basic                | Interactive Dashboard    |
| Alerts        | ❌                    | ✅                        |

---

## 🌍 Use Cases

* 💧 Groundwater resource management
* 🌾 Agricultural planning
* 🏭 Industrial water monitoring
* 🌦️ Climate impact analysis
* 🚨 Early warning systems

---

## 📌 Key Contributions

* 🔗 Integration of IoT + Deep Learning + Spatial Intelligence
* ⚡ Real-time prediction system
* 📊 Multi-parameter groundwater analysis
* 🧠 Advanced hybrid and ensemble modeling
* 🌍 Scalable framework for real-world deployment

---

## 📖 References

For detailed theoretical background, methodology, and results, refer to:

* 📄 Phase I Thesis Document
* 📄 Phase II Thesis Document

---

## 👨‍💻 Author

**HARIRAM S**

M.E. Computer Science and Engineering

Anna University, Chennai

---

## ⭐ Final Note

This project demonstrates a transition from **research-driven modeling (Phase I)** to a **practical, real-time intelligent system (Phase II)**, offering a scalable solution for **sustainable groundwater management** in climate-vulnerable regions.

---

> 🌱 *“Data-driven intelligence for sustainable water future.”*
