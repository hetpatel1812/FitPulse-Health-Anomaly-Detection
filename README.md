# 🩺 FitPulse: Health Anomaly Detection from Fitness Devices  

**FitPulse** is a Python-based health anomaly detection system developed as part of the **Infosys Springboard Internship Program** (Python Programming Domain).  
It processes **fitness tracker data** (heart rate, steps, and sleep) to identify unusual patterns and provide meaningful health insights.  

---

## 🔎 Introduction  

Fitness devices collect large amounts of health data, but interpreting it manually can be challenging.  
**FitPulse** applies **data preprocessing, feature extraction, anomaly detection, and visualization** to highlight unusual behavior in:  

- Heart Rate  
- Step Count  
- Sleep Patterns  

It also provides an **interactive dashboard** to explore anomalies and export reports.  

---

## 🚀 Features  

### 🔹 Data Collection & Preprocessing  
- Import heart rate, steps, and sleep data from **CSV/JSON**.  
- Clean timestamps, handle missing values, and align intervals.  

### 🔹 Feature Extraction & Modeling  
- Extract statistical features with **TSFresh**.  
- Model seasonal trends with **Facebook Prophet**.  
- Cluster behaviors using **KMeans** and **DBSCAN**.  

### 🔹 Anomaly Detection & Visualization  
- Rule-based anomalies (thresholds on HR, steps).  
- Model-based anomalies (residual errors, clustering outliers).  
- Visualize patterns with **Matplotlib** & **Plotly**.  

### 🔹 Dashboard for Insights  
- Built with **Streamlit**.  
- Upload fitness data, run detection, and view results.  
- Export reports to **CSV/PDF**.  

---


