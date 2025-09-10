# 🩺 FitPulse: Health Anomaly Detection from Fitness Devices  

FitPulse is a health anomaly detection system that processes **fitness tracker data** (heart rate, steps, and sleep) to identify unusual patterns and provide insights into user behavior.  

---

## 🚀 Features  

### 🔹 Data Collection & Preprocessing  
- Import heart rate, steps, and sleep data from **CSV/JSON**.  
- Clean timestamps, fix missing values, and align time intervals.  

### 🔹 Feature Extraction & Modeling  
- Extract statistical features using **[TSFresh](https://tsfresh.readthedocs.io/)**.  
- Use **Facebook Prophet** to model seasonal trends and detect deviations.  
- Apply clustering algorithms (**KMeans, DBSCAN**) to group behaviors.  

### 🔹 Anomaly Detection & Visualization  
- **Rule-based anomalies** (e.g., thresholds on HR, steps).  
- **Model-based anomalies** (residual errors, clustering outliers).  
- Visualizations with **Matplotlib** & **Plotly**.  

### 🔹 Dashboard for Insights  
- Interactive dashboard built with **Streamlit**.  
- Upload fitness tracker files, run anomaly detection, and visualize results.  
- Export reports to **PDF/CSV**.  

---

## 🛠️ Tools & Technologies  

- **Python** – main programming language.  
- **Libraries**:  
  - Data: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `plotly`  
  - ML/Stats: `scikit-learn`, `tsfresh`, `prophet`  
- **Clustering Algorithms**: KMeans, DBSCAN  
- **Streamlit** – interactive web app  
- **Data Formats**: CSV, JSON  

---

## 📂 Project Structure  

