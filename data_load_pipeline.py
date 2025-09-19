import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Uploader
# -------------------------------
def load_uploaded_files():
    st.header("📁 Upload Fitness Data")
    uploaded_files = st.file_uploader(
        "Choose fitness data file (CSV or JSON)", type=['csv', 'json'], accept_multiple_files=False
    )

    if uploaded_files:
        try:
            if uploaded_files.name.endswith(".csv"):
                df = pd.read_csv(uploaded_files)
            else:
                df = pd.DataFrame(json.load(uploaded_files))
            st.success(f"✅ Loaded {uploaded_files.name} ({len(df)} rows)")
            return df
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# -------------------------------
# Validation & Cleaning
# -------------------------------
def validate_and_clean(df: pd.DataFrame):
    if df.empty:
        return df

    # Standardize column names
    df = df.rename(columns={"datetime": "timestamp", "sleep_duration": "duration_minutes"})
    df.columns = df.columns.str.lower()

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Clean numeric columns
    for col in ["heart_rate", "step_count", "duration_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    return df

# -------------------------------
# Resampling
# -------------------------------
def resample_data(df, freq="1H", method="interpolate"):
    if "timestamp" not in df:
        return df

    df = df.set_index("timestamp").resample(freq).mean()

    if method == "interpolate":
        df = df.interpolate().bfill().ffill()
    elif method == "forward_fill":
        df = df.ffill()
    elif method == "backward_fill":
        df = df.bfill()
    elif method == "zero":
        df = df.fillna(0)
    elif method == "drop":
        df = df.dropna()

    return df.reset_index()

# -------------------------------
# Visualization
# -------------------------------
def plot_data(df):
    if df.empty:
        st.warning("No data to plot.")
        return

    st.subheader("📊 Data Preview")
    st.dataframe(df.head(20))

    metrics = [c for c in df.columns if c != "timestamp"]
    if metrics:
        choice = st.selectbox("Select metric to visualize", metrics)
        fig = go.Figure(go.Scatter(x=df["timestamp"], y=df[choice], mode="lines+markers"))
        fig.update_layout(title=f"{choice} over time", xaxis_title="Time", yaxis_title=choice)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Main App
# -------------------------------
def main():
    st.set_page_config(page_title="Fitness Preprocessor", page_icon="🏃", layout="wide")
    st.title("🏃 Fitness Data Preprocessor (Compatible with csv/json)")

    # Sidebar controls
    freq_map = {"1min": "1T", "5min": "5T", "15min": "15T", "30min": "30T", "1hour": "1H"}
    freq_choice = st.sidebar.selectbox("Resample Frequency", list(freq_map.keys()), index=4)
    fill_choice = st.sidebar.selectbox(
        "Missing Value Fill", ["interpolate", "forward_fill", "backward_fill", "zero", "drop"], index=0
    )

    # Pipeline
    raw = load_uploaded_files()
    if not raw.empty:
        clean = validate_and_clean(raw)
        aligned = resample_data(clean, freq=freq_map[freq_choice], method=fill_choice)
        st.success("✅ Pipeline completed")
        plot_data(aligned)

if __name__ == "__main__":
    main()

