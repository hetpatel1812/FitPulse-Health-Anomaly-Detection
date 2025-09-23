import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# -------------------------------
# Page Configuration & Styling
# -------------------------------
def apply_custom_styling():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;700&family=Roboto:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', 'Roboto', sans-serif !important;
        background: linear-gradient(135deg, #111215 0%, #232526 100%) !important;
    }
    .stApp {
        background: linear-gradient(135deg, #111215 0%, #232526 100%) !important;
    }
    /* Remove all default Streamlit containers and borders */
    .block-container, .css-1y4p8pa, .css-12oz5g7, .css-1d391kg, .element-container, .stMarkdown, .uploadedFile {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding-top: 0;
        padding-bottom: 0;
    }
    /* Hide Streamlit default header/footer/menu */
    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
    /* Remove all white backgrounds */
    div[data-testid="stAppViewContainer"], 
    div[data-testid="stHeader"],
    section[data-testid="stSidebar"] > div,
    .css-1lcbmhc,
    .css-1y0tads {
        background: transparent !important;
    }
    /* Modern glassmorphism card */
    .modern-card {
        background: rgba(30, 32, 36, 0.96);
        border-radius: 22px;
        padding: 2.2rem 2rem 2rem 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.22);
        border: 1px solid rgba(255,255,255,0.06);
        color: #f4f6fa;
        position: relative;
        overflow: hidden;
        animation: fadeInCard 1s cubic-bezier(.4,0,.2,1);
    }
    @keyframes fadeInCard {
        0% { opacity: 0; transform: translateY(30px);}
        100% { opacity: 1; transform: translateY(0);}
    }
    /* Animated gradient header */
    .fitpulse-header {
        background: linear-gradient(90deg, #232526, #0f2027, #232526, #0f2027);
        background-size: 400% 400%;
        animation: gradientMove 8s ease-in-out infinite;
        padding: 2.5rem 0 2rem 0;
        border-radius: 28px;
        margin-bottom: 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.22);
        border: 1px solid rgba(255,255,255,0.06);
        transition: box-shadow 0.5s;
    }
    .fitpulse-header:hover {
        box-shadow: 0 12px 40px 0 rgba(0,0,0,0.28);
    }
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .fitpulse-title {
        font-size: 3.2rem;
        font-weight: 700;
        color: #f4f6fa;
        text-shadow: 0 2px 8px rgba(0,0,0,0.18);
        margin: 0;
        letter-spacing: -1px;
        animation: fadeInDown 1.2s;
    }
    .fitpulse-subtitle {
        font-size: 1.25rem;
        color: #bfc9d1;
        margin-top: 0.7rem;
        font-weight: 400;
        animation: fadeInDown 1.5s;
    }
    @keyframes fadeInDown {
        0% { opacity: 0; transform: translateY(-30px);}
        100% { opacity: 1; transform: translateY(0);}
    }
    /* Sidebar Styling */
    .css-1d391kg, section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111215 0%, #232526 100%) !important;
        border-right: 1px solid #232526 !important;
    }
    .css-1d391kg .css-1v0mbdj, .css-1d391kg .stMarkdown, .css-1d391kg label {
        color: #e0e6ed !important;
    }
    .css-1d391kg .stSelectbox label, .css-1d391kg .stCheckbox label {
        color: #6ee7b7 !important;
        font-weight: 600;
    }
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #6ee7b7 0%, #232526 100%) !important;
        color: #111215 !important;
        border: none !important;
        border-radius: 22px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s cubic-bezier(.4,0,.2,1) !important;
        box-shadow: 0 4px 18px rgba(110,231,183,0.13) !important;
        animation: pulseBtn 2s infinite;
    }
    @keyframes pulseBtn {
        0% { box-shadow: 0 0 0 0 rgba(110,231,183,0.2);}
        70% { box-shadow: 0 0 0 10px rgba(110,231,183,0);}
        100% { box-shadow: 0 0 0 0 rgba(110,231,183,0);}
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.03) !important;
        box-shadow: 0 8px 32px rgba(110,231,183,0.18) !important;
        background: linear-gradient(90deg, #232526 0%, #6ee7b7 100%) !important;
        color: #f4f6fa !important;
    }
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #232526 0%, #6ee7b7 100%) !important;
        color: #f4f6fa !important;
        border: none !important;
        border-radius: 22px !important;
        padding: 0.8rem 2.2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        margin-top: 0.5rem;
        animation: fadeInCard 1.2s;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #6ee7b7 0%, #232526 100%) !important;
        color: #111215 !important;
    }
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        background: rgba(30, 32, 36, 0.97) !important;
        color: #f4f6fa !important;
        border: 1px solid #6ee7b7 !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
    }
    .stSelectbox label {
        color: #6ee7b7 !important;
        font-weight: 600 !important;
    }
    .stSelectbox [data-baseweb="select"] input {
        color: #f4f6fa !important;
    }
    .stSelectbox [data-baseweb="select"] svg {
        color: #6ee7b7 !important;
    }
    .stSelectbox [data-baseweb="menu"] {
        color: #f4f6fa !important;
        background: #111215 !important;
    }
    .stSelectbox [data-baseweb="option"] {
        color: #f4f6fa !important;
        background: #111215 !important;
    }
    .stSelectbox [data-baseweb="option"]:hover {
        background: #6ee7b7 !important;
        color: #111215 !important;
    }
    /* Checkbox styling */
    .stCheckbox > label {
        color: #6ee7b7 !important;
        font-weight: 600 !important;
    }
    /* Data Table Styling */
    .dataframe {
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 25px rgba(30,32,36,0.13) !important;
        background: rgba(30,32,36,0.96) !important;
        color: #f4f6fa !important;
        animation: fadeInCard 1.2s;
    }
    .dataframe th {
        background: #232526 !important;
        color: #6ee7b7 !important;
        font-weight: 700 !important;
    }
    .dataframe td {
        background: rgba(30,32,36,0.96) !important;
        color: #f4f6fa !important;
    }
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6ee7b7 0%, #232526 100%) !important;
    }
    /* Modern metric display */
    .modern-metric {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(110,231,183,0.10);
        border-radius: 18px;
        padding: 1.2rem 0.8rem;
        margin: 0.5rem 0;
        min-width: 120px;
        min-height: 80px;
        box-shadow: 0 2px 12px rgba(110,231,183,0.08);
        animation: fadeInCard 1.2s cubic-bezier(.4,0,.2,1);
        transition: box-shadow 0.4s;
    }
    .modern-metric:hover {
        box-shadow: 0 6px 24px rgba(110,231,183,0.18);
    }
    .modern-metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #6ee7b7;
        margin-bottom: 0.2rem;
        animation: popIn 0.8s;
    }
    @keyframes popIn {
        0% { transform: scale(0.7);}
        80% { transform: scale(1.1);}
        100% { transform: scale(1);}
    }
    .modern-metric-label {
        font-size: 1rem;
        color: #bfc9d1;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    /* Animations for fade in */
    @keyframes fadeInCard {
        0% { opacity: 0; transform: translateY(30px);}
        100% { opacity: 1; transform: translateY(0);}
    }
    </style>
    """, unsafe_allow_html=True)

def show_header():
    st.markdown("""
    <div class="fitpulse-header">
        <h1 class="fitpulse-title">FitPulse</h1>
        <p class="fitpulse-subtitle">Transform Your Fitness Data Into Actionable Insights</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Configuration
# -------------------------------
EXPECTED_COLUMNS = {
    'heart_rate': {'min': 30, 'max': 250, 'unit': 'bpm', 'color': '#7fd7d7', 'icon': '💓'},
    'step_count': {'min': 0, 'max': 50000, 'unit': 'steps', 'color': '#bfc9d1', 'icon': '👟'},
    'duration_minutes': {'min': 0, 'max': 1440, 'unit': 'min', 'color': '#a3b18a', 'icon': '⏱️'},
    'calories': {'min': 0, 'max': 10000, 'unit': 'kcal', 'color': '#e9c46a', 'icon': '🔥'},
    'distance': {'min': 0, 'max': 100, 'unit': 'km', 'color': '#8ecae6', 'icon': '📏'}
}

# -------------------------------
# Data Loading with Enhanced UI
# -------------------------------
def load_uploaded_files():
    with st.container():
        st.markdown("### 📁 Upload Your Fitness Data")
        st.markdown("*Supported formats: CSV, JSON*")
        uploaded_file = st.file_uploader(
            "", 
            type=['csv', 'json'], 
            accept_multiple_files=False,
            help="Upload your fitness tracking data from smartwatches, fitness apps, or manual logs"
        )
        if uploaded_file:
            try:
                file_size_kb = uploaded_file.size / 1024  # Show in KB
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    data = json.load(uploaded_file)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
                # Modern metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="modern-metric">
                        <span class="modern-metric-value">📄</span>
                        <span class="modern-metric-label">{uploaded_file.name}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="modern-metric">
                        <span class="modern-metric-value">{file_size_kb:.1f} KB</span>
                        <span class="modern-metric-label">File Size</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="modern-metric">
                        <span class="modern-metric-value">{len(df)}</span>
                        <span class="modern-metric-label">Records</span>
                    </div>
                    """, unsafe_allow_html=True)
                return df
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                return pd.DataFrame()
    return pd.DataFrame()

# -------------------------------
# Enhanced Validation & Cleaning
# -------------------------------
def validate_and_clean(df: pd.DataFrame):
    if df.empty:
        return df, {}
    st.markdown("### 🔧 Data Processing Pipeline")
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    # Step 1: Standardize columns
    status_text.text('🔄 Standardizing column names...')
    progress_bar.progress(20)
    df = df.rename(columns={
        "datetime": "timestamp", 
        "sleep_duration": "duration_minutes",
        "heart": "heart_rate",
        "steps": "step_count"
    })
    df.columns = df.columns.str.lower()
    # Step 2: Parse timestamps
    status_text.text('⏰ Processing timestamps...')
    progress_bar.progress(40)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
    # Step 3: Clean numeric columns
    status_text.text('🧮 Cleaning numeric data...')
    progress_bar.progress(60)
    cleaning_stats = {}
    for col in EXPECTED_COLUMNS.keys():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            config = EXPECTED_COLUMNS[col]
            df[col] = df[col].clip(lower=config['min'], upper=config['max'])
            df[col] = df[col].fillna(0)
            cleaning_stats[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean()
            }
    # Step 4: Final validation
    status_text.text('✅ Finalizing data...')
    progress_bar.progress(100)
    st.markdown('</div>', unsafe_allow_html=True)
    return df, cleaning_stats

# -------------------------------
# Enhanced Resampling
# -------------------------------
def resample_data(df, freq="1H", method="interpolate"):
    if df.empty or "timestamp" not in df.columns:
        return df
    st.markdown("### ⚡ Data Resampling")
    original_points = len(df)
    df_resampled = df.set_index("timestamp").resample(freq).mean()
    # Apply fill method
    if method == "interpolate":
        df_resampled = df_resampled.interpolate(method='time').bfill().ffill()
    elif method == "forward_fill":
        df_resampled = df_resampled.ffill()
    elif method == "backward_fill":
        df_resampled = df_resampled.bfill()
    elif method == "zero":
        df_resampled = df_resampled.fillna(0)
    elif method == "drop":
        df_resampled = df_resampled.dropna()
    df_final = df_resampled.reset_index()
    # Modern metrics row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="modern-metric">
            <span class="modern-metric-value">{original_points}</span>
            <span class="modern-metric-label">Original Points</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="modern-metric">
            <span class="modern-metric-value">{len(df_final)}</span>
            <span class="modern-metric-label">Resampled Points</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return df_final

# -------------------------------
# Enhanced Visualization
# -------------------------------
def plot_data(df, cleaning_stats):
    if df.empty:
        st.warning("⚠️ No data available for visualization")
        return
    st.markdown("### 📊 Data Insights Dashboard")
    # Data overview
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="modern-metric">
                <span class="modern-metric-value">{len(df):,}</span>
                <span class="modern-metric-label">Total Records</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if 'timestamp' in df.columns:
                time_span = (df['timestamp'].max() - df['timestamp'].min()).days
                st.markdown(f"""
                <div class="modern-metric">
                    <span class="modern-metric-value">{time_span}</span>
                    <span class="modern-metric-label">Days Tracked</span>
                </div>
                """, unsafe_allow_html=True)
        with col3:
            available_metrics = len([col for col in df.columns if col in EXPECTED_COLUMNS])
            st.markdown(f"""
            <div class="modern-metric">
                <span class="modern-metric-value">{available_metrics}</span>
                <span class="modern-metric-label">Metrics</span>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.markdown(f"""
            <div class="modern-metric">
                <span class="modern-metric-value">{completeness:.1f}%</span>
                <span class="modern-metric-label">Data Quality</span>
            </div>
            """, unsafe_allow_html=True)
    # Interactive visualization
    st.markdown("#### 📈 Interactive Visualization")
    available_metrics = [col for col in df.columns if col in EXPECTED_COLUMNS and col in df.columns]
    if available_metrics:
        selected_metric = st.selectbox(
            "Choose a metric to visualize:",
            available_metrics,
            format_func=lambda x: f"{EXPECTED_COLUMNS[x]['icon']} {x.replace('_', ' ').title()} ({EXPECTED_COLUMNS[x]['unit']})"
        )
        if selected_metric and 'timestamp' in df.columns and not df[selected_metric].isna().all():
            fig = go.Figure()
            color = EXPECTED_COLUMNS[selected_metric]['color']
            plot_df = df.dropna(subset=[selected_metric, 'timestamp'])
            if not plot_df.empty:
                fig.add_trace(go.Scatter(
                    x=plot_df['timestamp'],
                    y=plot_df[selected_metric],
                    mode='lines+markers',
                    name=selected_metric.replace('_', ' ').title(),
                    line=dict(color=color, width=3),
                    marker=dict(color=color, size=6),
                    hovertemplate=(
                        f'<b>{selected_metric.replace("_", " ").title()}</b><br>' +
                        'Time: %{x}<br>' +
                        f'Value: %{{y}} {EXPECTED_COLUMNS[selected_metric]["unit"]}<br>' +
                        '<extra></extra>'
                    )
                ))
                fig.update_layout(
                    title=f"{EXPECTED_COLUMNS[selected_metric]['icon']} {selected_metric.replace('_', ' ').title()} Trends",
                    xaxis_title="Time",
                    yaxis_title=f"{selected_metric.replace('_', ' ').title()} ({EXPECTED_COLUMNS[selected_metric]['unit']})",
                    template="plotly_white",
                    height=500,
                    hovermode='x unified',
                    showlegend=False,
                    plot_bgcolor='rgba(44,62,80,0.92)',
                    paper_bgcolor='rgba(44,62,80,0.92)',
                    font=dict(family="Inter, sans-serif", size=13, color="#f4f6fa"),
                    title_font=dict(size=17, color='#7fd7d7')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ No valid data available for {selected_metric}")
        else:
            st.warning("⚠️ No data available for the selected metric")
    else:
        st.info("ℹ️ No recognized fitness metrics found in the data")

    # ----------- Anomaly Detection Section -----------
    st.markdown("#### 🚨 Anomaly Data")
    if available_metrics and 'timestamp' in df.columns:
        # Show anomaly ranges for all metrics
        st.markdown("<b>Anomaly Ranges (IQR method):</b>", unsafe_allow_html=True)
        anomaly_ranges = []
        for metric in available_metrics:
            metric_data = df[metric].dropna()
            if not metric_data.empty:
                q1 = metric_data.quantile(0.25)
                q3 = metric_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                anomaly_ranges.append(
                    f"<span style='color:#6ee7b7;font-weight:600'>{EXPECTED_COLUMNS[metric]['icon']} {metric.replace('_',' ').title()}</span>: "
                    f"<span style='color:#f4f6fa;'>[{lower_bound:.1f} - {upper_bound:.1f}] {EXPECTED_COLUMNS[metric]['unit']}</span>"
                )
        st.markdown("<br>".join(anomaly_ranges), unsafe_allow_html=True)
        st.markdown("---")

        selected_anom_metric = st.selectbox(
            "Select metric for anomaly detection:",
            available_metrics,
            format_func=lambda x: f"{EXPECTED_COLUMNS[x]['icon']} {x.replace('_', ' ').title()}",
            key="anomaly_metric"
        )
        metric_data = df[[selected_anom_metric, 'timestamp']].dropna()
        if not metric_data.empty:
            # Simple anomaly detection: values outside 1.5*IQR
            q1 = metric_data[selected_anom_metric].quantile(0.25)
            q3 = metric_data[selected_anom_metric].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            anomalies = metric_data[
                (metric_data[selected_anom_metric] < lower_bound) |
                (metric_data[selected_anom_metric] > upper_bound)
            ]
            st.markdown(
                f"<div style='margin-bottom:0.7rem; color:#6ee7b7; font-weight:600;'>"
                f"Detected <span style='color:#f4f6fa;'>{len(anomalies)}</span> anomalies for <b>{selected_anom_metric.replace('_',' ').title()}</b>"
                f"</div>", unsafe_allow_html=True
            )
            if not anomalies.empty:
                st.dataframe(
                    anomalies.sort_values('timestamp').reset_index(drop=True).style.format({
                        selected_anom_metric: "{:.1f}"
                    }),
                    use_container_width=True
                )
                # Optional: plot anomalies on chart
                fig_anom = go.Figure()
                fig_anom.add_trace(go.Scatter(
                    x=metric_data['timestamp'],
                    y=metric_data[selected_anom_metric],
                    mode='lines',
                    name='Normal',
                    line=dict(color="#7fd7d7", width=2),
                    opacity=0.5
                ))
                fig_anom.add_trace(go.Scatter(
                    x=anomalies['timestamp'],
                    y=anomalies[selected_anom_metric],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color="#e76f51", size=10, symbol='x'),
                ))
                fig_anom.update_layout(
                    title=f"Anomaly Detection for {selected_anom_metric.replace('_',' ').title()}",
                    xaxis_title="Time",
                    yaxis_title=f"{selected_anom_metric.replace('_',' ').title()}",
                    template="plotly_white",
                    height=350,
                    showlegend=True,
                    plot_bgcolor='rgba(44,62,80,0.92)',
                    paper_bgcolor='rgba(44,62,80,0.92)',
                    font=dict(family="Inter, sans-serif", size=13, color="#f4f6fa"),
                    title_font=dict(size=16, color='#e76f51')
                )
                st.plotly_chart(fig_anom, use_container_width=True)
            else:
                st.info("No anomalies detected for this metric.")
        else:
            st.info("No data available for anomaly detection.")
    else:
        st.info("No metrics available for anomaly detection.")
    # ----------- End Anomaly Section -----------

    # Data preview
    st.markdown("#### 🔍 Data Preview")
    if not df.empty:
        st.dataframe(
            df.head(20).style.format({
                col: "{:.1f}" for col in df.select_dtypes(include=[np.number]).columns
            }), 
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Sidebar Configuration
# -------------------------------
def setup_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("### 📊 Resampling Settings")
        freq_map = {
            "1 minute": "1T", 
            "5 minutes": "5T", 
            "15 minutes": "15T", 
            "30 minutes": "30T", 
            "1 hour": "1H",
            "1 day": "1D"
        }
        freq_choice = st.selectbox("📈 Sampling Frequency", list(freq_map.keys()), index=4)
        st.markdown("### 🔧 Data Processing")
        fill_methods = {
            "Smart Interpolation": "interpolate",
            "Forward Fill": "forward_fill", 
            "Backward Fill": "backward_fill",
            "Fill with Zero": "zero",
            "Drop Missing": "drop"
        }
        fill_choice = st.selectbox("🔄 Missing Data Strategy", list(fill_methods.keys()), index=0)
        st.markdown("### 📋 Data Quality")
        show_stats = st.checkbox("📊 Show Processing Statistics", value=True)
        show_warnings = st.checkbox("⚠️ Show Data Warnings", value=True)
        st.markdown("---")
        st.markdown("### 💡 About FitPulse")
        st.markdown("""
        **FitPulse** helps you preprocess and analyze fitness data from:
        - Fitness apps
        - Smartwatches  
        - Activity trackers
        - Manual logs
        """)
        return freq_map[freq_choice], fill_methods[fill_choice], show_stats, show_warnings

# -------------------------------
# Main Application
# -------------------------------
def main():
    st.set_page_config(
        page_title="FitPulse - Fitness Data Preprocessor",
        page_icon="💓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_custom_styling()
    show_header()
    freq, fill_method, show_stats, show_warnings = setup_sidebar()
    with st.container():
        raw_data = load_uploaded_files()
        if not raw_data.empty:
            clean_data, cleaning_stats = validate_and_clean(raw_data)
            if not clean_data.empty:
                processed_data = resample_data(clean_data, freq=freq, method=fill_method)
                plot_data(processed_data, cleaning_stats)
                if not processed_data.empty:
                    # --- Improved Download Buttons Layout ---
                    st.markdown("""
                        <div style="display: flex; justify-content: center; gap: 2.5rem; margin: 2.5rem 0 1.5rem 0;">
                    """, unsafe_allow_html=True)
                    col2, col1 = st.columns([1,1], gap="large")  # Swapped positions
                    with col2:
                        json_data = processed_data.to_json(orient='records', date_format='iso')
                        st.download_button(
                            label="⬇️ Download as JSON",
                            data=json_data,
                            file_name=f"fitpulse_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    with col1:
                        csv_data = processed_data.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Download as CSV",
                            data=csv_data,
                            file_name=f"fitpulse_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    st.markdown("</div>", unsafe_allow_html=True)
                    # -----------------------------------------
        else:
            st.markdown("""
            <div style="margin-top:2.5rem; padding:2rem 2.5rem; border-radius:18px; background:rgba(44,62,80,0.85); box-shadow:0 8px 32px 0 rgba(31,38,135,0.10);">
            <h3 style="color:#7fd7d7; font-weight:700; margin-bottom:0.5rem;">👋 Welcome to FitPulse!</h3>
            <p style="color:#e0e6ed; font-size:1.1rem; margin-bottom:1.2rem;">
            Get started by uploading your fitness data above.<br>
            <b>FitPulse</b> supports:
            </p>
            <ul style="color:#bfc9d1; font-size:1.05rem;">
                <li>Heart rate monitoring</li>
                <li>Step counting & activity tracking</li>
                <li>Sleep duration analysis</li>
                <li>Calorie burn tracking</li>
                <li>Distance measurements</li>
            </ul>
            <p style="color:#bfc9d1; margin-top:1.2rem;">
            <b>Processing Features:</b>
            <ul>
                <li>Intelligent data cleaning</li>
                <li>Flexible time-based resampling</li>
                <li>Multiple missing data strategies</li>
                <li>Interactive visualizations</li>
                <li>Export in multiple formats</li>
            </ul>
            <b>Pro Tips:</b>
            <ul>
                <li>Ensure your data includes timestamp columns</li>
                <li>CSV files work best with standard column names</li>
                <li>Use the sidebar to customize processing options</li>
            </ul>
            </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
