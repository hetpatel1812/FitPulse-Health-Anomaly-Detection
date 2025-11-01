# Milestone 3: Anomaly Detection - Clean Modern UI
# Detecting Unusual Health Patterns Using Fitness Watch Data

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.signal import find_peaks
import json

# ============================================================================
# ANOMALY DETECTION METHODS
# ============================================================================

class ThresholdAnomalyDetector:
    """Rule-based anomaly detection using configurable thresholds."""
    
    def __init__(self):
        self.threshold_rules = {
            'heart_rate': {
                'metric_name': 'heart_rate',
                'min_threshold': 40,
                'max_threshold': 120,
                'sustained_minutes': 10,
                'description': 'Heart rate outside normal resting range'
            },
            'steps': {
                'metric_name': 'step_count',
                'min_threshold': 0,
                'max_threshold': 1000,
                'sustained_minutes': 5,
                'description': 'Unrealistic step count detected'
            },
            'sleep': {
                'metric_name': 'duration_minutes',
                'min_threshold': 180,
                'max_threshold': 720,
                'sustained_minutes': 0,
                'description': 'Unusual sleep duration'
            }
        }
        self.detected_anomalies = []
    
    def detect_anomalies(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect threshold-based anomalies in the data."""
        
        report = {
            'method': 'Threshold-Based',
            'data_type': data_type,
            'total_records': len(df),
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0,
            'threshold_info': {}
        }
        
        if data_type not in self.threshold_rules:
            return df, report
        
        rule = self.threshold_rules[data_type]
        metric_col = rule['metric_name']
        
        if metric_col not in df.columns:
            return df, report
        
        df_result = df.copy()
        df_result['threshold_anomaly'] = False
        df_result['anomaly_reason'] = ''
        df_result['severity'] = 'Normal'
        
        too_high = df_result[metric_col] > rule['max_threshold']
        too_low = df_result[metric_col] < rule['min_threshold']
        
        if rule['sustained_minutes'] > 0:
            window_size = rule['sustained_minutes']
            too_high_sustained = too_high.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            too_low_sustained = too_low.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            
            df_result.loc[too_high_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_high_sustained, 'anomaly_reason'] = f'High {metric_col}'
            df_result.loc[too_high_sustained, 'severity'] = 'High'
            
            df_result.loc[too_low_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_low_sustained, 'anomaly_reason'] = f'Low {metric_col}'
            df_result.loc[too_low_sustained, 'severity'] = 'Medium'
        else:
            df_result.loc[too_high, 'threshold_anomaly'] = True
            df_result.loc[too_high, 'anomaly_reason'] = f'Excessive {metric_col}'
            df_result.loc[too_high, 'severity'] = 'Medium'
            
            df_result.loc[too_low, 'threshold_anomaly'] = True
            df_result.loc[too_low, 'anomaly_reason'] = f'Insufficient {metric_col}'
            df_result.loc[too_low, 'severity'] = 'High'
        
        anomaly_count = df_result['threshold_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['threshold_info'] = {
            'min_threshold': rule['min_threshold'],
            'max_threshold': rule['max_threshold'],
            'sustained_minutes': rule['sustained_minutes']
        }
        
        anomalies = df_result[df_result['threshold_anomaly']]
        if len(anomalies) > 0:
            self.detected_anomalies.extend(anomalies.to_dict('records'))
        
        return df_result, report


class ResidualAnomalyDetector:
    """Model-based anomaly detection using Prophet forecast residuals."""
    
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.prophet_models = {}
        self.detected_anomalies = []
    
    def detect_anomalies_from_prophet(self, df: pd.DataFrame, forecast_df: pd.DataFrame, 
                                     data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies by comparing actual vs predicted values."""
        
        report = {
            'method': 'Prophet Residual-Based',
            'data_type': data_type,
            'threshold_std': self.threshold_std,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0
        }
        
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'sleep': 'duration_minutes'
        }
        
        if data_type not in metric_columns:
            return df, report
        
        metric_col = metric_columns[data_type]
        
        if metric_col not in df.columns:
            return df, report
        
        df_result = df.copy()
        df_result = df_result.sort_values('timestamp').reset_index(drop=True)
        
        forecast_aligned = forecast_df.copy()
        forecast_aligned = forecast_aligned.rename(columns={'ds': 'timestamp', 'yhat': 'predicted'})
        
        df_result = df_result.merge(
            forecast_aligned[['timestamp', 'predicted', 'yhat_lower', 'yhat_upper']], 
            on='timestamp', 
            how='left'
        )
        
        df_result['residual'] = df_result[metric_col] - df_result['predicted']
        
        residual_mean = df_result['residual'].mean()
        residual_std = df_result['residual'].std()
        
        threshold = self.threshold_std * residual_std
        df_result['residual_anomaly'] = np.abs(df_result['residual']) > threshold
        
        outside_interval = (df_result[metric_col] > df_result['yhat_upper']) | \
                          (df_result[metric_col] < df_result['yhat_lower'])
        
        df_result['residual_anomaly'] = df_result['residual_anomaly'] | outside_interval
        df_result['residual_anomaly_reason'] = ''
        df_result.loc[df_result['residual_anomaly'], 'residual_anomaly_reason'] = 'Deviates from predicted trend'
        
        anomaly_count = df_result['residual_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['residual_stats'] = {
            'mean': float(residual_mean),
            'std': float(residual_std),
            'threshold': float(threshold)
        }
        
        if anomaly_count > 0:
            self.detected_anomalies.extend(
                df_result[df_result['residual_anomaly']].to_dict('records')
            )
        
        return df_result, report


class ClusterAnomalyDetector:
    """Cluster-based anomaly detection."""
    
    def __init__(self):
        self.detected_anomalies = []
        self.cluster_info = {}
    
    def detect_cluster_outliers(self, feature_matrix: pd.DataFrame, 
                               cluster_labels: np.ndarray,
                               data_type: str,
                               outlier_threshold: float = 0.05) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies based on cluster membership."""
        
        report = {
            'method': 'Cluster-Based',
            'data_type': data_type,
            'total_clusters': 0,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0
        }
        
        df_result = feature_matrix.copy()
        df_result['cluster'] = cluster_labels
        
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        total_points = len(cluster_labels)
        
        anomalous_clusters = []
        
        for cluster_id, size in cluster_sizes.items():
            cluster_percentage = size / total_points
            
            if cluster_id == -1:
                anomalous_clusters.append(cluster_id)
            elif cluster_percentage < outlier_threshold:
                anomalous_clusters.append(cluster_id)
        
        df_result['cluster_anomaly'] = df_result['cluster'].isin(anomalous_clusters)
        df_result['cluster_anomaly_reason'] = ''
        
        for cluster_id in anomalous_clusters:
            if cluster_id == -1:
                reason = 'Noise point (DBSCAN)'
            else:
                reason = f'Belongs to small cluster #{cluster_id}'
            
            mask = df_result['cluster'] == cluster_id
            df_result.loc[mask, 'cluster_anomaly_reason'] = reason
        
        anomaly_count = df_result['cluster_anomaly'].sum()
        report['total_clusters'] = int(len(cluster_sizes))
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['cluster_distribution'] = cluster_sizes.to_dict()
        report['anomalous_clusters'] = [int(c) for c in anomalous_clusters]
        
        if anomaly_count > 0:
            self.detected_anomalies.extend(
                df_result[df_result['cluster_anomaly']].to_dict('records')
            )
        
        self.cluster_info = report
        
        return df_result, report


# ============================================================================
# ANOMALY VISUALIZATION
# ============================================================================

class AnomalyVisualizer:
    """Creates interactive visualizations highlighting detected anomalies."""
    
    def __init__(self):
        self.color_scheme = {
            'normal': '#1f77b4',
            'threshold': '#ff7f0e',
            'residual': '#d62728',
            'cluster': '#9467bd',
            'combined': '#e377c2'
        }
    
    def plot_heart_rate_anomalies(self, df: pd.DataFrame, title: str = "Heart Rate Anomaly Detection"):
        """Plot heart rate data with all detected anomalies highlighted."""
        
        fig = go.Figure()
        
        has_threshold = 'threshold_anomaly' in df.columns
        has_residual = 'residual_anomaly' in df.columns
        has_predicted = 'predicted' in df.columns
        
        normal_data = df.copy()
        if has_threshold:
            normal_data = normal_data[~normal_data['threshold_anomaly']]
        
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['heart_rate'],
            mode='lines',
            name='Normal Heart Rate',
            line=dict(color=self.color_scheme['normal'], width=2),
            hovertemplate='<b>Time:</b> %{x}<br><b>HR:</b> %{y} bpm<extra></extra>'
        ))
        
        if has_predicted:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['predicted'],
                mode='lines',
                name='Predicted Trend',
                line=dict(color='lightgreen', width=2, dash='dash'),
                opacity=0.7,
                hovertemplate='<b>Time:</b> %{x}<br><b>Predicted:</b> %{y:.1f} bpm<extra></extra>'
            ))
            
            if 'yhat_upper' in df.columns and 'yhat_lower' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(144, 238, 144, 0.2)',
                    line=dict(width=0),
                    name='Confidence Interval',
                    hoverinfo='skip'
                ))
        
        if has_threshold:
            threshold_anomalies = df[df['threshold_anomaly']]
            if len(threshold_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=threshold_anomalies['timestamp'],
                    y=threshold_anomalies['heart_rate'],
                    mode='markers',
                    name='Threshold Anomalies',
                    marker=dict(
                        color=self.color_scheme['threshold'],
                        size=12,
                        symbol='x',
                        line=dict(width=2)
                    ),
                    hovertemplate='<b>⚠️ Threshold Anomaly</b><br>Time: %{x}<br>HR: %{y} bpm<extra></extra>'
                ))
        
        if has_residual:
            residual_anomalies = df[df['residual_anomaly']]
            if len(residual_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=residual_anomalies['timestamp'],
                    y=residual_anomalies['heart_rate'],
                    mode='markers',
                    name='Model Deviation',
                    marker=dict(
                        color=self.color_scheme['residual'],
                        size=14,
                        symbol='diamond',
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate='<b>🔴 Model Deviation</b><br>Time: %{x}<br>HR: %{y} bpm<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50')),
            xaxis_title="Time",
            yaxis_title="Heart Rate (bpm)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
        )
        
        return fig
    
    def plot_steps_anomalies(self, df: pd.DataFrame, title: str = "Step Count Anomaly Detection"):
        """Plot step count data with anomalies."""
        
        fig = go.Figure()
        
        has_threshold = 'threshold_anomaly' in df.columns
        has_predicted = 'predicted' in df.columns
        
        normal_data = df.copy()
        if has_threshold:
            normal_data = normal_data[~normal_data['threshold_anomaly']]
        
        fig.add_trace(go.Bar(
            x=normal_data['timestamp'],
            y=normal_data['step_count'],
            name='Normal Step Count',
            marker_color=self.color_scheme['normal'],
            opacity=0.7,
            hovertemplate='<b>Time:</b> %{x}<br><b>Steps:</b> %{y}<extra></extra>'
        ))
        
        if has_predicted:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['predicted'],
                mode='lines',
                name='Predicted Trend',
                line=dict(color='orange', width=3, dash='dot'),
                hovertemplate='<b>Predicted:</b> %{y:.0f} steps<extra></extra>'
            ))
        
        if has_threshold:
            anomalies = df[df['threshold_anomaly']]
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=anomalies['timestamp'],
                    y=anomalies['step_count'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='red',
                        size=16,
                        symbol='star',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='<b>⚠️ Anomaly</b><br>Time: %{x}<br>Steps: %{y}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50')),
            xaxis_title="Time",
            yaxis_title="Step Count",
            hovermode='x unified',
            height=500,
            showlegend=True,
            barmode='overlay',
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='white'
        )
        
        return fig
    
    def plot_sleep_anomalies(self, df: pd.DataFrame, title: str = "Sleep Pattern Anomaly Detection"):
        """Plot sleep duration data with anomalies."""
        
        fig = go.Figure()
        
        has_threshold = 'threshold_anomaly' in df.columns
        
        df['duration_hours'] = df['duration_minutes'] / 60
        
        normal_data = df.copy()
        if has_threshold:
            normal_data = normal_data[~normal_data['threshold_anomaly']]
        
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['duration_hours'],
            mode='lines+markers',
            name='Normal Sleep Duration',
            line=dict(color=self.color_scheme['normal'], width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate='<b>Date:</b> %{x}<br><b>Sleep:</b> %{y:.1f} hours<extra></extra>'
        ))
        
        fig.add_hline(y=7, line_dash="dash", line_color="green", 
                     annotation_text="Recommended (7h)", annotation_position="right")
        fig.add_hline(y=3, line_dash="dash", line_color="red", 
                     annotation_text="Minimum (3h)", annotation_position="right")
        fig.add_hline(y=12, line_dash="dash", line_color="red", 
                     annotation_text="Maximum (12h)", annotation_position="right")
        
        if has_threshold:
            anomalies = df[df['threshold_anomaly']]
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=anomalies['timestamp'],
                    y=anomalies['duration_hours'],
                    mode='markers',
                    name='Sleep Anomalies',
                    marker=dict(
                        color='red',
                        size=16,
                        symbol='circle-open',
                        line=dict(width=3)
                    ),
                    hovertemplate='<b>⚠️ Unusual Sleep</b><br>Date: %{x}<br>Duration: %{y:.1f} hours<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50')),
            xaxis_title="Date",
            yaxis_title="Sleep Duration (hours)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            yaxis=dict(range=[0, 14]),
            plot_bgcolor='rgba(250,250,250,1)',
            paper_bgcolor='white'
        )
        
        return fig


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================

def create_sample_data_with_anomalies() -> Dict[str, pd.DataFrame]:
    """Create realistic sample data with intentional anomalies for testing."""
    
    timestamps = pd.date_range(start='2024-01-15 08:00:00', 
                               end='2024-01-15 20:00:00', freq='1min')
    
    base_hr = 70
    hr_data = []
    
    for i, ts in enumerate(timestamps):
        time_of_day = ts.hour + ts.minute / 60
        hr = base_hr
        
        if 9 <= time_of_day < 10:
            hr = 110 + np.random.normal(0, 5)
        elif 14 <= time_of_day < 15:
            hr = 95 + np.random.normal(0, 5)
        else:
            hr = 70 + np.random.normal(0, 3)
        
        if 11.5 <= time_of_day < 12:
            hr = 135 + np.random.normal(0, 5)
        
        if 16 <= time_of_day < 16.3:
            hr = 35 + np.random.normal(0, 2)
        
        if 18.5 <= time_of_day < 18.6:
            hr = 150
        
        hr_data.append(max(30, min(220, hr)))
    
    heart_rate_df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': hr_data
    })
    
    step_timestamps = pd.date_range(start='2024-01-15 08:00:00',
                                   end='2024-01-15 20:00:00', freq='5min')
    
    step_data = []
    for i, ts in enumerate(step_timestamps):
        time_of_day = ts.hour + ts.minute / 60
        
        if 8 <= time_of_day < 9:
            steps = 50 + np.random.randint(-10, 10)
        elif 12 <= time_of_day < 13:
            steps = 80 + np.random.randint(-15, 15)
        elif 17 <= time_of_day < 18:
            steps = 100 + np.random.randint(-20, 20)
        else:
            steps = 20 + np.random.randint(-5, 5)
        
        if 15 <= time_of_day < 15.2:
            steps = 1200
        
        step_data.append(max(0, steps))
    
    steps_df = pd.DataFrame({
        'timestamp': step_timestamps,
        'step_count': step_data
    })
    
    return {
        'heart_rate': heart_rate_df,
        'steps': steps_df
    }


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Health Anomaly Detection",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for modern look
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stMetric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stMetric label {
            color: rgba(255,255,255,0.9) !important;
            font-weight: 600 !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            color: white !important;
            font-size: 2rem !important;
        }
        .stMetric [data-testid="stMetricDelta"] {
            color: rgba(255,255,255,0.8) !important;
        }
        h1 {
            color: #1a202c;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        h2, h3 {
            color: #2d3748;
            font-weight: 600;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            margin: 10px 0;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
            transform: translateY(-2px);
        }
        .stSelectbox, .stNumberInput {
            background-color: white;
            border-radius: 10px;
        }
        div[data-testid="stDataFrame"] {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .stDownloadButton>button {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
        }
        .stInfo {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            border-radius: 8px;
        }
        .stSuccess {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            border-radius: 8px;
        }
        hr {
            margin: 2rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🏥 Health Anomaly Detection System")
        st.markdown("**Advanced monitoring and detection of unusual health patterns**")
    with col2:
        st.image("https://via.placeholder.com/150x150.png?text=Health+Monitor", width=150)
    
    st.markdown("---")
    
    # Configuration Section
    st.subheader("⚙️ Detection Configuration")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        threshold_std = st.number_input(
            "Residual Threshold (σ)",
            min_value=1.0, max_value=5.0, value=3.0, step=0.5,
            help="Standard deviation threshold for anomaly detection"
        )
    
    with col2:
        use_sample = st.selectbox(
            "Data Source",
            ["Sample Data with Anomalies", "Upload Custom Data"],
            index=0
        )
    
    with col3:
        detection_sensitivity = st.selectbox(
            "Sensitivity Level",
            ["Low", "Medium", "High"],
            index=1
        )
    
    with col4:
        st.write("")
        st.write("")
        run_button = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # Initialize detectors
    if 'threshold_detector' not in st.session_state:
        st.session_state.threshold_detector = ThresholdAnomalyDetector()
        st.session_state.residual_detector = ResidualAnomalyDetector(threshold_std)
        st.session_state.visualizer = AnomalyVisualizer()
    
    # Main execution
    if run_button:
        with st.spinner("🔍 Analyzing health data..."):
            
            # Generate sample data
            preprocessed_data = create_sample_data_with_anomalies()
            
            # Results containers
            all_results = {}
            all_reports = {}
            
            # Process each data type
            for data_type, df in preprocessed_data.items():
                
                # Detection
                df_with_anomalies, threshold_report = st.session_state.threshold_detector.detect_anomalies(
                    df, data_type
                )
                
                all_results[data_type] = df_with_anomalies
                all_reports[data_type] = {'threshold': threshold_report}
            
            # Display Results
            st.success("✅ Analysis Complete!")
            
            # Metrics Dashboard
            st.subheader("📊 Detection Summary")
            
            total_anomalies = sum(
                reports['threshold']['anomalies_detected'] 
                for reports in all_reports.values()
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Anomalies",
                    total_anomalies,
                    delta="Detected",
                    delta_color="inverse"
                )
            
            with col2:
                hr_anomalies = all_reports.get('heart_rate', {}).get('threshold', {}).get('anomalies_detected', 0)
                st.metric(
                    "Heart Rate Issues",
                    hr_anomalies,
                    delta=f"{all_reports.get('heart_rate', {}).get('threshold', {}).get('anomaly_percentage', 0):.1f}%"
                )
            
            with col3:
                step_anomalies = all_reports.get('steps', {}).get('threshold', {}).get('anomalies_detected', 0)
                st.metric(
                    "Step Count Issues",
                    step_anomalies,
                    delta=f"{all_reports.get('steps', {}).get('threshold', {}).get('anomaly_percentage', 0):.1f}%"
                )
            
            with col4:
                st.metric(
                    "Data Types Analyzed",
                    len(all_reports),
                    delta="Complete"
                )
            
            st.markdown("---")
            
            # Visualizations
            st.subheader("📈 Anomaly Visualizations")
            
            # Heart Rate
            if 'heart_rate' in all_results:
                st.markdown("### ❤️ Heart Rate Analysis")
                fig_hr = st.session_state.visualizer.plot_heart_rate_anomalies(
                    all_results['heart_rate']
                )
                st.plotly_chart(fig_hr, use_container_width=True)
                
                # Details
                hr_report = all_reports['heart_rate']['threshold']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Min Threshold:** {hr_report['threshold_info']['min_threshold']} bpm")
                with col2:
                    st.info(f"**Max Threshold:** {hr_report['threshold_info']['max_threshold']} bpm")
                with col3:
                    st.info(f"**Sustained Duration:** {hr_report['threshold_info']['sustained_minutes']} min")
            
            st.markdown("---")
            
            # Steps
            if 'steps' in all_results:
                st.markdown("### 👣 Step Count Analysis")
                fig_steps = st.session_state.visualizer.plot_steps_anomalies(
                    all_results['steps']
                )
                st.plotly_chart(fig_steps, use_container_width=True)
                
                # Details
                step_report = all_reports['steps']['threshold']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Total Records:** {step_report['total_records']}")
                with col2:
                    st.info(f"**Anomalies Found:** {step_report['anomalies_detected']}")
                with col3:
                    st.info(f"**Anomaly Rate:** {step_report['anomaly_percentage']:.2f}%")
            
            st.markdown("---")
            
            # Detailed Anomaly Table
            st.subheader("📋 Anomaly Details")
            
            for data_type, df in all_results.items():
                if 'threshold_anomaly' in df.columns:
                    anomalies = df[df['threshold_anomaly']].copy()
                    
                    if len(anomalies) > 0:
                        st.markdown(f"#### {data_type.replace('_', ' ').title()} Anomalies")
                        
                        # Prepare display data
                        display_df = anomalies[['timestamp', anomalies.columns[1], 'anomaly_reason', 'severity']].copy()
                        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "timestamp": st.column_config.TextColumn("Time", width="medium"),
                                "severity": st.column_config.TextColumn(
                                    "Severity",
                                    width="small",
                                )
                            }
                        )
            
            st.markdown("---")
            
            # Comparison Chart
            st.subheader("📊 Anomaly Distribution")
            
            # Create comparison data
            comparison_data = []
            for data_type, reports in all_reports.items():
                comparison_data.append({
                    'Data Type': data_type.replace('_', ' ').title(),
                    'Anomalies': reports['threshold']['anomalies_detected'],
                    'Total Records': reports['threshold']['total_records'],
                    'Percentage': reports['threshold']['anomaly_percentage']
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            fig_comparison = go.Figure()
            
            fig_comparison.add_trace(go.Bar(
                x=comp_df['Data Type'],
                y=comp_df['Anomalies'],
                name='Anomalies Detected',
                marker_color='#e74c3c',
                text=comp_df['Anomalies'],
                textposition='auto',
            ))
            
            fig_comparison.add_trace(go.Scatter(
                x=comp_df['Data Type'],
                y=comp_df['Percentage'],
                name='Anomaly Rate (%)',
                yaxis='y2',
                mode='lines+markers',
                marker=dict(size=12, color='#3498db'),
                line=dict(width=3)
            ))
            
            fig_comparison.update_layout(
                title="Anomaly Detection Comparison",
                xaxis_title="Data Type",
                yaxis_title="Number of Anomalies",
                yaxis2=dict(
                    title="Anomaly Rate (%)",
                    overlaying='y',
                    side='right'
                ),
                height=400,
                hovermode='x unified',
                showlegend=True,
                plot_bgcolor='rgba(250,250,250,1)',
                paper_bgcolor='white'
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            st.markdown("---")
            
            # Export Section
            st.subheader("💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON Report
                report_json = json.dumps(all_reports, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=report_json,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # CSV Export
                all_anomalies_list = []
                for data_type, df in all_results.items():
                    if 'threshold_anomaly' in df.columns:
                        anomalies = df[df['threshold_anomaly']].copy()
                        anomalies['data_type'] = data_type
                        all_anomalies_list.append(anomalies)
                
                if all_anomalies_list:
                    combined_df = pd.concat(all_anomalies_list, ignore_index=True)
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV Data",
                        data=csv,
                        file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col3:
                # Summary Report
                summary_text = f"""ANOMALY DETECTION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
=======
Total Anomalies Detected: {total_anomalies}
Data Types Analyzed: {len(all_reports)}
Detection Method: Threshold-Based
Sensitivity: {detection_sensitivity}

BREAKDOWN BY DATA TYPE
=====================
"""
                for data_type, reports in all_reports.items():
                    rep = reports['threshold']
                    summary_text += f"\n{data_type.upper()}:\n"
                    summary_text += f"  - Total Records: {rep['total_records']}\n"
                    summary_text += f"  - Anomalies: {rep['anomalies_detected']}\n"
                    summary_text += f"  - Rate: {rep['anomaly_percentage']:.2f}%\n"
                
                st.download_button(
                    label="📝 Download Summary",
                    data=summary_text,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            st.markdown("---")
            
           
            # Store results in session state
            st.session_state.analysis_results = all_results
            st.session_state.analysis_reports = all_reports
            
            st.balloons()
    
   

if __name__ == "__main__":
    main()