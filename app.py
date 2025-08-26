import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set pandas options to prevent scientific notation
pd.set_option('display.float_format', '{:.0f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 0)

st.set_page_config(
    page_title="CMU Housing Prediction", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Colorado Mesa University maroon theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #800020 0%, #600018 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: bold;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: #FFD700 !important;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .navbar {
        background: #800020;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .navbar-brand {
        color: white !important;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Sidebar Styling - Maroon Theme */
    .css-1d391kg {
        background: linear-gradient(180deg, #800020 0%, #600018 100%) !important;
    }
    
    .css-1l02zno {
        background: linear-gradient(180deg, #800020 0%, #600018 100%) !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #800020 0%, #600018 100%) !important;
    }
    
    [data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #800020 0%, #600018 100%) !important;
    }
    
    /* Sidebar text styling */
    .css-1d391kg .markdown-text-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .markdown-text-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
        font-weight: bold;
    }
    
    [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Sidebar selectbox styling */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white !important;
        border: 2px solid #FFD700 !important;
        border-radius: 8px !important;
        color: #800020 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label {
        color: #FFD700 !important;
        font-weight: bold !important;
    }
    
    /* Success and error messages in sidebar */
    [data-testid="stSidebar"] .stSuccess {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #FFD700 !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stError {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #FF6B6B !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-left: 4px solid #FFD700 !important;
        color: white !important;
    }
    
    [data-testid="metric-container"] {
        background: white;
        border: 2px solid #800020;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: #800020 !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #800020 0%, #A0002A 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stButton > button:hover {
        background: #600018 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    .section-header {
        color: #800020 !important;
        border-bottom: 2px solid #800020;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_trained_model():
    """Load the pre-trained Random Forest model"""
    try:
        with open('trained_model_data.pkl', 'rb') as f:
            data = pickle.load(f)
        
        rf_model = data['rf_model']
        training_features = data['training_features']
        X = data['X']
        y = data['y']
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_correlations = []
        for feature in feature_importance['feature']:
            try:
                correlation = X[feature].corr(y)
                if pd.isna(correlation):
                    correlation = 0.0
                feature_correlations.append(correlation)
            except:
                feature_correlations.append(0.0)
        
        feature_importance['correlation'] = feature_correlations
        feature_importance['effect'] = ['Positive' if corr > 0 else 'Negative' if corr < 0 else 'Neutral' for corr in feature_correlations]
        
        return rf_model, training_features, feature_importance, X, y
        
    except FileNotFoundError:
        st.error("Model file not found! Make sure trained_model_data.pkl is in the same directory.")
        return None, [], pd.DataFrame(), None, None

# Header
st.markdown("""
<div style='margin-top: -100px; margin-bottom: -20px;'>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Student Housing Needs Prediction System</h1>
     <p style="font-size:200%;">Continuing Students</p>
</div>
""", unsafe_allow_html=True)

# Load model and data
rf_model, training_features, feature_importance, X, y = load_trained_model()

# Sidebar Navigation
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image("Picture1.png", use_container_width=True)

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a section", 
    ["About", "Model Information", "Make Predictions"]
)

# Page 1: About
if page == "About":
    st.markdown('<h2 class="section-header">About</h2>', unsafe_allow_html=True)
    st.markdown("""
    This Streamlit application predicts whether continuing students will need university housing based on historical data from 2018-2023.

    ### 🚀 Features
    - **Model Information**: View Random Forest model details and feature importance  
    - **Predictions**: Upload student data and get housing predictions  
    - **Interactive Dashboard**: Clean, university-branded interface  
    - **Export Results**: Download predictions as CSV  

    ### 🔧 Model Details
    - **Algorithm**: Random Forest Classifier  
    - **Training Period**: 2018-2023 CMU student data  
    - **Accuracy**: ~90%  
    - **Features**: Academic, demographic, and enrollment data  

    ### 📱 How to Use
    - Navigate to the **Model Information** section to understand the model  
    - Go to **Make Predictions** to upload student data  
    - Upload a CSV file with student data (must include `TermPIDMKey`)  
    - Click **Run Predictions** to get results  
    - Download the predictions as CSV  

    ### 📋 Required Data Format
    Your CSV file must contain:  
    - **TermPIDMKey**: Student identifier  
    - All training features used in the model  
    - No missing required columns  

    ### 🛠️ Technical Stack
    - **Frontend**: Streamlit  
    - **ML Library**: scikit-learn  
    - **Data Processing**: pandas, numpy  
    - **Visualization**: matplotlib, seaborn  

    ---
    **Created:** 2025-08-26 Colorado Mesa University - Institutional Research
    """)

# Page 2: Model Information
elif page == "Model Information":
    # (leave your existing Model Information code here unchanged)
    st.markdown('<h2 class="section-header">Trained Model Information</h2>', unsafe_allow_html=True)
    # ...
    # existing content continues

# Page 3: Make Predictions
elif page == "Make Predictions":
    # (leave your existing Make Predictions code here unchanged)
    # ...
    pass
