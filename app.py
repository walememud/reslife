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

        return rf_model, training_features, feature_importance, X, y
    except FileNotFoundError:
        st.error("Model file not found! Make sure trained_model_data.pkl is in the same directory.")
        return None, [], pd.DataFrame(), None, None

# Header
st.markdown("""
<div style='margin-top: -100px; margin-bottom: -20px;'></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Student Housing Needs Prediction System</h1>
    <p style="font-size:200%;">Continuing Students</p>
</div>
""", unsafe_allow_html=True)

# Load model
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
    - **Predictions**: Upload student features and get housing predictions  
    - **Interactive Dashboard**: Clean, university-branded interface  
    - **Export Results**: Download predictions as CSV  

    ### 🔧 Model Details
    - **Algorithm**: Random Forest Classifier  
    - **Training Period**: 2018-2023 CMU student features  
    - **Accuracy**: ~90%  
    - **Features**: Academic, demographic, and enrollment variables  

    ### 📱 How to Use
    - Navigate to the **Model Information** section to understand the model  
    - Go to **Make Predictions** to upload student features  
    - Upload a CSV file with student features (must include `TermPIDMKey`)  
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
    st.markdown('<h2 class="section-header">Trained Model Information</h2>', unsafe_allow_html=True)
    if rf_model is not None:
        st.markdown('<h3 class="section-header">Prediction Model Summary</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Model Type:** Random Forest")
            st.info(f"**Training Features:** {len(training_features)}")
        with col2:
            st.info("**Training Period:** 2018–2023")
            st.info("**Accuracy:** ~90%")

        # Feature Importance
        st.markdown('<h3 class="section-header">Top 15 Most Important Features</h3>', unsafe_allow_html=True)
        if not feature_importance.empty:
            top_15 = feature_importance.head(15)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_15['feature'], top_15['importance'], color='#800020')
            ax.set_xlabel("Importance")
            ax.set_title("Top 15 Features")
            plt.gca().invert_yaxis()
            st.pyplot(fig)
            st.dataframe(top_15, use_container_width=True)
    else:
        st.error("Model not available.")

# Page 3: Make Predictions
elif page == "Make Predictions":
    if rf_model is not None:
        st.success("CMU Housing Prediction Model is ready")

        uploaded = st.file_uploader("Upload CSV with student features", type=['csv'])
        if uploaded:
            df_new = pd.read_csv(uploaded, dtype={'TermPIDMKey': str})
            st.dataframe(df_new.head())

            if 'TermPIDMKey' not in df_new.columns:
                st.error("Missing required column: TermPIDMKey")
            else:
                if st.button("Run Predictions"):
                    X_new = df_new.drop('TermPIDMKey', axis=1)
                    preds = rf_model.predict(X_new)
                    probs = rf_model.predict_proba(X_new)[:, 1]

                    results = pd.DataFrame({
                        "TermPIDMKey": df_new['TermPIDMKey'],
                        "Predicted_Has_Future_Housing": preds,
                        "Probability": probs.round(3)
                    })

                    st.dataframe(results.head(10))
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "Download Predictions",
                        csv,
                        f"CMU_housing_predictions_{pd.Timestamp.now().date()}.csv",
                        "text/csv"
                    )
        else:
            st.info("Upload a CSV file to make predictions")
    else:
        st.error("Model not available")
