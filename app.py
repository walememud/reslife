import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.ensemble import RandomForestClassifier

# Set pandas options
pd.set_option('display.float_format', '{:.0f}'.format)

st.set_page_config(
    page_title="CMU Housing Prediction", 
    page_icon="🏫", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #800020 0%, #600018 100%) !important;
    }
    
    [data-testid="stSidebar"] .markdown-text-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #FFD700 !important;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #800020 0%, #A0002A 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    
    .section-header {
        color: #800020 !important;
        border-bottom: 2px solid #800020;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_trained_model():
    """Load the pre-trained Random Forest model"""
    try:
        # Try main file first
        with open('trained_model_data.pkl', 'rb') as f:
            data = pickle.load(f)
        source = "main"
    except FileNotFoundError:
        try:
            # Try compressed file
            with gzip.open('trained_model_data.pkl.gz', 'rb') as f:
                data = joblib.load(f)
            source = "compressed"
        except FileNotFoundError:
            st.error("❌ Model files not found!")
            return None, [], pd.DataFrame(), None, None
    
    rf_model = data['rf_model']
    training_features = data['training_features']
    X = data['X']
    y = data['y']
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Calculate correlation for direction
    feature_correlations = []
    for feature in feature_importance['feature']:
        correlation = X[feature].corr(y)
        feature_correlations.append(correlation)
    
    feature_importance['correlation'] = feature_correlations
    feature_importance['effect'] = ['Positive' if corr > 0 else 'Negative' for corr in feature_correlations]
    
    st.success(f"✅ Model loaded from {source} file")
    
    return rf_model, training_features, feature_importance, X, y

# Header
st.markdown("""
<div class="main-header">
    <h1>🏫 Colorado Mesa University</h1>
    <p>Student Housing Needs Prediction System</p>
</div>
""", unsafe_allow_html=True)

# Load model and data
rf_model, training_features, feature_importance, X, y = load_trained_model()

# Sidebar
st.sidebar.markdown("### 🏫 Colorado Mesa University")
st.sidebar.markdown("**Housing Prediction System**")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a section", 
    ["📊 Model Information", "🔮 Make Predictions"]
)

# Page 1: Model Information
if page == "📊 Model Information":
    st.markdown('<h2 class="section-header">📊 Model Information</h2>', unsafe_allow_html=True)
    
    if rf_model is not None:
        # Model Details
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Model Type:** Random Forest")
            st.info(f"**Training Data:** 2018-2023 CMU Data")
            st.info(f"**Features:** {len(training_features)}")
        with col2:
            st.info("**Trees:** 100")
            st.info("**Max Depth:** 10")
            st.info("**Performance:** 90% Accuracy")
        
        # Feature Importance Chart
        st.markdown('<h3 class="section-header">📈 Top 15 Important Features</h3>', unsafe_allow_html=True)
        
        if not feature_importance.empty:
            top_15 = feature_importance.head(15)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(range(len(top_15)), top_15['importance'], 
                          color='#800020', alpha=0.8)
            
            ax.set_yticks(range(len(top_15)))
            ax.set_yticklabels(top_15['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 15 Most Important Features')
            ax.invert_yaxis()
            
            st.pyplot(fig)
            
            # Feature table
            st.dataframe(top_15[['feature', 'importance', 'effect']])

# Page 2: Make Predictions
elif page == "🔮 Make Predictions":
    st.markdown('<h2 class="section-header">🔮 Make Predictions</h2>', unsafe_allow_html=True)
    
    if rf_model is not None:
        st.success("✅ Model Ready for Predictions")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "📁 Upload Student Data (CSV)", 
            type=['csv']
        )
        
        if uploaded_file is not None:
            # Read file
            df_new = pd.read_csv(uploaded_file, dtype={'TermPIDMKey': str})
            st.success(f"✅ Uploaded: {df_new.shape[0]:,} students")
            
            with st.expander("👀 Preview Data"):
                st.dataframe(df_new.head())
            
            if 'TermPIDMKey' not in df_new.columns:
                st.error("❌ Missing TermPIDMKey column!")
            else:
                if st.button("🎯 Run Predictions", type="primary"):
                    with st.spinner("🔄 Making predictions..."):
                        
                        # Handle TermPIDMKey properly
                        def convert_scientific_to_full(value):
                            try:
                                if isinstance(value, str) and ('E' in value.upper() or 'e' in value):
                                    return f"{float(value):.0f}"
                                else:
                                    return f"{float(value):.0f}"
                            except:
                                return str(value)
                        
                        id_column = df_new['TermPIDMKey'].apply(convert_scientific_to_full)
                        X_new = df_new.drop('TermPIDMKey', axis=1)
                        
                        # Check features match
                        if set(X_new.columns) != set(training_features):
                            missing = set(training_features) - set(X_new.columns)
                            extra = set(X_new.columns) - set(training_features)
                            
                            if missing:
                                st.error(f"❌ Missing features: {list(missing)}")
                                st.stop()
                            if extra:
                                st.warning(f"⚠️ Extra features removed: {list(extra)}")
                                X_new = X_new[training_features]
                        
                        # Make predictions
                        predictions = rf_model.predict(X_new)
                        probabilities = rf_model.predict_proba(X_new)
                        
                        # Results
                        results_df = pd.DataFrame({
                            'TermPIDMKey': id_column,
                            'Predicted_Has_Future_Housing': predictions,
                            'Probability_Has_Housing': probabilities[:, 1].round(4)
                        })
                    
                    st.success("✅ Predictions Complete!")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    has_housing = int(predictions.sum())
                    with col1:
                        st.metric("Total Students", f"{len(results_df):,}")
                    with col2:
                        st.metric("Need Housing", f"{has_housing:,}")
                    with col3:
                        st.metric("No Housing", f"{len(predictions)-has_housing:,}")
                    with col4:
                        st.metric("Housing Rate", f"{predictions.mean():.1%}")
                    
                    # Sample results
                    st.subheader("📋 Sample Results")
                    sample = results_df.head(10).copy()
                    sample['Status'] = sample['Predicted_Has_Future_Housing'].map({1: '🏠 Yes', 0: '❌ No'})
                    display_sample = sample[['TermPIDMKey', 'Status', 'Probability_Has_Housing']]
                    display_sample.columns = ['Student ID', 'Needs Housing', 'Probability']
                    st.dataframe(display_sample, use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "💾 Download Results",
                        csv,
                        f"housing_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
        else:
            st.info("👆 Upload a CSV file to make predictions")
    else:
        st.error("❌ Model not available")

# Sidebar status
st.sidebar.markdown("---")
if rf_model is not None:
    st.sidebar.success("✅ Model Ready")
    st.sidebar.info(f"📊 Features: {len(training_features)}")
else:
    st.sidebar.error("❌ Model Not Available")
