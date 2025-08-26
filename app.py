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
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate correlation for direction (with NaN handling)
        feature_correlations = []
        for feature in feature_importance['feature']:
            try:
                correlation = X[feature].corr(y)
                # Handle NaN correlations (constant features)
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
<div class="main-header">
    <h1>Student Housing Needs Prediction System</h1>
    <p>Continuing Students</p>
</div>
""", unsafe_allow_html=True)

# Load model and data
rf_model, training_features, feature_importance, X, y = load_trained_model()

# Sidebar Navigation - centered and bigger logo
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image("Picture1.png", use_column_width=True)

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Choose a section", 
    ["Model Information", "Make Predictions"]
)

# Page 1: Model Information
if page == "Model Information":
    st.markdown('<h2 class="section-header">Trained Model Information</h2>', unsafe_allow_html=True)
    
    if rf_model is not None:
        # Model Details
        st.markdown('<h3 class="section-header">Model Configuration</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Model Type:** Random Forest (No Class Balancing)")
            st.info(f"**Training Data:** 2018-2023 CMU Student Data")
            st.info(f"**Number of Features:** {len(training_features)}")
        with col2:
            st.info("**Number of Trees:** 100")
            st.info("**Max Depth:** 10")
            st.info("**Training Approach:** Unbalanced (Natural Learning)")
        
        # Detailed Classification Report
        st.markdown('<h3 class="section-header">Detailed Classification Report</h3>', unsafe_allow_html=True)
        
        # Create classification report data
        classification_data = {
            'Class': ['No Future Housing', 'Has Future Housing', '', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
            'Precision': ['0.93', '0.66', '', '', '0.80', '0.90'],
            'Recall': ['0.95', '0.60', '', '', '0.77', '0.90'],
            'F1-Score': ['0.94', '0.63', '', '0.90', '0.78', '0.90'],
            'Support': ['9,401', '1,556', '', '10,957', '10,957', '10,957']
        }
        
        classification_df = pd.DataFrame(classification_data)
        
        # Style the dataframe
        def style_classification_report(row):
            if row.name == 2:  # Empty row
                return ['background-color: transparent'] * len(row)
            elif row.name == 3:  # Accuracy row
                return ['background-color: #f0f2f6; font-weight: bold'] * len(row)
            elif row.name in [4, 5]:  # Average rows
                return ['background-color: #e8f4f8'] * len(row)
            elif row['Class'] == 'Has Future Housing':
                return ['background-color: #fff2e6'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = classification_df.style.apply(style_classification_report, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Key insights from classification report
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Accuracy", "90.0%", help="Correct predictions out of all predictions")
        with col2:
            st.metric("Housing Class Precision", "66%", help="Accuracy when predicting 'Has Future Housing'")
        with col3:
            st.metric("Housing Class Recall", "60%", help="Percentage of actual housing cases correctly identified")
        
        # Top 15 Feature Importance Chart
        st.markdown('<h3 class="section-header">Top 15 Most Important Features</h3>', unsafe_allow_html=True)
        
        if not feature_importance.empty:
            top_15_features = feature_importance.head(15)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            bars = ax.barh(range(len(top_15_features)), top_15_features['importance'], 
                          color='#800020', alpha=0.8, edgecolor='#600018', linewidth=1)
            
            ax.set_yticks(range(len(top_15_features)))
            ax.set_yticklabels(top_15_features['feature'], fontsize=10)
            ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold', color='#800020')
            ax.set_title('Top 15 Most Important Features for Housing Prediction', 
                        fontsize=14, fontweight='bold', color='#800020', pad=20)
            ax.invert_yaxis()
            
            ax.grid(True, alpha=0.3, color='#800020', linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#800020')
            ax.spines['bottom'].set_color('#800020')
            ax.tick_params(colors='#800020')
            
            st.pyplot(fig)
            
            # Feature importance table
            st.markdown('<h3 class="section-header">Feature Importance Details</h3>', unsafe_allow_html=True)
            display_features = top_15_features[['feature', 'importance']].copy()
            display_features['importance'] = display_features['importance'].round(4)
            display_features.index = range(1, len(display_features) + 1)
            display_features.columns = ['Feature Name', 'Importance Score']

            # Style the dataframe to left-align the importance score column
            styled_features = display_features.style.set_properties(
                subset=['Importance Score'], 
                **{'text-align': 'left'}
            )
            st.dataframe(display_features, use_container_width=True)
    else:
        st.error("Model not available. Please check if trained_model_data.pkl exists.")

# Page 2: Make Predictions
elif page == "Make Predictions":
    
    if rf_model is not None:
        st.success("CMU Housing Prediction Model is ready")
        
        # Upload prediction data
        prediction_file = st.file_uploader(
            "Upload Student Data (CSV Format)", 
            type=['csv']
        )
        
        if prediction_file is not None:
            # Read CSV and force TermPIDMKey to be string
            df_new = pd.read_csv(prediction_file, dtype={'TermPIDMKey': str})
            
            st.success(f"File uploaded: **{df_new.shape[0]:,}** students, **{df_new.shape[1]}** features")
            
            with st.expander("Preview Data"):
                st.dataframe(df_new.head())
            
            if 'TermPIDMKey' not in df_new.columns:
                st.error("Missing required column: **TermPIDMKey**")
            else:
                if st.button("Run Predictions", type="primary", use_container_width=True):
                    
                    with st.spinner("Making predictions..."):
                        
                        # Prepare features with proper TermPIDMKey handling
                        def convert_scientific_to_full(value):
                            try:
                                if isinstance(value, str):
                                    if 'E' in value.upper() or 'e' in value:
                                        float_val = float(value)
                                        return f"{float_val:.0f}"
                                    else:
                                        return str(value)
                                else:
                                    return f"{float(value):.0f}"
                            except:
                                return str(value)
                        
                        id_column = df_new['TermPIDMKey'].apply(convert_scientific_to_full)
                        X_new = df_new.drop('TermPIDMKey', axis=1)
                        
                        # Check features
                        training_features_set = set(training_features)
                        prediction_features_set = set(X_new.columns)
                        
                        if training_features_set != prediction_features_set:
                            missing = training_features_set - prediction_features_set
                            extra = prediction_features_set - training_features_set
                            
                            if missing:
                                st.error(f"Missing features: {list(missing)}")
                                st.stop()
                            if extra:
                                st.warning(f"Removing extra features: {list(extra)}")
                                X_new = X_new[training_features]
                        
                        # Make predictions
                        predictions = rf_model.predict(X_new)
                        probabilities = rf_model.predict_proba(X_new)
                        
                        # Create results dataframe with properly formatted TermPIDMKey
                        results_df = pd.DataFrame({
                            'TermPIDMKey': id_column,
                            'Predicted_Has_Future_Housing': predictions,
                            'Probability_Has_Housing': probabilities[:, 1].round(4)
                        })
                        
                        # Ensure TermPIDMKey stays as string in results
                        results_df['TermPIDMKey'] = results_df['TermPIDMKey'].astype(str)
                    
                    st.success("Predictions completed!")
                    
                    # Summary
                    col1, col2, col3, col4 = st.columns(4)
                    has_housing = int(predictions.sum())
                    with col1:
                        st.metric("Total Students", f"{len(results_df):,}")
                    with col2:
                        st.metric("Need Housing", f"{has_housing:,}")
                    with col3:
                        st.metric("No Housing Needed", f"{len(predictions)-has_housing:,}")
                    with col4:
                        st.metric("Housing Rate", f"{predictions.mean():.1%}")
                    
                    # Sample results
                    st.subheader("Sample Results")
                    sample = results_df.head(10).copy()
                    sample['TermPIDMKey'] = sample['TermPIDMKey'].astype(str)
                    sample['Status'] = sample['Predicted_Has_Future_Housing'].map({1: 'Yes', 0: 'No'})
                    display_sample = sample[['TermPIDMKey', 'Status', 'Probability_Has_Housing']].copy()
                    display_sample.columns = ['Student ID', 'Needs Housing', 'Probability']
                    
                    # Format the display to prevent scientific notation
                    styled_sample = display_sample.style.format({'Student ID': lambda x: f"{x}"})
                    st.dataframe(styled_sample, use_container_width=True, hide_index=True)
                    
                    # Download - ensure proper string formatting in CSV
                    csv_results = results_df.copy()
                    csv_results['TermPIDMKey'] = csv_results['TermPIDMKey'].astype(str)
                    
                    # Create CSV without float formatting to preserve probability decimals
                    csv_buffer = io.StringIO()
                    csv_results.to_csv(csv_buffer, index=False)
                    csv = csv_buffer.getvalue()
                    
                    st.download_button(
                        "Download Full Results",
                        csv,
                        f"CMU_housing_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    
        else:
            st.info("Upload a CSV file to make predictions")
    else:
        st.error("Model not available")

# Sidebar
st.sidebar.markdown("---")
if rf_model is not None:
    st.sidebar.success("Model Ready")
    st.sidebar.info(f"Features: {len(training_features)}")
else:
    st.sidebar.error("Model Not Available")
