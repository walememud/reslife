# 🏫 CMU Housing Prediction App

Colorado Mesa University Student Housing Needs Prediction System

## 📊 About
This Streamlit application predicts whether continuing students will need university housing based on historical data from 2018-2023.

## 🚀 Features
- **Model Information**: View Random Forest model details and feature importance
- **Predictions**: Upload student data and get housing predictions
- **Interactive Dashboard**: Clean, university-branded interface
- **Export Results**: Download predictions as CSV

## 🔧 Model Details
- **Algorithm**: Random Forest Classifier
- **Training Period**: 2018-2023 CMU student data
- **Accuracy**: ~90%
- **Features**: Academic, demographic, and enrollment data

## 📱 How to Use
1. Navigate to the "Model Information" section to understand the model
2. Go to "Make Predictions" to upload student data
3. Upload a CSV file with student data (must include TermPIDMKey)
4. Click "Run Predictions" to get results
5. Download the predictions as CSV

## 📋 Required Data Format
Your CSV file must contain:
- `TermPIDMKey`: Student identifier
- All training features used in the model
- No missing required columns

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 📈 Model Performance
- Overall Accuracy: 90%
- Housing Class Precision: 66%
- Housing Class Recall: 60%
- F1-Score: 78%

## 🎨 Design
Custom Colorado Mesa University maroon theme with gold accents.

---
*Created: 2025-08-26*
*Colorado Mesa University - Institutional Research*
