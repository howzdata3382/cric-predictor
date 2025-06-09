
# 🏏 CricViz Ball-Tracking Runs Predictor

This is an end-to-end machine learning project built to predict runs scored in cricket based on ball-tracking data.

## 🔍 Features
- Supports both Regression (predict exact runs) and Classification (categorical prediction: 0,1,2,3,4,6).
- SHAP-based model interpretation for coaching insights.
- Streamlit app for interactive use.
- Powered by XGBoost, scikit-learn, pandas, and SHAP.

## 🚀 How to Use

### Option 1: Streamlit Cloud (Recommended)
1. Fork or clone this repository.
2. Push to your GitHub account.
3. Go to [Streamlit Cloud](https://streamlit.io/cloud) and click "New App".
4. Point to `app.py` and click Deploy.

### Option 2: Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
.
├── app.py                  # Streamlit app
├── model_xgb_reg.pkl       # Trained regression model
├── model_xgb_clf.pkl       # Trained classifier
├── requirements.txt        # Python dependencies
└── README.md               # Project instructions
```

## 📊 Model Insights
- Trained on 200k+ CricViz deliveries.
- Regression R² ~0.02, Classification accuracy ~46%.
- SHAP used for feature importance and interpretability.

---
