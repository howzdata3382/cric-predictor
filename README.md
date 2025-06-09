
# 🏏 CricViz Ball-Tracking Runs Predictor

This is a complete machine learning project that predicts cricket runs based on ball-tracking features using regression and classification models. Built for the CricViz Data Science Task (Jan 2025), it includes model explainability and an interactive web interface.

## 🚀 Live Streamlit App

You can deploy this app interactively using [Streamlit Cloud](https://streamlit.io/cloud) by connecting this GitHub repo.

## 📊 App Features

- Regression: Predicts **exact runs** (e.g. 1.34) per ball.
- Classification: Predicts **run category** (0,1,2,3,4,6).
- Interactive Sidebar: Choose ball attributes like bowling type, speed, angles.
- SHAP-based Explainability: See **which features influenced** predictions.
- Visuals: Bar plot of top SHAP values and feature impact.

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── model_xgb_reg.pkl       # Trained regression model
├── model_xgb_clf.pkl       # Trained classification model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🧪 Try It Locally (Optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy to Streamlit Cloud

1. Fork or clone this repository
2. Sign in at [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New App”**
4. Point to `app.py` in this repo and click **Deploy**

You'll get a public URL like:
```
https://yourusername-cricviz-predictor.streamlit.app
```

---

## 📚 Data & Model

- Trained on 200k+ CricViz ball-tracking rows
- Models used: XGBoost Regressor and Classifier
- Evaluation:
  - Regression R² ~ 0.02, MAE ~ 1.17
  - Classification accuracy ~ 46%

## 👨‍🏫 For Coaches

SHAP explanations show how bowling style, hand, speed, and bounce position affect run scoring. Use the app to simulate bowling strategies and learn trends.

---

## 🔒 Confidentiality

This project uses proprietary CricViz data and is intended only for assessment purposes. Do not reuse the dataset beyond evaluation.
