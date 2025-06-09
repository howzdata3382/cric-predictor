
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from xgboost import XGBRegressor, XGBClassifier

# Page title
st.title("🏏 CricViz Ball-Tracking Runs Predictor")
st.markdown("""
This app uses ML models to predict runs from ball-tracking features.
Choose delivery features on the sidebar and view predictions & SHAP explanations.
""")

# Sidebar Inputs
st.sidebar.header("🔧 Delivery Inputs")
batting_hand = st.sidebar.selectbox("Batter Hand", ['right', 'left'])
bowling_hand = st.sidebar.selectbox("Bowler Hand", ['right', 'left'])
bowling_type = st.sidebar.selectbox("Bowling Type", ['pace', 'spin'])

release_speed = st.sidebar.slider("Release Speed (kph)", 80.0, 160.0, 130.0)
swing_angle = st.sidebar.slider("Swing Angle (°)", -20.0, 20.0, 0.0)
deviation = st.sidebar.slider("Deviation (°)", -10.0, 10.0, 0.0)
release_pos_y = st.sidebar.slider("Release Position Y (m)", -2.5, 2.5, 0.0)
release_pos_z = st.sidebar.slider("Release Position Z (m)", 0.5, 3.5, 2.0)
bounce_pos_x = st.sidebar.slider("Bounce Position X (m)", -2.0, 2.0, 0.0)
bounce_pos_y = st.sidebar.slider("Bounce Position Y (m)", -2.0, 2.0, 0.0)
crease_pos_y = st.sidebar.slider("Crease Y (m)", -2.0, 2.0, 0.0)
crease_pos_z = st.sidebar.slider("Crease Z (m)", 0.0, 3.0, 1.0)
stumps_pos_y = st.sidebar.slider("Stumps Y (m)", -2.0, 2.0, 0.0)
stumps_pos_z = st.sidebar.slider("Stumps Z (m)", 0.0, 3.0, 0.5)
bounce_vel_ratio_z = st.sidebar.slider("Bounce Velocity Ratio Z", -1.0, 1.0, 0.5)
release_angle = st.sidebar.slider("Release Angle (°)", -20.0, 20.0, 0.0)
drop_angle = st.sidebar.slider("Drop Angle (°)", -30.0, -1.0, -15.0)
bounce_angle = st.sidebar.slider("Bounce Angle (°)", 1.0, 30.0, 10.0)

# Model input
input_dict = {
    'release_speed_kph': release_speed,
    'swing_angle': swing_angle,
    'deviation': deviation,
    'release_position_y': release_pos_y,
    'release_position_z': release_pos_z,
    'bounce_position_x': bounce_pos_x,
    'bounce_position_y': bounce_pos_y,
    'crease_position_y': crease_pos_y,
    'crease_position_z': crease_pos_z,
    'stumps_position_y': stumps_pos_y,
    'stumps_position_z': stumps_pos_z,
    'bounce_velocity_ratio_z': bounce_vel_ratio_z,
    'release_angle': release_angle,
    'drop_angle': drop_angle,
    'bounce_angle': bounce_angle,
    'batting_hand_left': 1 if batting_hand == 'left' else 0,
    'bowling_hand_left': 1 if bowling_hand == 'left' else 0,
    'bowling_type_pace': 1 if bowling_type == 'pace' else 0,
}

input_df = pd.DataFrame([input_dict])

# Load models
xgb_reg = joblib.load("model_xgb_reg.pkl")
xgb_clf = joblib.load("model_xgb_clf.pkl")

# Predictions
reg_output = xgb_reg.predict(input_df)[0]
clf_output = int(xgb_clf.predict(input_df)[0])

st.subheader("📈 Predictions")
st.write(f"Predicted Runs (Regression): **{reg_output:.2f}**")
st.write(f"Predicted Run Category (Classification): **{clf_output}**")

# SHAP Explainability
st.subheader("🔍 SHAP Explanation")
explainer = shap.TreeExplainer(xgb_reg)
shap_vals = explainer.shap_values(input_df)
shap_df = pd.DataFrame({
    'feature': input_df.columns,
    'shap_value': shap_vals[0],
    'value': input_df.iloc[0].values
}).sort_values(by='shap_value', key=abs, ascending=False)

st.dataframe(shap_df.head(8))

fig, ax = plt.subplots()
shap_df_top = shap_df.head(8)
ax.barh(shap_df_top['feature'], shap_df_top['shap_value'], color='skyblue')
ax.set_xlabel("SHAP Value")
ax.set_title("Top Feature Impacts")
st.pyplot(fig)

st.caption("Model trained using XGBoost with CricViz ball-tracking data.")
