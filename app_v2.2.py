import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# Data loading and preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df.drop(columns=['wicket'])
    df = df[df['runs'] != 5]

    numerical_features = ['release_speed_kph', 'swing_angle', 'deviation',
                        'release_position_y', 'release_position_z',
                        'bounce_position_y', 'bounce_position_x',
                        'crease_position_y', 'crease_position_z',
                        'stumps_position_y', 'stumps_position_z',
                        'bounce_velocity_ratio_z', 'release_angle',
                        'drop_angle', 'bounce_angle']

    categorical_features = ['batting_hand', 'bowling_hand', 'bowling_type']

    X = df[numerical_features + categorical_features]
    y_reg = df['runs']
    y_clf = df['runs'].astype('category')

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42)

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, numerical_features, categorical_features

# Model training
def train_models(X_train, y_reg_train, y_clf_train, preprocessor):
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])
    reg_pipeline.fit(X_train, y_reg_train)

    label_encoder = LabelEncoder()
    y_clf_train_encoded = label_encoder.fit_transform(y_clf_train)
    num_classes = len(label_encoder.classes_)

    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, objective='multi:softprob', num_class=num_classes))
    ])
    clf_pipeline.fit(X_train, y_clf_train_encoded)

    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(reg_pipeline, 'model_xgb_reg.pkl')
    joblib.dump(clf_pipeline, 'model_xgb_clf.pkl')

    return reg_pipeline, clf_pipeline

# SHAP analysis
def generate_shap_plot(reg_pipeline, X_train, X_test):
    preprocessor = reg_pipeline.named_steps['preprocessor']
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    num_features = X_train.select_dtypes(include=['float64', 'int64']).columns
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        input_features=X_train.select_dtypes(include=['object']).columns)
    all_features = np.concatenate([num_features, cat_features])

    explainer = shap.Explainer(reg_pipeline.named_steps['regressor'])
    shap_values = explainer(X_test_processed)

    plt.figure()
    shap.summary_plot(shap_values, X_test_processed, feature_names=all_features, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

    return explainer, all_features

# Streamlit app
def main():
    st.markdown("""
    <h1 style='text-align: center;'>🏏 Cricket Runs Predictor</h1>
""", unsafe_allow_html=True)
    st.write("Predict runs from ball-tracking data using simplified cricket language")

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, numerical_features, categorical_features = load_data()

    reg_pipeline, clf_pipeline = train_models(X_train, y_reg_train, y_clf_train, preprocessor)
    label_encoder = joblib.load('label_encoder.pkl')
    st.success("Models trained successfully!")

    # User Inputs
    bowling_type = st.radio("Bowler Type", ["Pace (Fast Bowler)", "Spin (Spinner)"])
    bowling_type_raw = "pace" if "Pace" in bowling_type else "spin"

    speed_options = {
        "70-90 KPH (Slow)": 80,
        "90-110 KPH (Medium-Slow)": 100,
        "110-120 KPH (Medium)": 115,
        "120-125 KPH (Medium-Fast)": 122.5,
        "125-130 KPH (Fast)": 127.5,
        "130-135 KPH": 132.5,
        "135-140 KPH": 137.5,
        "140-145 KPH": 142.5,
        "145+ KPH (Very Fast)": 147.5
    }
    release_speed_kph = speed_options[st.selectbox("Ball Speed", list(speed_options.keys()))]

    batting_hand = st.radio("Batsman Handedness", ["Right-hand", "Left-hand"])
    batting_hand_raw = "right" if batting_hand == "Right-hand" else "left"

    line_options = {
        "↖ Wide Outside Off Stump": -0.5,
        "← Outside Off Stump": -0.35,
        "◀ On Off Stump": -0.22,
        "⬅ Middle & Off": -0.11,
        "⬇ On Middle Stump": 0.0,
        "➡ Middle & Leg": 0.11,
        "▶ On Leg Stump": 0.22,
        "↘ Down Leg Side": 0.35
    }
    with st.sidebar:
        st.markdown("### ℹ️ What does 'Line of the Ball' mean?")
        st.markdown("""
        - **↖ Wide Outside Off**: Ball pitching well outside off stump
        - **⬇ On Middle Stump**: Ball in line with the middle stump
        - **↘ Down Leg Side**: Ball drifting far down leg
        """)
    bounce_position_y = line_options[st.selectbox("Line of the Ball", list(line_options.keys()))]

    length_options = {
        "🎯 Full / Yorker": 1.0,
        "✅ Good Length": 4.0,
        "⬆ Short / Back of a Length": 6.5,
        "⚠️ Bouncer": 8.5,
        "🚫 Full Toss": -1.0
    }
    with st.sidebar:
        st.markdown("### ℹ️ What does 'Length of the Ball' mean?")
        st.markdown("""
        - **🎯 Full / Yorker**: Ball landing near the batsman's crease
        - **✅ Good Length**: Ideal area for bowling, hard to judge
        - **⚠️ Bouncer**: Short-pitched ball rising sharply
        - **🚫 Full Toss**: No bounce before reaching batsman
        """)
    bounce_position_x = length_options[st.selectbox("Length of the Ball", list(length_options.keys()))]

    if batting_hand == "Right-hand":
        movement_options = {
            "No Movement / Straight": 0.0,
            "In-swing / Off-break": 1.0,
            "Out-swing / Leg-break": -1.0
        }
    else:
        movement_options = {
            "No Movement / Straight": 0.0,
            "In-swing / Off-break": -1.0,
            "Out-swing / Leg-break": 1.0
        }

    movement_label = st.selectbox("Movement of the Ball", list(movement_options.keys()))
    swing_angle = movement_options[movement_label]
    deviation = swing_angle

    input_data = pd.DataFrame({
        'release_speed_kph': [release_speed_kph],
        'swing_angle': [swing_angle],
        'deviation': [deviation],
        'release_position_y': [0.0],
        'release_position_z': [2.0],
        'bounce_position_y': [bounce_position_y],
        'bounce_position_x': [bounce_position_x],
        'crease_position_y': [0.0],
        'crease_position_z': [0.5],
        'stumps_position_y': [0.0],
        'stumps_position_z': [0.5],
        'bounce_velocity_ratio_z': [-0.5],
        'release_angle': [0.0],
        'drop_angle': [-15.0],
        'bounce_angle': [10.0],
        'batting_hand': [batting_hand_raw],
        'bowling_hand': ["right"],
        'bowling_type': [bowling_type_raw]
    })

    if st.button("Predict Runs"):
        reg_pred = reg_pipeline.predict(input_data)[0]
        clf_pred_encoded = clf_pipeline.predict(input_data)[0]
        clf_pred = label_encoder.inverse_transform([clf_pred_encoded])[0]

        # Round regression prediction
        rounded_runs = round(reg_pred, 1)

        # Classify regression output into interpretation buckets
        if rounded_runs < 1:
            run_type = "Dot Ball"
        elif 1 <= rounded_runs < 2:
            run_type = "Single"
        elif 2 <= rounded_runs < 4:
            run_type = "2 or 3 runs"
        elif 4 <= rounded_runs <= 6:
            run_type = "Boundary"
        else:
            run_type = "Unusual High Runs"

        # Display both
        st.success(f"Predicted runs (regression): {rounded_runs} → *{run_type}*")
        st.success(f"Predicted run category (classification): {clf_pred} run(s)")

        # Combined summary
        st.info(f"The regression model estimates {rounded_runs} runs likely ({run_type}), while the classifier predicts an outcome of {clf_pred} run(s). Use both for contextual insights.")

        explainer, feature_names = generate_shap_plot(reg_pipeline, X_train, X_test)
        input_processed = reg_pipeline.named_steps['preprocessor'].transform(input_data)
        shap_values = explainer(input_processed)

        st.markdown("""
    <h2 style='text-align: left; color: #1f77b4;'>📊 Feature Impact</h2>
""", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values.values, features=input_processed, feature_names=feature_names, show=False)
        st.pyplot(fig)

    if st.checkbox("Show Model Performance"):
        y_reg_pred = reg_pipeline.predict(X_test)
        y_clf_pred_encoded = clf_pipeline.predict(X_test)
        y_clf_pred = label_encoder.inverse_transform(y_clf_pred_encoded)
        y_clf_test_decoded = label_encoder.transform(y_clf_test)

        st.markdown("""
    <h2 style='text-align: left; color: #2ca02c;'>📈 Regression Metrics</h2>
""", unsafe_allow_html=True)
        st.metric("MAE", f"{mean_absolute_error(y_reg_test, y_reg_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_reg_test, y_reg_pred):.2f}")

        st.markdown("""
    <h2 style='text-align: left; color: #d62728;'>🎯 Classification Metrics</h2>
""", unsafe_allow_html=True)
        st.metric("Accuracy", f"{accuracy_score(y_clf_test_decoded, y_clf_pred_encoded):.2f}")
        st.metric("Precision", f"{precision_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted'):.2f}")
        st.metric("Recall", f"{recall_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted'):.2f}")
        st.metric("F1 Score", f"{f1_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted'):.2f}")

        if os.path.exists('shap_summary.png'):
            st.image('shap_summary.png', caption='Overall Feature Importance')
        else:
            st.warning("SHAP summary plot is not available yet. Please run a prediction first.")

if __name__ == "__main__":
    main()
