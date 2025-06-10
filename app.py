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

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor

# Model training
def train_models(X_train, y_reg_train, y_clf_train, preprocessor):
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])
    reg_pipeline.fit(X_train, y_reg_train)

    # Encode classification labels to 0...N-1
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
    st.title("🏏 Cricket Runs Prediction")
    st.write("Predict runs from ball-tracking data using XGBoost")

    # Load data
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor = load_data()

    # Load or train models
    try:
        reg_pipeline = joblib.load('model_xgb_reg.pkl')
        clf_pipeline = joblib.load('model_xgb_clf.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        st.success("Models loaded successfully!")
    except:
        st.info("Training models...")
        reg_pipeline, clf_pipeline = train_models(X_train, y_reg_train, y_clf_train, preprocessor)
        label_encoder = joblib.load('label_encoder.pkl')
        st.success("Models trained and saved!")

    # Prediction interface
    st.sidebar.header("Input Parameters")

    col1, col2 = st.columns(2)
    with col1:
        release_speed_kph = st.slider("Release Speed (kph)", 70.0, 150.0, 120.0)
        swing_angle = st.slider("Swing Angle", -5.0, 5.0, 0.0)
        release_position_y = st.slider("Release Position Y", -2.0, 2.0, 0.0)
        release_position_z = st.slider("Release Position Z", 1.0, 3.0, 2.0)
        bounce_position_y = st.slider("Bounce Position Y", -2.0, 2.0, 0.0)
        bounce_position_x = st.slider("Bounce Position X", -2.0, 15.0, 5.0)

    with col2:
        crease_position_y = st.slider("Crease Position Y", -2.0, 2.0, 0.0)
        crease_position_z = st.slider("Crease Position Z", 0.0, 2.0, 0.5)
        bounce_velocity_ratio_z = st.slider("Bounce Velocity Ratio Z", -1.0, 0.0, -0.5)
        release_angle = st.slider("Release Angle", -20.0, 20.0, 0.0)
        drop_angle = st.slider("Drop Angle", -25.0, 0.0, -15.0)
        bounce_angle = st.slider("Bounce Angle", 0.0, 25.0, 10.0)

    batting_hand = st.sidebar.selectbox("Batting Hand", ["left", "right"])
    bowling_hand = st.sidebar.selectbox("Bowling Hand", ["left", "right"])
    bowling_type = st.sidebar.selectbox("Bowling Type", ["pace", "spin"])
    deviation = st.sidebar.slider("Deviation", -10.0, 10.0, 0.0)

    input_data = pd.DataFrame({
        'release_speed_kph': [release_speed_kph],
        'swing_angle': [swing_angle],
        'deviation': [deviation],
        'release_position_y': [release_position_y],
        'release_position_z': [release_position_z],
        'bounce_position_y': [bounce_position_y],
        'bounce_position_x': [bounce_position_x],
        'crease_position_y': [crease_position_y],
        'crease_position_z': [crease_position_z],
        'stumps_position_y': [0.0],  # Default value
        'stumps_position_z': [0.5],   # Default value
        'bounce_velocity_ratio_z': [bounce_velocity_ratio_z],
        'release_angle': [release_angle],
        'drop_angle': [drop_angle],
        'bounce_angle': [bounce_angle],
        'batting_hand': [batting_hand],
        'bowling_hand': [bowling_hand],
        'bowling_type': [bowling_type]
    })

    if st.button("Predict Runs"):
        reg_pred = reg_pipeline.predict(input_data)[0]
        clf_pred_encoded = clf_pipeline.predict(input_data)[0]
        clf_pred = label_encoder.inverse_transform([clf_pred_encoded])[0]

        st.success(f"Predicted runs: {reg_pred:.1f}")
        st.success(f"Predicted run category: {clf_pred}")

        # SHAP explanation
        explainer, feature_names = generate_shap_plot(reg_pipeline, X_train, X_test)
        input_processed = reg_pipeline.named_steps['preprocessor'].transform(input_data)
        shap_values = explainer(input_processed)

        st.subheader("Feature Impact")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    if st.checkbox("Show Model Performance"):
        y_reg_pred = reg_pipeline.predict(X_test)
        y_clf_pred_encoded = clf_pipeline.predict(X_test)
        y_clf_pred = label_encoder.inverse_transform(y_clf_pred_encoded)
        y_clf_test_decoded = label_encoder.transform(y_clf_test)

        st.subheader("Regression Metrics")
        st.metric("MAE", f"{mean_absolute_error(y_reg_test, y_reg_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_reg_test, y_reg_pred):.2f}")

        st.subheader("Classification Metrics")
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
