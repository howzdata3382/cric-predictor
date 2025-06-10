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

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor = load_data()

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

    st.sidebar.header("Input Parameters")

    def map_range(option, mapping):
        return mapping[option]

    # Bucket definitions
    speed_options = {"70–90": 80, "90–110": 100, "110–130": 120, "130–150": 140}
    swing_options = {"< -2": -3, "-2 to 0": -1, "0 to 2": 1, "> 2": 3}
    deviation_options = swing_options
    release_y_options = {"< -1": -1.5, "-1 to 0": -0.5, "0 to 1": 0.5, "> 1": 1.5}
    release_z_options = {"1–1.5": 1.25, "1.5–2": 1.75, "2–2.5": 2.25, "2.5–3": 2.75}
    bounce_y_options = release_y_options
    bounce_x_options = {"0–4": 2, "4–8": 6, "8–12": 10, "12–15": 13.5}
    crease_y_options = release_y_options
    crease_z_options = {"0–0.5": 0.25, "0.5–1": 0.75, "1–1.5": 1.25, "1.5–2": 1.75}
    velocity_ratio_options = {"< -0.75": -0.85, "-0.75 to -0.5": -0.625, "-0.5 to -0.25": -0.375, "> -0.25": -0.1}
    angle_options = {"< -10": -15, "-10 to 0": -5, "0 to 10": 5, "> 10": 15}
    drop_options = {"< -20": -22.5, "-20 to -10": -15, "-10 to 0": -5}
    bounce_angle_options = {"0–8": 4, "8–16": 12, "16–25": 20}

    col1, col2 = st.columns(2)
    with col1:
        release_speed_kph = map_range(st.selectbox("Release Speed (kph)", list(speed_options.keys())), speed_options)
        swing_angle = map_range(st.selectbox("Swing Angle", list(swing_options.keys())), swing_options)
        release_position_y = map_range(st.selectbox("Release Position Y", list(release_y_options.keys())), release_y_options)
        release_position_z = map_range(st.selectbox("Release Position Z", list(release_z_options.keys())), release_z_options)
        bounce_position_y = map_range(st.selectbox("Bounce Position Y", list(bounce_y_options.keys())), bounce_y_options)
        bounce_position_x = map_range(st.selectbox("Bounce Position X", list(bounce_x_options.keys())), bounce_x_options)

    with col2:
        crease_position_y = map_range(st.selectbox("Crease Position Y", list(crease_y_options.keys())), crease_y_options)
        crease_position_z = map_range(st.selectbox("Crease Position Z", list(crease_z_options.keys())), crease_z_options)
        bounce_velocity_ratio_z = map_range(st.selectbox("Bounce Velocity Ratio Z", list(velocity_ratio_options.keys())), velocity_ratio_options)
        release_angle = map_range(st.selectbox("Release Angle", list(angle_options.keys())), angle_options)
        drop_angle = map_range(st.selectbox("Drop Angle", list(drop_options.keys())), drop_options)
        bounce_angle = map_range(st.selectbox("Bounce Angle", list(bounce_angle_options.keys())), bounce_angle_options)

    batting_hand = st.sidebar.selectbox("Batting Hand", ["left", "right"])
    bowling_hand = st.sidebar.selectbox("Bowling Hand", ["left", "right"])
    bowling_type = st.sidebar.selectbox("Bowling Type", ["pace", "spin"])
    deviation = map_range(st.selectbox("Deviation", list(deviation_options.keys())), deviation_options)

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
        'stumps_position_y': [0.0],
        'stumps_position_z': [0.5],
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
