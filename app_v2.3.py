import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# Define features globally so they are accessible throughout the script.
numerical_features = [
    'release_speed_kph', 'swing_angle', 'deviation',
    'release_position_y', 'release_position_z',
    'bounce_position_y', 'bounce_position_x',
    'crease_position_y', 'crease_position_z',
    'stumps_position_y', 'stumps_position_z',
    'bounce_velocity_ratio_z', 'release_angle',
    'drop_angle', 'bounce_angle'
]
categorical_features = [
    'batting_hand', 'bowling_hand', 'bowling_type',
    'line_category', 'length_category', 'movement_category', 'speed_category'
]


# --- Helper Functions for Feature Engineering ---
def categorize_line(bounce_y, batting_hand):
    """Categorizes the line of the ball into cricketing terms."""
    if batting_hand == 'right':
        if bounce_y <= -0.4: return 'Wide Outside Off Stump'
        elif -0.4 < bounce_y <= -0.25: return 'Outside Off Stump'
        elif -0.25 < bounce_y <= -0.1: return 'On Off Stump'
        elif -0.1 < bounce_y <= -0.05: return 'Middle & Off'
        elif -0.05 < bounce_y <= 0.05: return 'On Middle Stump'
        elif 0.05 < bounce_y <= 0.15: return 'Middle & Leg'
        elif 0.15 < bounce_y <= 0.3: return 'On Leg Stump'
        else: return 'Down Leg Side'
    else: # Left-hand batsman - 'y' coordinates are mirrored for off/leg
        if bounce_y >= 0.4: return 'Wide Outside Off Stump'
        elif 0.25 <= bounce_y < 0.4: return 'Outside Off Stump'
        elif 0.1 <= bounce_y < 0.25: return 'On Off Stump'
        elif 0.05 <= bounce_y < 0.1: return 'Middle & Off'
        elif -0.05 <= bounce_y < 0.05: return 'On Middle Stump'
        elif -0.15 <= bounce_y < -0.05: return 'Middle & Leg'
        elif -0.3 <= bounce_y < -0.15: return 'On Leg Stump'
        else: return 'Down Leg Side'

def categorize_length(bounce_x):
    """Categorizes the length of the ball into cricketing terms."""
    if bounce_x <= 1.5: return 'Full/Yorker'
    elif 1.5 < bounce_x <= 5.0: return 'Good Length'
    elif 5.0 < bounce_x <= 8.0: return 'Short/Back of a Length'
    else: return 'Bouncer'

def categorize_movement(swing_angle, deviation, bowling_type, batting_hand):
    """Categorizes swing/seam/spin movement based on handedness and bowling type."""
    movement_threshold_swing = 0.5
    movement_threshold_dev = 0.5

    is_swinging = abs(swing_angle) > movement_threshold_swing
    is_deviating = abs(deviation) > movement_threshold_dev

    if not is_swinging and not is_deviating:
        return 'No Movement / Straight'

    if bowling_type == 'pace':
        if batting_hand == 'right':
            if swing_angle > movement_threshold_swing or deviation > movement_threshold_dev:
                return 'In-swing / Leg-cutter'
            elif swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'Out-swing / Off-cutter'
        else: # Left-hand batsman
            if swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'In-swing / Leg-cutter'
            elif swing_angle > movement_threshold_swing or deviation > movement_threshold_dev:
                return 'Out-swing / Off-cutter'
    elif bowling_type == 'spin':
        if batting_hand == 'right':
            if swing_angle > movement_threshold_swing or deviation > movement_threshold_dev:
                return 'Off-break / Googly'
            elif swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'Leg-break / Flipper'
        else: # Left-hand batsman
            if swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'Off-break / Googly'
            elif swing_angle > movement_threshold_swing or deviation > -movement_threshold_dev:
                return 'Leg-break / Flipper'
    return 'Other Movement'

def categorize_runs_for_classification(runs):
    """Categorizes runs into broader, more balanced classes for classification."""
    if runs == 0:
        return 'Dot Ball'
    elif runs == 1:
        return 'Single'
    elif runs in [2, 3]:
        return 'Two or Three Runs'
    elif runs in [4, 6]:
        return 'Boundary'
    else:
        return 'Other Runs'

# Data loading and preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df.drop(columns=['wicket'])
    df = df[df['runs'] != 5]

    # --- New Feature Engineering ---
    df['line_category'] = df.apply(lambda row: categorize_line(row['bounce_position_y'], row['batting_hand']), axis=1)
    df['length_category'] = df['bounce_position_x'].apply(categorize_length)
    df['movement_category'] = df.apply(lambda row: categorize_movement(row['swing_angle'], row['deviation'], row['bowling_type'], row['batting_hand']), axis=1)
    df['speed_category'] = pd.cut(df['release_speed_kph'],
                                   bins=[0, 90, 110, 120, 125, 130, 135, 140, 145, 200],
                                   labels=['Very Slow', 'Slow', 'Medium', 'Medium-Fast', 'Fast', 'Very Fast', 'Express', 'Extreme', 'Blazing'],
                                   right=False)

    y_reg = df['runs']
    y_clf = df['runs'].apply(categorize_runs_for_classification) # Apply new categorization

    # Combine X and y_clf for inverse recommendations
    df_for_inverse = df.copy()
    df_for_inverse['run_category'] = y_clf

    # Compute inverse recommendations
    inverse_recommendations = {}
    for run_cat in df_for_inverse['run_category'].unique():
        subset = df_for_inverse[df_for_inverse['run_category'] == run_cat]
        if not subset.empty:
            inverse_recommendations[run_cat] = {
                'line_category': subset['line_category'].mode()[0] if not subset['line_category'].empty else 'N/A',
                'length_category': subset['length_category'].mode()[0] if not subset['length_category'].empty else 'N/A',
                'movement_category': subset['movement_category'].mode()[0] if not subset['movement_category'].empty else 'N/A',
                'batting_hand': subset['batting_hand'].mode()[0] if not subset['batting_hand'].empty else 'N/A',
                'bowling_hand': subset['bowling_hand'].mode()[0] if not subset['bowling_hand'].empty else 'N/A',
                'bowling_type': subset['bowling_type'].mode()[0] if not subset['bowling_type'].empty else 'N/A',
                'speed_category': subset['speed_category'].mode()[0] if not subset['speed_category'].empty else 'N/A',
                'release_speed_kph_median': subset['release_speed_kph'].median() if not subset['release_speed_kph'].empty else 'N/A'
            }
        else:
            inverse_recommendations[run_cat] = {
                'line_category': 'No data', 'length_category': 'No data',
                'movement_category': 'No data', 'batting_hand': 'No data',
                'bowling_hand': 'No data', 'bowling_type': 'No data',
                'speed_category': 'No data', 'release_speed_kph_median': 'No data'
            }

    # Define X using global feature lists after feature engineering
    X = df[numerical_features + categorical_features]

    # --- Preprocessing Pipeline ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
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

    # Split data AFTER feature engineering but BEFORE fitting preprocessor
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42)

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, inverse_recommendations

# Model training
def train_models(X_train, y_reg_train, y_clf_train, preprocessor):
    # --- Regression Model Training ---
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=5))
    ])
    reg_pipeline.fit(X_train, y_reg_train)

    # --- Classification Model Training ---
    label_encoder = LabelEncoder()
    y_clf_train_encoded = label_encoder.fit_transform(y_clf_train)
    num_classes = len(label_encoder.classes_)

    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, objective='multi:softprob', num_class=num_classes,
                                     n_estimators=500, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='mlogloss'))
    ])
    clf_pipeline.fit(X_train, y_clf_train_encoded)

    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(reg_pipeline, 'model_xgb_reg.pkl')
    joblib.dump(clf_pipeline, 'model_xgb_clf.pkl')

    return reg_pipeline, clf_pipeline

# SHAP analysis
def generate_shap_plot(reg_pipeline, X_train, X_test):
    preprocessor = reg_pipeline.named_steps['preprocessor']
    regressor = reg_pipeline.named_steps['regressor']

    # The preprocessor within the pipeline is already fitted during pipeline.fit(X_train, y_train).
    # No explicit check for 'fitted_feature_names_in_' is needed as get_feature_names_out()
    # will correctly retrieve feature names if the transformer is fitted.
    
    # Get processed feature names
    numerical_features_processed = numerical_features
    categorical_features_processed = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_features_processed = np.concatenate([numerical_features_processed, categorical_features_processed])

    # Transform data for SHAP explainer
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_processed)

    plt.figure()
    shap.summary_plot(shap_values, X_test_processed, feature_names=all_features_processed, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

    return explainer, all_features_processed

# Streamlit app
def main():
    st.title("🏏 Cricket Runs Prediction & Ball Recommendation App")

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, inverse_recommendations = load_data()

    reg_pipeline, clf_pipeline = train_models(X_train, y_reg_train, y_clf_train, preprocessor)
    label_encoder = joblib.load('label_encoder.pkl')
    st.success("Models trained successfully!")

    tab1, tab2 = st.tabs(["Predict Runs (Forward)", "Recommend Ball (Reverse)"])

    with tab1:
        st.header("Predict Runs from Ball Parameters")
        st.write("Predict runs from ball-tracking data using simplified cricket language")

        # --- User Inputs for Forward Prediction ---
        bowling_type = st.radio("Bowler Type", ["Pace (Fast Bowler)", "Spin (Spinner)"], key="bowling_type_forward")
        bowling_type_raw = "pace" if "Pace" in bowling_type else "spin"

        speed_options = {
            "Very Slow (0-90 KPH)": 45,
            "Slow (90-110 KPH)": 100,
            "Medium (110-120 KPH)": 115,
            "Medium-Fast (120-125 KPH)": 122.5,
            "Fast (125-130 KPH)": 127.5,
            "Very Fast (130-135 KPH)": 132.5,
            "Express (135-140 KPH)": 137.5,
            "Extreme (140-145 KPH)": 142.5,
            "Blazing (145+ KPH)": 147.5
        }
        release_speed_kph_input_label = st.selectbox("Ball Speed", list(speed_options.keys()), key="speed_forward")
        release_speed_kph_input = speed_options[release_speed_kph_input_label]


        batting_hand = st.radio("Batsman Handedness", ["Right-hand", "Left-hand"], key="batsman_forward")
        batting_hand_raw = "right" if batting_hand == "Right-hand" else "left"

        line_category_input = st.selectbox("Line of the Ball", [
            "Wide Outside Off Stump", "Outside Off Stump", "On Off Stump",
            "Middle & Off", "On Middle Stump", "Middle & Leg",
            "On Leg Stump", "Down Leg Side"
        ], key="line_forward")
        st.sidebar.markdown("### ℹ️ What does 'Line of the Ball' mean?")
        st.sidebar.markdown("""
        - **Wide Outside Off**: Ball pitching well outside off stump
        - **On Middle Stump**: Ball in line with the middle stump
        - **Down Leg Side**: Ball drifting far down leg
        """)

        length_category_input = st.selectbox("Length of the Ball", [
            "Full/Yorker", "Good Length", "Short/Back of a Length", "Bouncer"
        ], key="length_forward")
        st.sidebar.markdown("### ℹ️ What does 'Length of the Ball' mean?")
        st.sidebar.markdown("""
        - **🎯 Full / Yorker**: Ball landing near the batsman's crease
        - **✅ Good Length**: Ideal area for bowling, hard to judge
        - **⚠️ Bouncer**: Short-pitched ball rising sharply
        - **🚫 Full Toss**: No bounce before reaching batsman (Note: Model predicts actual bounce, not full toss directly)
        """)

        movement_options_display = [
            "No Movement / Straight",
            "In-swing / Off-break",
            "Out-swing / Leg-break"
        ]
        movement_label_input = st.selectbox("Movement of the Ball", movement_options_display, key="movement_forward")

        # --- Generate raw inputs based on user selection and data averages ---
        df_train_raw = X_train

        typical_values = {}
        for col in numerical_features:
            typical_values[col] = df_train_raw[col].median()

        typical_values['release_speed_kph'] = release_speed_kph_input

        if movement_label_input == "No Movement / Straight":
            typical_values['swing_angle'] = 0.0
            typical_values['deviation'] = 0.0
        else:
            avg_abs_swing = df_train_raw['swing_angle'].abs().median()
            avg_abs_dev = df_train_raw['deviation'].abs().median()

            if batting_hand_raw == "right":
                if movement_label_input == "In-swing / Off-break":
                    typical_values['swing_angle'] = avg_abs_swing
                    typical_values['deviation'] = avg_abs_dev
                elif movement_label_input == "Out-swing / Leg-break":
                    typical_values['swing_angle'] = -avg_abs_swing
                    typical_values['deviation'] = -avg_abs_dev
            else: # Left-hand batsman
                if movement_label_input == "In-swing / Off-break":
                    typical_values['swing_angle'] = -avg_abs_swing
                    typical_values['deviation'] = -avg_abs_dev
                elif movement_label_input == "Out-swing / Leg-break":
                    typical_values['swing_angle'] = avg_abs_swing
                    typical_values['deviation'] = avg_abs_dev

        avg_bounce_x_for_length = df_train_raw.groupby('length_category')['bounce_position_x'].median().to_dict()
        typical_values['bounce_position_x'] = avg_bounce_x_for_length.get(length_category_input, typical_values['bounce_position_x'])

        avg_bounce_y_for_line = df_train_raw.groupby(['line_category', 'batting_hand'])['bounce_position_y'].median().to_dict()
        key = (line_category_input, batting_hand_raw)
        typical_values['bounce_position_y'] = avg_bounce_y_for_line.get(key, typical_values['bounce_position_y'])

        input_data = pd.DataFrame({
            'release_speed_kph': [typical_values['release_speed_kph']],
            'swing_angle': [typical_values['swing_angle']],
            'deviation': [typical_values['deviation']],
            'release_position_y': [typical_values['release_position_y']],
            'release_position_z': [typical_values['release_position_z']],
            'bounce_position_y': [typical_values['bounce_position_y']],
            'bounce_position_x': [typical_values['bounce_position_x']],
            'crease_position_y': [typical_values['crease_position_y']],
            'crease_position_z': [typical_values['crease_position_z']],
            'stumps_position_y': [typical_values['stumps_position_y']],
            'stumps_position_z': [typical_values['stumps_position_z']],
            'bounce_velocity_ratio_z': [typical_values['bounce_velocity_ratio_z']],
            'release_angle': [typical_values['release_angle']],
            'drop_angle': [typical_values['drop_angle']],
            'bounce_angle': [typical_values['bounce_angle']],
            'batting_hand': [batting_hand_raw],
            'bowling_hand': [typical_values.get('bowling_hand', 'right')],
            'bowling_type': [bowling_type_raw],
            'line_category': [line_category_input],
            'length_category': [length_category_input],
            'movement_category': [movement_label_input],
            'speed_category': [
                pd.cut([release_speed_kph_input],
                       bins=[0, 90, 110, 120, 125, 130, 135, 140, 145, 200],
                       labels=['Very Slow', 'Slow', 'Medium', 'Medium-Fast', 'Fast', 'Very Fast', 'Express', 'Extreme', 'Blazing'],
                       right=False)[0]
            ]
        })

        if st.button("Predict Runs", key="predict_button"):
            reg_pred = reg_pipeline.predict(input_data)[0]
            clf_pred_encoded = clf_pipeline.predict(input_data)[0]
            clf_pred = label_encoder.inverse_transform([clf_pred_encoded])[0]

            rounded_runs = round(reg_pred, 1)

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

            st.success(f"Predicted runs (regression): {rounded_runs} → *{run_type}*")
            st.success(f"Predicted run category (classification): {clf_pred}")
            st.info(f"The regression model estimates {rounded_runs} runs likely ({run_type}), while the classifier predicts an outcome of {clf_pred}.")

            st.subheader("How this specific ball's features impacted the prediction:")
            fig_instance = plt.figure(figsize=(10, 6))
            shap.initjs()
            
            input_processed = reg_pipeline.named_steps['preprocessor'].transform(input_data)
            explainer, feature_names = generate_shap_plot(reg_pipeline, X_train, X_test)
            shap_values_instance = explainer.shap_values(input_processed)

            explanation_obj = shap.Explanation(
                values=shap_values_instance[0],
                base_values=explainer.expected_value,
                data=input_processed[0],
                feature_names=feature_names
            )
            shap.waterfall_plot(explanation_obj, show=False)
            plt.tight_layout()
            st.pyplot(fig_instance)
            plt.close(fig_instance)

        if st.checkbox("Show Model Performance", key="show_performance_forward"):
            y_reg_pred = reg_pipeline.predict(X_test)
            y_clf_pred_encoded = clf_pipeline.predict(X_test)
            y_clf_test_decoded = label_encoder.transform(y_clf_test)

            st.subheader("Regression Metrics (on Test Set)")
            st.metric("MAE", f"{mean_absolute_error(y_reg_test, y_reg_pred):.2f}")
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.2f}")
            st.metric("R²", f"{r2_score(y_reg_test, y_reg_pred):.2f}")

            st.subheader("Classification Metrics (on Test Set)")
            st.metric("Accuracy", f"{accuracy_score(y_clf_test_decoded, y_clf_pred_encoded):.2f}")
            st.metric("Precision (Weighted)", f"{precision_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted', zero_division=0):.2f}")
            st.metric("Recall (Weighted)", f"{recall_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted', zero_division=0):.2f}")
            st.metric("F1 Score (Weighted)", f"{f1_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted', zero_division=0):.2f}")

            st.subheader("Overall Feature Importance (from Regression Model)")
            if os.path.exists('shap_summary.png'):
                st.image('shap_summary.png', caption='Overall Feature Importance from Test Data')
            else:
                st.warning("SHAP summary plot is not available yet. Please run a prediction first.")

    with tab2:
        st.header("Recommend Ball Characteristics for Desired Run Outcome")
        st.write("Select a desired run outcome, and we'll suggest typical ball characteristics.")

        run_outcome_options = sorted(list(inverse_recommendations.keys())) # Sort for consistent display
        selected_run_outcome = st.selectbox("Desired Run Outcome", run_outcome_options, key="reverse_run_outcome")

        if selected_run_outcome:
            recommendation = inverse_recommendations.get(selected_run_outcome)
            if recommendation:
                st.subheader(f"To achieve a '{selected_run_outcome}', consider:")
                st.markdown(f"- **Line of the Ball:** `{recommendation['line_category']}`")
                st.markdown(f"- **Length of the Ball:** `{recommendation['length_category']}`")
                st.markdown(f"- **Movement of the Ball:** `{recommendation['movement_category']}`")
                # Format batting_hand and bowling_type for better display
                batting_hand_display = recommendation['batting_hand'].replace('right', 'Right-hand').replace('left', 'Left-hand').title()
                bowling_type_display = recommendation['bowling_type'].title()
                st.markdown(f"- **For a Batsman Handedness:** `{batting_hand_display}`")
                st.markdown(f"- **For a Bowler Type:** `{bowling_type_display}`")
                st.markdown(f"- **Typical Speed Category:** `{recommendation['speed_category']}` (around `{recommendation['release_speed_kph_median']:.1f} KPH`)")
            else:
                st.warning("No recommendations found for this run outcome.")

if __name__ == "__main__":
    main()