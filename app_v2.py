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

# Define features globally so they are accessible throughout the script
# This fixes the NameError in the main function.
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
    # These will be dynamically created and added in load_data
    'line_category', 'length_category', 'movement_category', 'speed_category'
]


# --- Helper Functions for Feature Engineering ---
def categorize_line(bounce_y, batting_hand):
    """Categorizes the line of the ball into cricketing terms."""
    # Define thresholds for a right-hand batsman. Adjust for left-hand later.
    # These thresholds are approximate and might need tuning based on data distribution
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
        if bounce_y >= 0.4: return 'Wide Outside Off Stump' # +y is off for leftie
        elif 0.25 <= bounce_y < 0.4: return 'Outside Off Stump'
        elif 0.1 <= bounce_y < 0.25: return 'On Off Stump'
        elif 0.05 <= bounce_y < 0.1: return 'Middle & Off'
        elif -0.05 <= bounce_y < 0.05: return 'On Middle Stump'
        elif -0.15 <= bounce_y < -0.05: return 'Middle & Leg'
        elif -0.3 <= bounce_y < -0.15: return 'On Leg Stump'
        else: return 'Down Leg Side' # -y is leg for leftie

def categorize_length(bounce_x):
    """Categorizes the length of the ball into cricketing terms."""
    # These thresholds are approximate and might need tuning based on data distribution
    # and cricketing definitions
    if bounce_x <= 1.5: return 'Full/Yorker' # Close to or behind crease (1.22m)
    elif 1.5 < bounce_x <= 5.0: return 'Good Length' # Typical good length deliveries
    elif 5.0 < bounce_x <= 8.0: return 'Short/Back of a Length' # Shorter pitched
    else: return 'Bouncer' # Very short, usually over 8m

def categorize_movement(swing_angle, deviation, bowling_type, batting_hand):
    """Categorizes swing/seam/spin movement based on handedness and bowling type."""
    # Define thresholds for significant movement
    movement_threshold_swing = 0.5 # degrees
    movement_threshold_dev = 0.5 # degrees

    is_swinging = abs(swing_angle) > movement_threshold_swing
    is_deviating = abs(deviation) > movement_threshold_dev

    if not is_swinging and not is_deviating:
        return 'No Movement / Straight'

    if bowling_type == 'pace':
        if batting_hand == 'right':
            if swing_angle > movement_threshold_swing or deviation > movement_threshold_dev:
                return 'In-swing / Leg-cutter' # Positive means left-to-right (in-swing for rightie)
            elif swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'Out-swing / Off-cutter' # Negative means right-to-left (out-swing for rightie)
        else: # Left-hand batsman
            if swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'In-swing / Leg-cutter' # Negative means right-to-left (in-swing for leftie)
            elif swing_angle > movement_threshold_swing or deviation > movement_threshold_dev:
                return 'Out-swing / Off-cutter' # Positive means left-to-right (out-swing for leftie)
    elif bowling_type == 'spin':
        if batting_hand == 'right':
            if swing_angle > movement_threshold_swing or deviation > movement_threshold_dev:
                return 'Off-break / Googly' # Positive means left-to-right (off-break for rightie)
            elif swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'Leg-break / Flipper' # Negative means right-to-left (leg-break for rightie)
        else: # Left-hand batsman
            if swing_angle < -movement_threshold_swing or deviation < -movement_threshold_dev:
                return 'Off-break / Googly' # Negative means right-to-left (off-break for leftie)
            elif swing_angle > movement_threshold_swing or deviation > -movement_threshold_dev:
                return 'Leg-break / Flipper' # Positive means left-to-right (leg-break for leftie)
    return 'Other Movement' # Fallback for edge cases

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
    else: # This handles '5' (no longer in data), or any other unexpected values
        return 'Other Runs'


# Data loading and preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df = df.drop(columns=['wicket']) # Dropping 'wicket' as per original code
    df = df[df['runs'] != 5] # Exclude 'runs' == 5 as per original code

    # --- New Feature Engineering ---
    df['line_category'] = df.apply(lambda row: categorize_line(row['bounce_position_y'], row['batting_hand']), axis=1)
    df['length_category'] = df['bounce_position_x'].apply(categorize_length)
    df['movement_category'] = df.apply(lambda row: categorize_movement(row['swing_angle'], row['deviation'], row['bowling_type'], row['batting_hand']), axis=1)
    df['speed_category'] = pd.cut(df['release_speed_kph'],
                                   bins=[0, 90, 110, 120, 125, 130, 135, 140, 145, 200], # Adjusted bins
                                   labels=['Very Slow', 'Slow', 'Medium', 'Medium-Fast', 'Fast', 'Very Fast', 'Express', 'Extreme', 'Blazing'],
                                   right=False) # Ensure labels match the bins

    # X will now include original numerical + original categorical + new engineered categorical features
    # Ensure all features that the preprocessor expects are in X.
    # The global `numerical_features` and `categorical_features` are used here.
    all_input_features = numerical_features + [f for f in categorical_features if f not in ['batting_hand', 'bowling_hand', 'bowling_type']] + ['batting_hand', 'bowling_hand', 'bowling_type']
    X = df[all_input_features] # Select columns based on the global lists

    y_reg = df['runs']
    y_clf_raw = df['runs'] # Keep original runs for classification to map to new categories
    y_clf = y_clf_raw.apply(categorize_runs_for_classification) # Apply new categorization


    # --- Preprocessing Pipeline ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) # Added StandardScaler for numerical features
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features), # Use global numerical_features
            ('cat', categorical_transformer, categorical_features) # Use global categorical_features
        ])

    # Split data AFTER feature engineering but BEFORE fitting preprocessor
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42)

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor

# Model training
def train_models(X_train, y_reg_train, y_clf_train, preprocessor):
    # --- Regression Model Training ---
    # Example: Hyperparameter tuning for XGBRegressor
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, max_depth=5)) # Example tuned params
    ])
    reg_pipeline.fit(X_train, y_reg_train)

    # --- Classification Model Training ---
    label_encoder = LabelEncoder()
    y_clf_train_encoded = label_encoder.fit_transform(y_clf_train)
    num_classes = len(label_encoder.classes_)

    # Example: Hyperparameter tuning for XGBClassifier
    clf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, objective='multi:softprob', num_class=num_classes,
                                     n_estimators=500, learning_rate=0.05, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')) # Example tuned params
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

    # Get feature names after preprocessing
    # This requires fitting the preprocessor to X_train to get correct one-hot encoded names
    # The preprocessor within the pipeline is already fitted during pipeline.fit(X_train, y_train)
    # However, if this function is called before a full pipeline fit, ensure it's fitted.
    # For robust usage, ensure preprocessor is fitted *before* getting feature names for transform.
    # This explicit fit might be redundant if reg_pipeline.fit already happened, but ensures it.
    if not hasattr(preprocessor, 'named_transformers_') or not preprocessor.named_transformers_['cat'].named_steps['onehot'].fitted_feature_names_in_:
         preprocessor.fit(X_train)
    
    # Get processed feature names
    numerical_features_processed = numerical_features # These names remain the same
    categorical_features_processed = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        [f for f in categorical_features if f not in ['batting_hand', 'bowling_hand', 'bowling_type']] + ['batting_hand', 'bowling_hand', 'bowling_type'] # Ensure correct original cat features are passed
    )
    all_features_processed = np.concatenate([numerical_features_processed, categorical_features_processed])

    # Transform data for SHAP explainer
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_test_processed)

    plt.figure()
    # Use feature_names from all_features_processed
    shap.summary_plot(shap_values, X_test_processed, feature_names=all_features_processed, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()

    return explainer, all_features_processed

# Streamlit app
def main():
    st.title("🏏 Cricket Runs Prediction")
    st.write("Predict runs from ball-tracking data using simplified cricket language")

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor = load_data()

    reg_pipeline, clf_pipeline = train_models(X_train, y_reg_train, y_clf_train, preprocessor)
    label_encoder = joblib.load('label_encoder.pkl')
    st.success("Models trained successfully!")

    # --- User Inputs (Now Reflecting Feature Engineering) ---
    bowling_type = st.radio("Bowler Type", ["Pace (Fast Bowler)", "Spin (Spinner)"])
    bowling_type_raw = "pace" if "Pace" in bowling_type else "spin"

    speed_options = {
        "Very Slow (0-90 KPH)": 45, # Centered value for bin
        "Slow (90-110 KPH)": 100,
        "Medium (110-120 KPH)": 115,
        "Medium-Fast (120-125 KPH)": 122.5,
        "Fast (125-130 KPH)": 127.5,
        "Very Fast (130-135 KPH)": 132.5,
        "Express (135-140 KPH)": 137.5,
        "Extreme (140-145 KPH)": 142.5,
        "Blazing (145+ KPH)": 147.5
    }
    release_speed_kph_input_label = st.selectbox("Ball Speed", list(speed_options.keys()))
    release_speed_kph_input = speed_options[release_speed_kph_input_label]


    batting_hand = st.radio("Batsman Handedness", ["Right-hand", "Left-hand"])
    batting_hand_raw = "right" if batting_hand == "Right-hand" else "left"

    line_category_input = st.selectbox("Line of the Ball", [
        "Wide Outside Off Stump", "Outside Off Stump", "On Off Stump",
        "Middle & Off", "On Middle Stump", "Middle & Leg",
        "On Leg Stump", "Down Leg Side"
    ])
    st.sidebar.markdown("### ℹ️ What does 'Line of the Ball' mean?")
    st.sidebar.markdown("""
    - **Wide Outside Off**: Ball pitching well outside off stump
    - **On Middle Stump**: Ball in line with the middle stump
    - **Down Leg Side**: Ball drifting far down leg
    """)

    length_category_input = st.selectbox("Length of the Ball", [
        "Full/Yorker", "Good Length", "Short/Back of a Length", "Bouncer" # Removed Full Toss as it's a projected value
    ])
    st.sidebar.markdown("### ℹ️ What does 'Length of the Ball' mean?")
    st.sidebar.markdown("""
    - **🎯 Full / Yorker**: Ball landing near the batsman's crease
    - **✅ Good Length**: Ideal area for bowling, hard to judge
    - **⚠️ Bouncer**: Short-pitched ball rising sharply
    - **🚫 Full Toss**: No bounce before reaching batsman (Note: Model predicts actual bounce, not full toss directly)
    """)

    # Simplified movement options (will be mapped back to raw swing/deviation for input)
    movement_options_display = [
        "No Movement / Straight",
        "In-swing / Off-break",
        "Out-swing / Leg-break"
    ]
    movement_label_input = st.selectbox("Movement of the Ball", movement_options_display)

    # --- Generate raw inputs based on user selection and data averages ---
    # Get representative values from the training data for other features
    df_train_raw = X_train # X_train is the raw dataframe before preprocessing pipeline

    # Calculate means/medians for numerical features from training data for 'typical' ball
    typical_values = {}
    for col in numerical_features: # Use the globally defined numerical_features here
        typical_values[col] = df_train_raw[col].median() # Median is robust to outliers

    # Overwrite the few chosen by user input
    typical_values['release_speed_kph'] = release_speed_kph_input

    # Map movement_label_input back to swing_angle/deviation based on typical values and general direction
    if movement_label_input == "No Movement / Straight":
        typical_values['swing_angle'] = 0.0
        typical_values['deviation'] = 0.0
    else:
        avg_abs_swing = df_train_raw['swing_angle'].abs().median() # Use median
        avg_abs_dev = df_train_raw['deviation'].abs().median() # Use median

        # Adjust the sign based on batting hand and movement type
        if batting_hand_raw == "right":
            if movement_label_input == "In-swing / Off-break":
                typical_values['swing_angle'] = avg_abs_swing # Positive for in-swing to rightie
                typical_values['deviation'] = avg_abs_dev # Positive for off-break to rightie
            elif movement_label_input == "Out-swing / Leg-break":
                typical_values['swing_angle'] = -avg_abs_swing # Negative for out-swing to rightie
                typical_values['deviation'] = -avg_abs_dev # Negative for leg-break to rightie
        else: # Left-hand batsman
            if movement_label_input == "In-swing / Off-break":
                typical_values['swing_angle'] = -avg_abs_swing # Negative for in-swing to leftie
                typical_values['deviation'] = -avg_abs_dev # Negative for off-break to leftie
            elif movement_label_input == "Out-swing / Leg-break":
                typical_values['swing_angle'] = avg_abs_swing # Positive for out-swing to leftie
                typical_values['deviation'] = avg_abs_dev # Positive for leg-break to leftie

    # Get typical bounce_x for selected length category from the training data
    avg_bounce_x_for_length = df_train_raw.groupby('length_category')['bounce_position_x'].median().to_dict()
    typical_values['bounce_position_x'] = avg_bounce_x_for_length.get(length_category_input, typical_values['bounce_position_x'])

    # Get typical bounce_y for selected line category from the training data, considering batting hand
    # This requires a more granular grouping to get representative 'bounce_position_y' values
    # For simplicity, we'll try to find the median for line_category and batting_hand combination
    avg_bounce_y_for_line = df_train_raw.groupby(['line_category', 'batting_hand'])['bounce_position_y'].median().to_dict()
    key = (line_category_input, batting_hand_raw)
    typical_values['bounce_position_y'] = avg_bounce_y_for_line.get(key, typical_values['bounce_position_y'])

    # Create the input DataFrame with all required features (original raw + new engineered)
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
        'bowling_hand': [typical_values.get('bowling_hand', 'right')], # Default to 'right' if not in typical_values
        'bowling_type': [bowling_type_raw],
        'line_category': [line_category_input],
        'length_category': [length_category_input],
        'movement_category': [movement_label_input],
        # The speed category for input should be derived from the selected kph input, not another selectbox
        'speed_category': [
            pd.cut([release_speed_kph_input],
                   bins=[0, 90, 110, 120, 125, 130, 135, 140, 145, 200],
                   labels=['Very Slow', 'Slow', 'Medium', 'Medium-Fast', 'Fast', 'Very Fast', 'Express', 'Extreme', 'Blazing'],
                   right=False)[0]
        ]
    })

    if st.button("Predict Runs"):
        # Make predictions
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
        st.success(f"Predicted run category (classification): {clf_pred}")

        # Combined summary
        st.info(f"The regression model estimates {rounded_runs} runs likely ({run_type}), while the classifier predicts an outcome of {clf_pred}.")

        # Generate and display SHAP plot for the *specific input*
        explainer, feature_names = generate_shap_plot(reg_pipeline, X_train, X_test) # Pass X_train for fitting preprocessor
        input_processed = reg_pipeline.named_steps['preprocessor'].transform(input_data)
        shap_values_instance = explainer.shap_values(input_processed)

        st.subheader("How this specific ball's features impacted the prediction:")
        fig_instance = plt.figure(figsize=(10, 6)) # Use plt.figure() for explicit figure creation
        shap.initjs() # For JS visualization if running in notebook, not strictly needed for st.pyplot
        
        explanation_obj = shap.Explanation(
            values=shap_values_instance[0], # Assuming a single prediction, so [0]
            base_values=explainer.expected_value,
            data=input_processed[0], # Corresponding processed input data point
            feature_names=feature_names
        )
        shap.waterfall_plot(explanation_obj, show=False)
        plt.tight_layout()
        st.pyplot(fig_instance) # Pass the created figure object
        plt.close(fig_instance) # Close plot to prevent display issues

    if st.checkbox("Show Model Performance"):
        y_reg_pred = reg_pipeline.predict(X_test)
        y_clf_pred_encoded = clf_pipeline.predict(X_test)
        y_clf_test_decoded = label_encoder.transform(y_clf_test) # Ensure y_clf_test is encoded for metrics

        st.subheader("Regression Metrics (on Test Set)")
        st.metric("MAE", f"{mean_absolute_error(y_reg_test, y_reg_pred):.2f}")
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.2f}")
        st.metric("R²", f"{r2_score(y_reg_test, y_reg_pred):.2f}")

        st.subheader("Classification Metrics (on Test Set)")
        # Use average='weighted' for multi-class classification
        st.metric("Accuracy", f"{accuracy_score(y_clf_test_decoded, y_clf_pred_encoded):.2f}")
        st.metric("Precision (Weighted)", f"{precision_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted', zero_division=0):.2f}")
        st.metric("Recall (Weighted)", f"{recall_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted', zero_division=0):.2f}")
        st.metric("F1 Score (Weighted)", f"{f1_score(y_clf_test_decoded, y_clf_pred_encoded, average='weighted', zero_division=0):.2f}")

        # Display overall SHAP summary plot if it exists
        st.subheader("Overall Feature Importance (from Regression Model)")
        if os.path.exists('shap_summary.png'):
            st.image('shap_summary.png', caption='Overall Feature Importance from Test Data')
        else:
            st.warning("SHAP summary plot is not available yet. Please run a prediction first.")

if __name__ == "__main__":
    main()