import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Title
st.title("🌾 Crop Yield Prediction Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Crop Yield Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())
    
    # Encode categorical variable
    encoder = LabelEncoder()
    df["Crop Type"] = encoder.fit_transform(df["Crop Type"])
    
    # Define features and target
    X = df[["Year", "Temperature (°C)", "Rainfall (mm)", "Soil Quality (pH)", "Crop Type", "Fertilizer Used (kg/ha)"]]
    y = df["Crop Yield (kg/ha)"]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)
    st.write(f"### Model Accuracy (R² Score): {accuracy:.2f}")
    
    # Feature Importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    fig, ax = plt.subplots()
    feature_importance.sort_values().plot(kind='barh', ax=ax)
    plt.title("Feature Importance")
    st.pyplot(fig)
    
    # Prediction Input Form
    st.write("## Predict Crop Yield")
    year = st.number_input("Year", min_value=2018, max_value=2025, step=1)
    temp = st.number_input("Temperature (°C)", min_value=10.0, max_value=50.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=500.0, step=0.1)
    soil_pH = st.number_input("Soil Quality (pH)", min_value=4.0, max_value=9.0, step=0.1)
    crop_type = st.selectbox("Crop Type", df["Crop Type"].unique())
    fertilizer = st.number_input("Fertilizer Used (kg/ha)", min_value=30.0, max_value=300.0, step=1.0)
    
    # Predict Button
if st.button("Predict Crop Yield"):
    # Ensure encoder has seen all crop types
    encoder.classes_ = np.array(df["Crop Type"].unique())

    # Handle unseen crop types
    if crop_type in encoder.classes_:
        crop_encoded = encoder.transform([crop_type])[0]
    else:
        crop_encoded = -1  # Assign an unknown label to avoid errors

    # Prepare input data
    input_data = np.array([[year, temp, rainfall, soil_pH, crop_encoded, fertilizer]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    st.write(f"### Predicted Crop Yield: {prediction:.2f} kg/ha")


