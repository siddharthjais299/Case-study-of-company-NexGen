import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="NexGen Logistics", layout="wide")

st.title("🚚 NexGen Logistics Predictive Delivery Optimizer")
st.write("This app predicts delivery delays using mock logistics data and a trained Random Forest model.")

MODEL_FILENAME = "delay_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"

#  Load Trained Assets
model = None
feature_cols = None

try:
    # 1. Load the Model
    with open(MODEL_FILENAME, "rb") as f:
        model = pickle.load(f)
    # 2. Load the Feature Columns list
    with open(FEATURES_FILENAME, "rb") as f:
        feature_cols = pickle.load(f)
    st.success("✅ Model and Feature Definitions loaded successfully!")
except FileNotFoundError:
    st.error(
        f"❌ Required files not found! Please ensure **{MODEL_FILENAME}** and **{FEATURES_FILENAME}** are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ An error occurred during file loading: {e}")
    st.stop()

# Sample Input Section
st.subheader("Enter Order Details for Prediction")

# Use st.columns for a clean, responsive layout
col1, col2, col3 = st.columns(3)

with col1:
    order_value = st.number_input("Order Value (₹)", 500, 50000, 15000)
    distance = st.number_input("Distance (km)", 10, 2000, 500)
    traffic_delay = st.slider("Traffic Delay (hours)", 0.0, 5.0, 1.0, 0.1)

with col2:
    priority = st.selectbox("Priority Level", ['Express', 'Standard', 'Economy'])
    category = st.selectbox("Product Category", ['Electronics', 'Fashion', 'Healthcare', 'Food & Beverage'])
    warehouse = st.selectbox("Origin Warehouse", ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore'])

with col3:
    weather = st.selectbox("Weather Impact (1=High, 0=Low)", [0, 1])
    labor = st.number_input("Labor Cost (₹)", 100, 1000, 250)
    fuel = st.number_input("Fuel Cost (₹)", 1000, 5000, 2500)
    maint = st.number_input("Maintenance Cost (₹)", 100, 1000, 500)


# Prediction Logic Function
def prepare_and_predict(input_data_dict):
    """Replicates the training pipeline steps to prepare data for prediction."""

    # 1. Create a DataFrame from the single input
    input_df = pd.DataFrame([input_data_dict])

    # 2. Replicate Feature Engineering (CRITICAL)
    input_df['Total_Cost'] = input_df['Labor Cost'] + input_df['Fuel Cost'] + input_df['Maintenance Cost']
    input_df['Cost_Per_KM'] = input_df['Total_Cost'] / (input_df['Distance (km)'] + 1)
    input_df['Value_to_Cost_Ratio'] = input_df['Order Value'] / (input_df['Total_Cost'] + 1)

    # 3. Replicate One-Hot Encoding (CRITICAL)
    categorical_cols = ['Priority Level', 'Origin Warehouse', 'Product Category']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # 4. Align Columns with the training data (CRITICAL FIX)
    # This step ensures the input DataFrame has all the columns the model expects,
    # even if they are zero (e.g., if Priority_Level_Express was not selected)

    # Add missing columns (as zeros)
    missing_cols = set(feature_cols) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0

    # Drop columns not needed (only possible if a new category was introduced)
    input_df = input_df[feature_cols]

    # 5. Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Return (prediction, probability of being delayed, probability of being on-time)
    return prediction, probability[1], probability[0]


# Predict Button
if st.button("🚀 Predict Delivery Outcome", type="primary"):

    # Compile raw input data
    raw_input_data = {
        'Order Value': order_value,
        'Distance (km)': distance,
        'Traffic Delay (hours)': traffic_delay,
        'Weather Impact': weather,
        'Labor Cost': labor,
        'Fuel Cost': fuel,
        'Maintenance Cost': maint,
        'Priority Level': priority,
        'Product Category': category,
        'Origin Warehouse': warehouse,
    }

    try:
        result, prob_delayed, prob_ontime = prepare_and_predict(raw_input_data)

        # Display Results
        st.subheader("Prediction Result")

        if result == 1:
            st.error(f"🚨 **High Risk of Delay**")
            st.metric("Delay Probability", f"{prob_delayed * 100:.2f}%")
            st.warning("Immediate action is recommended: Consider rerouting or upgrading priority service.")
        else:
            st.success(f"✅ **Likely On-Time Delivery**")
            st.metric("On-Time Confidence", f"{prob_ontime * 100:.2f}%")
            st.info("The current plan appears optimized. Monitor traffic conditions.")

        # Optional: Show a bar chart of the probabilities
        prob_df = pd.DataFrame({
            'Status': ['Delayed', 'On-Time'],
            'Probability': [prob_delayed, prob_ontime]
        })
        st.bar_chart(prob_df.set_index('Status'))

    except Exception as e:
        st.error(f"A prediction processing error occurred. Check input values. Details: {e}")

st.markdown("---")
st.caption("Developed by Aks Jaiswal. case study of OFI company.")
