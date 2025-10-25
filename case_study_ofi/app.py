import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px  # Import Plotly for interactive charts

st.set_page_config(page_title="NexGen Logistics", layout="wide")

st.title("🚚 NexGen Logistics Predictive Delivery Optimizer")
st.markdown(
    "This app predicts delivery delays using a trained Random Forest model and provides **contextual analytics**.")

MODEL_FILENAME = "delay_model.pkl"
FEATURES_FILENAME = "feature_columns.pkl"

# Load Trained Assets 
model = None
feature_cols = None

try:
    # 1. Load the Model
    with open(MODEL_FILENAME, "rb") as f:
        model = pickle.load(f)
    # 2. Load the Feature Columns list
    with open(FEATURES_FILENAME, "rb") as f:
        feature_cols = pickle.load(f)
    st.sidebar.success("✅ Model and Feature Definitions loaded successfully!")
except FileNotFoundError:
    st.error(
        f" Required files not found! Please ensure **{MODEL_FILENAME}** and **{FEATURES_FILENAME}** are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f" An error occurred during file loading: {e}")
    st.stop()


# Mock Historical Data Generator (for context visualization) 
@st.cache_data
def generate_contextual_data():
    """Generates a small mock dataset for scatter plot context."""
    np.random.seed(42)
    data_size = 150
    return pd.DataFrame({
        'Order Value (₹)': np.random.randint(500, 50000, data_size),
        'Distance (km)': np.random.randint(10, 2000, data_size),
        'Priority Level': np.random.choice(['Express', 'Standard', 'Economy'], data_size, p=[0.3, 0.5, 0.2]),
        'Historical Risk Score': np.random.uniform(0.1, 0.9, data_size).round(2),
        'Order ID': [f'ORD-{1000 + i}' for i in range(data_size)]
    })


historical_df = generate_contextual_data()

# Sample Input Section (Prediction) 
st.subheader("Enter Order Details for Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    order_value = st.number_input("Order Value (₹)", 500, 50000, 15000, key='pred_value')
    distance = st.number_input("Distance (km)", 10, 2000, 500, key='pred_distance')
    traffic_delay = st.slider("Traffic Delay (hours)", 0.0, 5.0, 1.0, 0.1, key='pred_traffic')

with col2:
    priority = st.selectbox("Priority Level", ['Express', 'Standard', 'Economy'], key='pred_priority')
    category = st.selectbox("Product Category", ['Electronics', 'Fashion', 'Healthcare', 'Food & Beverage'],
                            key='pred_category')
    warehouse = st.selectbox("Origin Warehouse", ['Mumbai', 'Delhi', 'Chennai', 'Kolkata', 'Bangalore'],
                             key='pred_warehouse')

with col3:
    weather = st.selectbox("Weather Impact (1=High, 0=Low)", [0, 1], key='pred_weather')
    labor = st.number_input("Labor Cost (₹)", 100, 1000, 250, key='pred_labor')
    fuel = st.number_input("Fuel Cost (₹)", 1000, 5000, 2500, key='pred_fuel')
    maint = st.number_input("Maintenance Cost (₹)", 100, 1000, 500, key='pred_maint')


#  Prediction and Feature Preparation Logic (Fixing scope shadowing) 
def prepare_and_predict(input_data_dict, traffic_override=None):
    """
    Replicates the training pipeline steps to prepare data for prediction.
    Allows overriding traffic_delay for sensitivity analysis.
    """

    # 1. Create a DataFrame from the single input
    input_df = pd.DataFrame([input_data_dict])

    # Apply traffic override if provided
    if traffic_override is not None:
        input_df['Traffic Delay (hours)'] = traffic_override

    # 2. Replicate Feature Engineering (CRITICAL)
    # NOTE: Using temporary variable names (e.g., _total_cost) prevents shadowing warnings
    _total_cost = input_df['Labor Cost'].iloc[0] + input_df['Fuel Cost'].iloc[0] + input_df['Maintenance Cost'].iloc[0]
    _cost_per_km = _total_cost / (input_df['Distance (km)'].iloc[0] + 1)
    _value_to_cost_ratio = input_df['Order Value'].iloc[0] / (_total_cost + 1)

    input_df['Total_Cost'] = _total_cost
    input_df['Cost_Per_KM'] = _cost_per_km
    input_df['Value_to_Cost_Ratio'] = _value_to_cost_ratio

    # 3. Replicate One-Hot Encoding (CRITICAL)
    categorical_cols = ['Priority Level', 'Origin Warehouse', 'Product Category']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # 4. Align Columns with the training data (CRITICAL FIX)
    missing_cols = set(feature_cols) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0

    input_df = input_df[feature_cols]

    # 5. Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Return prediction, probabilities, and cost metrics
    return prediction, probability[1], probability[0], _total_cost, _cost_per_km, _value_to_cost_ratio, input_df


# Initialize prediction variables outside the button block to prevent 'unbound' warning
result, prob_delayed, prob_ontime = None, None, None

# Prediction and Visualization Button 
if st.button("🚀 Predict Delivery Outcome & Analyze", type="primary"):

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
        # Run prediction for the current input
        result, prob_delayed, prob_ontime, total_cost, cost_per_km, value_to_cost_ratio, feature_df = prepare_and_predict(
            raw_input_data)

        st.subheader("Prediction Result & Analytical Insights")

        # Row 1: Prediction and Cost Metrics
        col_res, col_metric_1, col_metric_2, col_metric_3 = st.columns(4)
        with col_res:
            if result == 1:
                st.error(f"🚨 **HIGH RISK of DELAY**")
                st.metric("Delay Probability", f"{prob_delayed * 100:.2f}%")
            else:
                st.success(f"✅ **LIKELY ON-TIME**")
                st.metric("On-Time Confidence", f"{prob_ontime * 100:.2f}%")

        with col_metric_1:
            st.metric("Total Cost", f"₹ {total_cost:,.0f}")
        with col_metric_2:
            st.metric("Cost per KM", f"₹ {cost_per_km:.2f}")
        with col_metric_3:
            st.metric("Value-to-Cost Ratio", f"{value_to_cost_ratio:.2f}")

        st.markdown("---")

        #  Row 2: Visualizations 
        vis_col1, vis_col2 = st.columns(2)

        
        # VISUALIZATION 1: Probability Distribution (Interactive Pie Chart)
    
        with vis_col1:
            st.markdown("##### 1. Prediction Outcome Distribution")
            prob_df = pd.DataFrame({
                'Status': ['Delayed', 'On-Time'],
                'Probability': [prob_delayed, prob_ontime],
            })
            fig_pie = px.pie(
                prob_df,
                values='Probability',
                names='Status',
                title='Chance of Delay vs. On-Time Delivery',
                color='Status',
                color_discrete_map={'Delayed': '#EF5350', 'On-Time': '#66BB6A'}
            )
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        
        # VISUALIZATION 2: Cost Component Breakdown (Interactive Bar Chart)
        # -----------------------------------------------------
        with vis_col2:
            st.markdown("##### 2. Operational Cost Breakdown")
            cost_df = pd.DataFrame({
                'Cost Type': ['Labor Cost', 'Fuel Cost', 'Maintenance Cost'],
                'Amount (₹)': [labor, fuel, maint]
            })
            fig_bar = px.bar(
                cost_df,
                x='Cost Type',
                y='Amount (₹)',
                color='Cost Type',
                title=f'Cost Components (Total: ₹ {total_cost:,.0f})',
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        
        # VISUALIZATION 4: Traffic Sensitivity Analysis (Interactive Line Chart)
        
        st.markdown("##### 4. Traffic Delay Sensitivity (Hypothetical Delay Risk)")

        traffic_scenarios = np.linspace(0.0, 5.0, 10).round(1)
        sensitivity_data = []

        for delay in traffic_scenarios:
            # We can skip the prediction logic if prob_delayed is None, but since
            # the button guarantees the prediction runs, this is fine.
            # Using the full return from prepare_and_predict is safest.
            _, prob_delayed_scenario, _, _, _, _, _ = prepare_and_predict(raw_input_data, traffic_override=delay)
            sensitivity_data.append(
                {'Traffic Delay (hours)': delay, 'Predicted Delay Risk (%)': prob_delayed_scenario * 100})

        sensitivity_df = pd.DataFrame(sensitivity_data)

        fig_line = px.area(
            sensitivity_df,
            x='Traffic Delay (hours)',
            y='Predicted Delay Risk (%)',
            title='Delay Risk vs. Traffic Delay (Other Factors Fixed)',
            line_shape='spline'
        )
        fig_line.add_vline(x=traffic_delay, line_width=2, line_dash="dash", line_color="orange",
                           annotation_text="Current Input")
        st.plotly_chart(fig_line, use_container_width=True)


    except Exception as e:
        st.error(f"A prediction processing error occurred. Details: {e}")

st.markdown("---")

# New Section: Historical Data Analytics Dashboard 
st.header("📊 Historical Data Analytics Dashboard")
st.markdown("Filter the mock historical order data to analyze past performance and risk.")

# Filters (User Inputs/Selections) 
filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    # Filter 1: Priority Level Selection
    selected_priorities = st.multiselect(
        "Filter by Priority Level",
        options=historical_df['Priority Level'].unique(),
        default=historical_df['Priority Level'].unique()
    )

with filter_col2:
    # Filter 2: Order Value Range Slider
    min_val, max_val = int(historical_df['Order Value (₹)'].min()), int(historical_df['Order Value (₹)'].max())
    value_range = st.slider(
        "Filter by Order Value (₹)",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )

# Apply Filters Dynamically 
filtered_df = historical_df[
    historical_df['Priority Level'].isin(selected_priorities) &
    (historical_df['Order Value (₹)'] >= value_range[0]) &
    (historical_df['Order Value (₹)'] <= value_range[1])
    ]

with filter_col3:
    st.info(f"Showing **{len(filtered_df)}** of {len(historical_df)} historical orders.")


    #  Download Functionality 
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')


    csv = convert_df_to_csv(filtered_df)

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_logistics_data.csv',
        mime='text/csv',
        help='Exports the data currently visible after applying filters.'
    )

# -----------------------------------------------------
# VISUALIZATION 3 (Modified): Order Contextual Risk Map (Interactive Scatter Plot)
# Now dynamically responds to filters.
# -----------------------------------------------------
st.markdown("---")
st.markdown("##### 3. Order Value vs. Distance (Historical Context)")
st.dataframe(
    filtered_df[['Order ID', 'Priority Level', 'Order Value (₹)', 'Distance (km)', 'Historical Risk Score']].head(),
    use_container_width=True)

# Combine historical and current point (Fixing unbound local variable warning)
combined_df = filtered_df.copy()
combined_df['Type'] = 'Historical Orders'

# Only display current prediction if it was successfully run
if result is not None:
    current_order_df = pd.DataFrame({
        'Order Value (₹)': [order_value],
        'Distance (km)': [distance],
        'Priority Level': [priority],
        # prob_delayed is guaranteed to be set if result is set, as they are assigned together
        'Historical Risk Score': [prob_delayed],
        'Order ID': ['CURRENT-PREDICTION'],
        'Type': ['Current Order']
    })
    combined_df = pd.concat([combined_df, current_order_df])

fig_scatter = px.scatter(
    combined_df,
    x='Distance (km)',
    y='Order Value (₹)',
    color='Type',
    size='Historical Risk Score',
    hover_data=['Priority Level', 'Order ID', 'Historical Risk Score'],
    title=f'Filtered Orders: {len(filtered_df)} items analyzed',
    color_discrete_map={'Historical Orders': 'lightgrey', 'Current Order': 'darkblue'}
)
fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
st.caption("Developed by Aks Jaiswal. Case study of OFI company.")
