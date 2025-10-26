# NexGen Logistics: Predictive Delivery Optimizer

This repository contains a Streamlit web application designed for NexGen Logistics. The application uses a pre-trained Random Forest model to predict the probability of delivery delays and provides an interactive dashboard for contextual analytics and performance analysis.

## 🚀 Live Demo

The application is deployed and accessible at the following URL:

**[https://case-study-of-company-nexgen-6.onrender.com](https://case-study-of-company-nexgen-6.onrender.com)**

## ✨ Features

*   **Delay Prediction:** Enter order details to get a real-time prediction of whether a delivery is likely to be on-time or delayed.
*   **Probability Score:** View the confidence score for on-time delivery and the probability of a delay.
*   **Cost Analytics:** Automatically calculates and displays key operational metrics like Total Cost, Cost per KM, and Value-to-Cost Ratio.
*   **Interactive Visualizations:**
    *   **Prediction Distribution:** A pie chart showing the breakdown of on-time vs. delayed probability.
    *   **Cost Breakdown:** A bar chart visualizing the components of the total operational cost (Labor, Fuel, Maintenance).
    *   **Traffic Sensitivity Analysis:** An area chart showing how the risk of delay changes with varying traffic conditions.
    *   **Historical Context Map:** A scatter plot that places the current order prediction within the context of historical data, plotted against Order Value and Distance.
*   **Historical Data Dashboard:**
    *   Filter a mock historical dataset by priority level and order value.
    *   View and download the filtered data as a CSV file.

## ⚙️ How It Works

The application uses a machine learning model to predict delivery outcomes.

1.  **Input:** The user provides details for a specific order, including value, distance, priority, product category, origin, and associated costs.
2.  **Feature Engineering:** The application processes the raw input by creating new features like `Total_Cost`, `Cost_Per_KM`, and `Value_to_Cost_Ratio`.
3.  **Data Preparation:** Categorical features (e.g., 'Priority Level', 'Origin Warehouse') are one-hot encoded. The data is then aligned with the feature set used during model training (`feature_columns.pkl`).
4.  **Prediction:** The prepared features are fed into the pre-trained Random Forest classifier (`delay_model.pkl`) to predict the outcome (1 for Delayed, 0 for On-Time) and the associated probabilities.
5.  **Visualization:** The results are displayed through a series of interactive charts created with Plotly, providing actionable insights to the user.

## 🛠️ Technical Stack

*   **Backend:** Python
*   **Web Framework:** Streamlit
*   **Machine Learning:** Scikit-learn
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Plotly
*   **Model:** Pre-trained Random Forest Classifier (`.pkl`)

## 📋 Project Structure

```
.
├── case_study_ofi/
│   ├── app.py                  # Main Streamlit application source code
│   ├── delay_model.pkl         # Pre-trained Random Forest model
│   ├── feature_columns.pkl     # List of feature columns for the model
│   ├── requirements.txt        # Python dependencies
│   └── runtime.txt             # Python version for deployment
└── README.md
```

## 🔧 Local Setup and Installation

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/siddharthjais299/Case-study-of-company-NexGen.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd Case-study-of-company-NexGen/case_study_ofi
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

The application will open in your default web browser.
