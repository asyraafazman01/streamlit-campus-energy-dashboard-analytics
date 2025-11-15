import streamlit as st
import pandas as pd
import plotly.express as px
import time
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set the page configuration (this should be the first Streamlit command)
st.set_page_config(page_title="Campus Energy Dashboard", layout="wide")

# Cache data loading for performance
@st.cache_data
def load_data():
    # Load the data files with limited rows for testing purposes
    energy = pd.read_csv('data/building_consumption.csv', nrows=1000, parse_dates=["timestamp"])
    meta = pd.read_csv('data/building_meta.csv', nrows=1000)
    
    # Print the column names to ensure correct merging keys (for debugging)
    print("Energy Data Columns:", energy.columns)
    print("Meta Data Columns:", meta.columns)
    
    # Merge energy data with building metadata on 'campus_id'
    if 'campus_id' not in energy.columns or 'campus_id' not in meta.columns:
        raise KeyError("Both 'energy' and 'meta' dataframes must contain the 'campus_id' column")

    # Perform the merge based on 'campus_id'
    df = energy.merge(meta, on="campus_id", how="left")
    
    # Calculate cost_rm if not available
    if 'cost_rm' not in df.columns:
        # Assuming a fixed rate of 0.5 RM per kWh (you can adjust this as per your data)
        rate_per_kwh = 0.5
        df["cost_rm"] = df["consumption"] * rate_per_kwh
    
    # Add derived columns (example: kWh per m²)
    df["kwh_per_m2"] = df["consumption"] / df["gross_floor_area"]
    df["is_weekend"] = df["timestamp"].dt.weekday >= 5
    df["hour"] = df["timestamp"].dt.hour  # Extract the hour from the timestamp for peak hour analysis
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # Extract the day of the week for analysis
    df["month"] = df["timestamp"].dt.month  # Extract month
    
    return df

# Display spinner while data is loading
with st.spinner('Loading data...'):
    time.sleep(2)  # Simulate loading time (can be removed if data is small)
    df = load_data()

st.success('Data loaded!')

# Set title
st.title("⚡ Campus Energy Usage Dashboard")

# Sidebar for user selection
st.sidebar.header("Filters")
selected_view = st.sidebar.radio("Select View", ("Daily", "Monthly"))

# Do not filter data by building type here for comparison across buildings
df_filtered = df.copy()  # Copy entire dataset for comparison

# Data Aggregation (Daily or Monthly View)
if selected_view == "Monthly":
    # Convert to string (to make it compatible with Plotly)
    df_filtered["month"] = df_filtered["timestamp"].dt.to_period("M").astype(str)
    df_filtered = df_filtered.groupby(["month", "category"]).agg({"consumption": "sum", "cost_rm": "sum", "kwh_per_m2": "mean"}).reset_index()
else:
    df_filtered["day"] = df_filtered["timestamp"].dt.date
    df_filtered = df_filtered.groupby(["day", "category"]).agg({"consumption": "sum", "cost_rm": "sum", "kwh_per_m2": "mean"}).reset_index()

# Display KPIs
st.header(f"Energy Consumption Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total kWh", f"{df_filtered['consumption'].sum():,.0f}")
col2.metric("Total Cost (RM)", f"{df_filtered['cost_rm'].sum():,.2f}")
col3.metric("Average kWh/m²", f"{df_filtered['kwh_per_m2'].mean():.2f}")

# **Line Chart** for Energy Consumption (Showing comparison across building categories)
fig_line = px.line(df_filtered, x="month" if selected_view == "Monthly" else "day", 
                   y="consumption", color="category", 
                   title="Energy Consumption Trend Across Buildings")
st.plotly_chart(fig_line)

# **Bar Chart** for Energy Consumption Comparison Across Buildings
fig_bar = px.bar(df_filtered, x="category", y="consumption", color="category", title="Energy Consumption Comparison Across Buildings")
st.plotly_chart(fig_bar)

# **Bar Chart** for Energy Cost Comparison Across Buildings
fig_bar_cost = px.bar(df_filtered, x="category", y="cost_rm", color="category", title="Energy Cost Comparison Across Buildings")
st.plotly_chart(fig_bar_cost)

# **Prediction of Energy Consumption using Random Forest Regressor**:

# Prepare the features for prediction
features = df[['hour', 'day_of_week', 'month', 'gross_floor_area']]
target = df['consumption']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Show model performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display prediction performance
st.subheader("Prediction Model Performance")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualize predicted vs actual consumption
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
fig_pred = px.scatter(df_pred, x="Actual", y="Predicted", labels={"Actual": "Actual Consumption (kWh)", "Predicted": "Predicted Consumption (kWh)"})
st.plotly_chart(fig_pred)

# **Anomaly Detection using Isolation Forest**:

# Use Isolation Forest for anomaly detection
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['consumption', 'cost_rm']])  # Scaling features for anomaly detection

# Train Isolation Forest model
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomalies = iso_forest.fit_predict(scaled_data)

# Add anomaly detection result to the dataframe
df['anomaly'] = anomalies

# Filter anomalies
df_anomalies = df[df['anomaly'] == -1]

# Display anomalies
st.subheader("Anomalies Detected")
st.write(f"Number of anomalies detected: {df_anomalies.shape[0]}")
st.dataframe(df_anomalies)

# **Visualize Anomalies**
fig_anomalies = px.scatter(df, x='timestamp', y='consumption', color='anomaly', title="Anomalies in Energy Consumption")
st.plotly_chart(fig_anomalies)

# **Peak Usage and Cost Analysis by Hour**
# Group by hour and aggregate consumption to find peak hours
df_hourly = df.groupby("hour").agg({"consumption": "sum"}).reset_index()

# Plot Peak Usage Hours (Bar Chart)
fig_peak_usage = px.bar(df_hourly, x="hour", y="consumption", title="Peak Energy Usage by Hour", labels={"hour": "Hour of the Day", "consumption": "Total Consumption (kWh)"})
st.plotly_chart(fig_peak_usage)

# Group by hour and aggregate cost to find peak cost hours
df_hourly_cost = df.groupby("hour").agg({"cost_rm": "sum"}).reset_index()

# Plot Peak Cost Hours (Bar Chart)
fig_peak_cost = px.bar(df_hourly_cost, x="hour", y="cost_rm", title="Peak Energy Cost by Hour", labels={"hour": "Hour of the Day", "cost_rm": "Total Cost (RM)"})
st.plotly_chart(fig_peak_cost)

# Display insights
st.subheader("Insights")
st.write("• Peak energy consumption occurs during weekdays and certain hours.")
st.write("• Average energy usage trends can help optimize HVAC settings.")
st.write("• Identifying high energy consumption can lead to more efficient energy management.")

# Allow users to download the filtered data
st.download_button(
    label="Download Filtered Data",
    data=df_filtered.to_csv(index=False),
    file_name=f"energy_data_comparison.csv",
    mime="text/csv"
)
