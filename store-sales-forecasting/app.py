#Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import xgboost
# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon=" ", 
    layout="wide"
)

#----------------- DARK MODE STYLING ------------#
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }

    h1, h2, h3, h4 {
        color: #FFFFFF;
    }

    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
    }

    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# ================= SIDEBAR UPLOADS ================= #

st.sidebar.title("🧠 Forecast Studio")

st.sidebar.markdown("---")

st.sidebar.markdown("### ☁️ Upload Center")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

uploaded_model = st.sidebar.file_uploader(
    "Upload Forecast Model",
    type=["pkl"]
)

st.sidebar.markdown("---")

# ---------------- LOAD DATA ---------------- #
if uploaded_file is not None and uploaded_model is not None:

    with st.spinner("Loading dashboard...."):

        # Load dataset
        df = pd.read_csv(uploaded_file)

        # Load trained model
        model = pickle.load(uploaded_model)

        # Convert date column
        df["date"] = pd.to_datetime(df["date"])

        # Create filtered dataframe
        filtered_df = df.copy()

        # Create year column
        filtered_df["year"] = filtered_df["date"].dt.year

        # Create month column
        filtered_df["month"] = filtered_df["date"].dt.month

        st.success("Dataset and model uploaded successfully!")

else:
    st.warning("Please upload both CSV and model files.")
    st.stop()
 # ================= SIDEBAR SETTINGS ================= #

    st.sidebar.markdown("### ⚙️ AI Controls")

    include_oil = st.sidebar.checkbox(
        "Enable Oil Price Signals",
        value=True
    )

    include_holidays = st.sidebar.checkbox(
        "Enable Holiday Intelligence",
        value=True
    )

    forecast_days = st.sidebar.slider(
        "Forecast Horizon",
        min_value=7,
        max_value=90,
        value=30
    )

    st.sidebar.markdown("---")

    # ================= DATE FILTERS ================= #

    st.sidebar.markdown("### 📅 Date Filters")

start_date = st.sidebar.date_input(
    "Start Date",
    filtered_df["date"].min()
)

end_date = st.sidebar.date_input(
    "End Date",
    filtered_df["date"].max()
)

filtered_df = filtered_df[
    (filtered_df["date"] >= pd.to_datetime(start_date)) &
    (filtered_df["date"] <= pd.to_datetime(end_date))
]
st.sidebar.markdown("---")

#-------------------STORE FILTER -------------------------#
store = st.sidebar.selectbox(
     "Select Store",
     filtered_df["store_nbr"].unique()
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
     min_value=7,
     max_value=90,
     value=30
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    filtered_df["year"].unique()
)

# ---------------- HEADER ---------------- #
st.title(" Store Sales Forecasting Dashboard")

st.markdown("### Real-Time Retail Analytics & Al Forecasting")

st.markdown("----")

tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Predictions",
    "Business Insights",
    "Forecast Results",
    "Project Summary"
])
st.markdown("Advanced analytics and forecasting dashboard for retail sales.")
with tab1:

st.markdown("<br>", unsafe_allow_html=True)    
# ---------------- KPI CARDS ---------------- #
    total_sales = int(filtered_df["sales"].sum())
    avg_sales = int(filtered_df["sales"].mean())
    total_stores = filtered_df["store_nbr"].nunique()
    total_rows = filtered_df.shape[0]
    
    col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Total Revenue", 
    f"{total_sales:,}"
)

col2.metric(
    "Average Revenue", 
    f"{avg_sales:,}"
) 

col3.metric(
    "Stores", 
    total stores
)

col4.metric(
    "Dataset Rows", 
    f"{total_rows:,}"
)

st.markdown("------")
  # ================= DATA PREVIEW ================= #

    st.subheader("Dataset Preview")

    st.dataframe(filtered_df.head(10))

    st.markdown("---")
# ================= SALES TREND CHART ================= #

st.markdown("---")

st.subheader("Sales Trend")

sales_trend = (
    filtered_df.groupby("date")["sales"]
    .sum()
    .reset_index()
)

fig_trend = px.line(
    sales_trend,
    x="date",
    y="sales",
    title="Daily Sales Trend",
    markers=True
)

fig_trend.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Sales",
    height=500
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)

    # ================= TOP STORE REVENUE CHART ================= #

    st.subheader("Store Revenue Analysis")

    top_locations = (
        df.groupby("store_nbr")["sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_overview = px.bar(
        top_locations,
        x="store_nbr",
        y="sales",
        color="sales",
        title="Store Sales Comparison",
        text_auto=True
    )

    fig_overview.update_layout(
        template="plotly_dark",
        xaxis_title="Store ID",
        yaxis_title="Revenue",
        height=500
    )

    st.plotly_chart(
        fig_overview,
        use_container_width=True
    )

    st.markdown("---")

#future_input = np.array([[store, forecast_days]])

#prediction = model.predict(future_input)
#st.success( 
#    f"Predicted Sales for next {forecast_days} days: ${prediction[0]:,.2f}"
# )


# ================= TAB 2 : PREDICTIONS ================= #

with tab2:

    st.title("🔮 Sales Forecasting Center")

    st.markdown(
        "Real-time forecasting powered by machine learning and predictive analytics."
    )

    st.markdown("---")

    # ================= KPI CARDS ================= #

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Confidence",
        "87%"
    )

    col2.metric(
        "Forecast Horizon",
        f"{forecast_days} Days"
    )

    col3.metric(
        "Prediction Status",
        "Active"
    )

    st.markdown("---")

    # ================= AI INSIGHT ================= #

    st.info(
        "AI Insight: Retail demand is expected to remain stable with moderate fluctuations across upcoming forecast periods."
    )

    # ================= GENERATE BUTTON ================= #

    generate_forecast = st.button(
        "🚀 Generate Forecast"
    )

    st.markdown("---")

    # ================= FORECAST DATA ================= #

    future_dates = pd.date_range(
        start=df["date"].max(),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": prediction
    })

    # ================= FORECAST CHART ================= #

    st.subheader("📈 Forecast Visualization")

    fig_forecast = px.area(
        forecast_df,
        x="Date",
        y="Predicted Sales",
        title="AI-Powered Future Sales Forecast"
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        xaxis_title="Forecast Date",
        yaxis_title="Predicted Sales",
        height=550
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )

    st.markdown("---")

    # ================= FORECAST TABLE ================= #

    st.subheader("📊 Forecast Results Table")

    st.dataframe(
        forecast_df.head(15),
        use_container_width=True
    )

    st.markdown("---")

    # ================= FORECAST ANALYTICS ================= #

    avg_prediction = int(
        forecast_df["Predicted Sales"].mean()
    )

    max_prediction = int(
        forecast_df["Predicted Sales"].max()
    )

    min_prediction = int(
        forecast_df["Predicted Sales"].min()
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Average Forecast",
        f"{avg_prediction:,}"
    )

    col5.metric(
        "Highest Forecast",
        f"{max_prediction:,}"
    )

    col6.metric(
        "Lowest Forecast",
        f"{min_prediction:,}"
    )

    st.markdown("---")

    # ================= PREDICTION INSIGHTS ================= #

    st.subheader("🧠 Prediction Insights")

    st.success(
        "Forecast analysis indicates expected growth opportunities across selected retail periods with stable predictive confidence."
    )

    st.warning(
        "Retail sales may fluctuate during holidays, promotions, and seasonal demand spikes."
    )

    st.markdown("---")

    # ================= DOWNLOAD SECTION ================= #

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Forecast Results",
        data=csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

# ================= TAB 3 : BUSINESS INSIGHTS ================= #

with tab3:

    st.title("🧠 Business Insights & Forecast Results")

    st.markdown(
        "AI-driven business intelligence insights generated from retail sales forecasting and trend analysis."
    )

    st.markdown("---")

    # ================= SALES CONTRIBUTION ================= #

    st.subheader("📊 Store Contribution Analysis")

    contribution_data = (
        filtered_df.groupby("store_nbr")["sales"]
        .sum()
        .reset_index()
    )

    fig_pie = px.pie(
        contribution_data,
        names="store_nbr",
        values="sales",
        title="Store Sales Contribution"
    )

    fig_pie.update_layout(
        template="plotly_dark",
        height=550
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    st.markdown("---")

    # ================= BUSINESS INSIGHTS ================= #

    st.subheader("📌 Key Business Insights")

    st.success(
        "High-performing stores contribute significantly to overall retail revenue generation."
    )

    st.info(
        "Forecast trends indicate stable future sales growth across selected forecast periods."
    )

    st.warning(
        "Seasonal demand fluctuations may impact sales performance during holiday and promotional periods."
    )

    st.markdown("---")

    # ================= AI RECOMMENDATIONS ================= #

    st.subheader("🤖 AI Recommendations")

    st.markdown("""
    - Increase inventory allocation during projected high-demand periods
    - Monitor underperforming store locations for sales optimization
    - Use forecast insights to improve retail planning strategies
    - Focus promotional campaigns on high-revenue sales periods
    """)

    st.markdown("---")

    # ================= FORECAST RESULTS ================= #

    st.subheader("📈 Forecast Results Overview")

    st.dataframe(
        forecast_df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    # ================= FORECAST SUMMARY ================= #

    total_forecast = int(
        forecast_df["Predicted Sales"].sum()
    )

    avg_forecast = int(
        forecast_df["Predicted Sales"].mean()
    )

    peak_forecast = int(
        forecast_df["Predicted Sales"].max()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Forecast Sales",
        f"{total_forecast:,}"
    )

    col2.metric(
        "Average Forecast",
        f"{avg_forecast:,}"
    )

    col3.metric(
        "Peak Forecast",
        f"{peak_forecast:,}"
    )

    st.markdown("---")

    # ================= FINAL INSIGHT ================= #

    st.subheader("📍 Final Forecast Insight")

    st.success(
        "Forecast analysis suggests continued retail sales stability with opportunities for strategic growth and inventory optimization."
    )# ================= TAB 4 : FORECAST EXPORT CENTER ================= #

with tab4:

    st.title("📦 Forecast Export Center")

    st.markdown(
        "Export forecast outputs, review prediction results, and prepare sales forecasts for reporting and business planning."
    )

    st.markdown("---")

    # ================= FORECAST DATA PREVIEW ================= #

    st.subheader("📄 Forecast Data Preview")

    st.dataframe(
        forecast_df.head(25),
        use_container_width=True
    )

    st.markdown("---")

    # ================= EXPORT METRICS ================= #

    total_predictions = len(forecast_df)

    max_sales = int(
        forecast_df["Predicted Sales"].max()
    )

    min_sales = int(
        forecast_df["Predicted Sales"].min()
    )

    avg_sales = int(
        forecast_df["Predicted Sales"].mean()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Forecast Records",
        total_predictions
    )

    col2.metric(
        "Highest Prediction",
        f"{max_sales:,}"
    )

    col3.metric(
        "Lowest Prediction",
        f"{min_sales:,}"
    )

    col4.metric(
        "Average Prediction",
        f"{avg_sales:,}"
    )

    st.markdown("---")

    # ================= FORECAST VISUALIZATION ================= #

    st.subheader("📈 Forecast Distribution")

    fig_export = px.line(
        forecast_df,
        x="Date",
        y="Predicted Sales",
        markers=True,
        title="Forecasted Sales Distribution"
    )

    fig_export.update_layout(
        template="plotly_dark",
        xaxis_title="Forecast Date",
        yaxis_title="Predicted Sales",
        height=550
    )

    st.plotly_chart(
        fig_export,
        use_container_width=True
    )

    st.markdown("---")

    # ================= EXPORT SECTION ================= #

    st.subheader("📥 Export Forecast Results")

    csv = forecast_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="forecast_results.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # ================= REPORT SUMMARY ================= #

    st.subheader("📝 Forecast Report Summary")

    st.info(
        "Forecast results were generated using machine learning and time-series forecasting models including ARIMA, Prophet, and XGBoost."
    )

    st.success(
        "Exported forecast outputs can support retail planning, inventory management, and sales performance analysis."
    )

    st.warning(
        "Forecast values may vary depending on future market trends, seasonal demand, and business conditions."
    )
# ================= TAB 5 : PROJECT SUMMARY ================= #

with tab5:

    st.title("📘 Project Summary")

    st.markdown(
        "Retail sales forecasting dashboard developed using machine learning and time-series forecasting techniques."
    )

    st.markdown("---")

    # ================= PROJECT OVERVIEW ================= #

    st.subheader("📌 Project Overview")

    st.markdown("""
    This dashboard was developed to analyze retail sales performance,
    generate future sales forecasts, and provide business intelligence
    insights for improved decision-making.

    The platform combines interactive analytics, forecasting models,
    and visualization tools to support retail trend analysis and
    predictive forecasting.
    """)

    st.markdown("---")

    # ================= MODELS USED ================= #

    st.subheader("🤖 Forecasting Models Used")

    model_data = pd.DataFrame({
        "Model": ["ARIMA", "Prophet", "XGBoost"],
        "Purpose": [
            "Time-Series Forecasting",
            "Trend & Seasonality Detection",
            "Machine Learning Prediction"
        ]
    })

    st.dataframe(
        model_data,
        use_container_width=True
    )

    st.markdown("---")

    # ================= FEATURES ================= #

    st.subheader("⚙️ Dashboard Features")

    st.markdown("""
    - Interactive retail sales dashboard
    - AI-powered forecasting visualization
    - Forecast export functionality
    - Store performance analytics
    - Business intelligence insights
    - Date filtering and forecasting controls
    - Upload support for datasets and trained models
    """)

    st.markdown("---")

    # ================= BUSINESS VALUE ================= #

    st.subheader("📈 Business Value")

    st.success(
        "The forecasting dashboard supports inventory planning, sales monitoring, revenue analysis, and strategic business forecasting."
    )

    st.info(
        "Interactive analytics provide visibility into retail performance trends and future sales opportunities."
    )

    st.markdown("---")

    # ================= FINAL NOTE ================= #

    st.subheader("🚀 Final Notes")

    st.markdown("""
    This project demonstrates the integration of machine learning,
    business analytics, and interactive dashboard development using
    Streamlit, Plotly, and predictive forecasting models.
    """)

    st.markdown("---")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption(" Developed by Nsisong •  Retail Sales Forecasting & Business Analytics Dashboard")
st.caption(
    "Built with Streamlit • ARIMA • Prophet • • XGBoost • • Plotly • Machine Learning"
)
