
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Saudi AI Drought Dashboard",
    page_icon="🌦️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #667085;
        font-size: 17px;
        margin-bottom: 22px;
    }

    .status-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dfe3e8;
        margin-bottom: 15px;
    }

    .normal {
        border-left: 8px solid #2e8b57;
    }

    .watch {
        border-left: 8px solid #d99500;
    }

    .warning {
        border-left: 8px solid #c0392b;
    }
    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parent

RF_FILE = BASE_DIR / "random_forest_dashboard_data.csv"
LSTM_FILE = BASE_DIR / "lstm_dashboard_data.csv"

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    rf_data = pd.read_csv(RF_FILE)
    lstm_data = pd.read_csv(LSTM_FILE)

    rf_data["Date"] = pd.to_datetime(
        rf_data["Date"],
        errors="coerce"
    )

    lstm_data["Date"] = pd.to_datetime(
        lstm_data["Date"],
        errors="coerce"
    )

    return rf_data, lstm_data


rf_df, lstm_df = load_data()

city_coordinates = {
    "Riyadh": [24.7136, 46.6753],
    "Taif": [21.2854, 40.4248]
}

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Saudi Arabia AI Drought Early Warning Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'LSTM SPI-3 regression and Random Forest drought '
    'classification for Riyadh and Taif'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

rf_cities = set(
    rf_df["City"].dropna().astype(str).unique()
)

lstm_cities = set(
    lstm_df["City"].dropna().astype(str).unique()
)

available_cities = sorted(
    rf_cities.intersection(lstm_cities)
)

if not available_cities:
    st.error(
        "No matching cities were found in both model datasets."
    )
    st.stop()

selected_city = st.sidebar.selectbox(
    "Select city",
    available_cities
)

rf_city = (
    rf_df[rf_df["City"] == selected_city]
    .sort_values("Date")
    .reset_index(drop=True)
)

lstm_city = (
    lstm_df[lstm_df["City"] == selected_city]
    .sort_values("Date")
    .reset_index(drop=True)
)

rf_dates = set(rf_city["Date"].dropna())
lstm_dates = set(lstm_city["Date"].dropna())

common_dates = sorted(
    rf_dates.intersection(lstm_dates)
)

if common_dates:

    selected_date = st.sidebar.selectbox(
        "Select month",
        common_dates,
        index=len(common_dates) - 1,
        format_func=lambda value: value.strftime("%B %Y")
    )

    rf_row = rf_city[
        rf_city["Date"] == selected_date
    ].iloc[-1]

    lstm_row = lstm_city[
        lstm_city["Date"] == selected_date
    ].iloc[-1]

else:

    st.sidebar.warning(
        "The two model test periods do not share identical "
        "dates. Latest available values are being shown."
    )

    selected_date = None
    rf_row = rf_city.iloc[-1]
    lstm_row = lstm_city.iloc[-1]

# ============================================================
# VALUES
# ============================================================

rf_probability = float(
    rf_row["Drought_Probability"]
)

rf_prediction = int(
    rf_row["Predicted_Drought"]
)

actual_drought = int(
    rf_row["Actual_Drought"]
)

actual_spi = float(
    lstm_row["Actual_SPI_3"]
)

predicted_spi = float(
    lstm_row["Predicted_SPI_3"]
)

if rf_prediction == 1:
    risk_status = "Drought Warning"
    risk_class = "warning"
    marker_color = "red"

elif rf_probability >= 40:
    risk_status = "Drought Watch"
    risk_class = "watch"
    marker_color = "orange"

else:
    risk_status = "Normal Conditions"
    risk_class = "normal"
    marker_color = "green"

# ============================================================
# MAP AND MODEL SUMMARY
# ============================================================

map_column, summary_column = st.columns(
    [1.4, 1],
    gap="large"
)

with map_column:

    st.subheader("Interactive Saudi Arabia Map")

    drought_map = folium.Map(
        location=[23.5, 44.5],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    for city in available_cities:

        coordinates = city_coordinates.get(city)

        if coordinates is None:
            continue

        city_rf = (
            rf_df[rf_df["City"] == city]
            .sort_values("Date")
        )

        city_lstm = (
            lstm_df[lstm_df["City"] == city]
            .sort_values("Date")
        )

        if city == selected_city:
            map_rf_row = rf_row
            map_lstm_row = lstm_row
        else:
            map_rf_row = city_rf.iloc[-1]
            map_lstm_row = city_lstm.iloc[-1]

        map_probability = float(
            map_rf_row["Drought_Probability"]
        )

        map_prediction = int(
            map_rf_row["Predicted_Drought"]
        )

        map_spi = float(
            map_lstm_row["Predicted_SPI_3"]
        )

        if map_prediction == 1:
            map_status = "Drought Warning"
            map_color = "red"
        elif map_probability >= 40:
            map_status = "Drought Watch"
            map_color = "orange"
        else:
            map_status = "Normal Conditions"
            map_color = "green"

        popup = f"""
        <div style="width:240px">
        <h3>{city}</h3>
        <b>Status:</b> {map_status}<br>
        <b>RF drought probability:</b>
        {map_probability:.1f}%<br>
        <b>RF prediction:</b>
        {"Drought" if map_prediction == 1 else "No drought"}<br>
        <b>LSTM predicted SPI-3:</b>
        {map_spi:.2f}
        </div>
        """

        folium.CircleMarker(
            location=coordinates,
            radius=15 if city == selected_city else 11,
            color=map_color,
            fill=True,
            fill_color=map_color,
            fill_opacity=0.8,
            weight=4 if city == selected_city else 2,
            tooltip=f"{city}: {map_status}",
            popup=folium.Popup(
                popup,
                max_width=300
            )
        ).add_to(drought_map)

    st_folium(
        drought_map,
        height=480,
        use_container_width=True,
        returned_objects=[]
    )

    st.caption(
        "Green: normal | Orange: drought watch | "
        "Red: drought warning"
    )

with summary_column:

    st.subheader("Current Model Summary")

    displayed_date = (
        selected_date.strftime("%B %Y")
        if selected_date is not None
        else "Latest available result"
    )

    st.markdown(
        f"""
        <div class="status-card {risk_class}">
        <h2>{risk_status}</h2>
        <b>City:</b> {selected_city}<br>
        <b>Month:</b> {displayed_date}
        </div>
        """,
        unsafe_allow_html=True
    )

    metric_1, metric_2 = st.columns(2)

    metric_1.metric(
        "RF Drought Probability",
        f"{rf_probability:.1f}%"
    )

    metric_2.metric(
        "RF Classification",
        "Drought" if rf_prediction == 1
        else "No Drought"
    )

    metric_3, metric_4 = st.columns(2)

    metric_3.metric(
        "LSTM Predicted SPI-3",
        f"{predicted_spi:.2f}"
    )

    metric_4.metric(
        "Actual SPI-3",
        f"{actual_spi:.2f}"
    )

    metric_5, metric_6 = st.columns(2)

    metric_5.metric(
        "Actual Condition",
        "Drought" if actual_drought == 1
        else "No Drought"
    )

    metric_6.metric(
        "LSTM SPI Error",
        f"{abs(actual_spi - predicted_spi):.2f}"
    )

    st.info(
        "A month is classified as an actual drought when "
        "SPI-3 is less than or equal to −1."
    )

# ============================================================
# WEATHER CONDITIONS
# ============================================================

st.divider()
st.subheader("Meteorological Conditions")

weather_variables = [
    ("Temperature_at_2m", "Temperature", "°C"),
    (
        "Relative_Humidity_at_2m",
        "Relative Humidity",
        "%"
    ),
    (
        "Surface_Pressure",
        "Surface Pressure",
        "kPa"
    ),
    (
        "Wind_Speed_at_2m",
        "Wind Speed",
        "m/s"
    ),
    (
        "Monthly_Precipitation",
        "Monthly Precipitation",
        "mm"
    )
]

available_weather = [
    item
    for item in weather_variables
    if item[0] in rf_row.index
]

if available_weather:

    weather_columns = st.columns(
        len(available_weather)
    )

    for column, (
        variable,
        label,
        unit
    ) in zip(
        weather_columns,
        available_weather
    ):

        value = rf_row[variable]

        column.metric(
            label,
            (
                f"{float(value):.2f} {unit}"
                if pd.notna(value)
                else "Unavailable"
            )
        )

else:

    st.info(
        "Weather columns were not included in the "
        "Random Forest dashboard CSV."
    )

# ============================================================
# LSTM GRAPH
# ============================================================

st.divider()
st.subheader(
    f"LSTM Actual vs Predicted SPI-3 — {selected_city}"
)

lstm_chart_data = (
    lstm_city[
        [
            "Date",
            "Actual_SPI_3",
            "Predicted_SPI_3"
        ]
    ]
    .dropna()
    .set_index("Date")
)

st.line_chart(
    lstm_chart_data,
    use_container_width=True
)

st.caption(
    "The LSTM predicts the numerical SPI-3 drought severity. "
    "The drought threshold is SPI-3 ≤ −1."
)

# ============================================================
# RANDOM FOREST GRAPH
# ============================================================

st.subheader(
    f"Random Forest Drought Probability — {selected_city}"
)

rf_chart_data = (
    rf_city[
        ["Date", "Drought_Probability"]
    ]
    .dropna()
    .set_index("Date")
)

st.line_chart(
    rf_chart_data,
    use_container_width=True
)

# ============================================================
# TABLE
# ============================================================

st.divider()
st.subheader("Random Forest Prediction History")

history_columns = [
    column
    for column in [
        "Date",
        "City",
        "SPI_3",
        "Drought_Probability",
        "Prediction_Label",
        "Actual_Label",
        "Risk_Status"
    ]
    if column in rf_city.columns
]

history_table = (
    rf_city[history_columns]
    .sort_values("Date", ascending=False)
)

st.dataframe(
    history_table,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# MODEL EXPLANATION
# ============================================================

with st.expander("Model information"):

    st.markdown(
        """
        ### LSTM regression model

        The LSTM uses six months of meteorological history to
        predict the numerical SPI-3 value for the following
        month.

        Final reported performance:

        - MAE: **0.4674**
        - RMSE: **0.6157**
        - R²: **0.5520**

        ### Random Forest classifier

        The Random Forest uses six months of five weather
        variables, producing 30 input features.

        It predicts:

        - Drought probability
        - Drought or no-drought classification

        ### Important distinction

        - **Actual drought:** determined by actual SPI-3 ≤ −1
        - **LSTM output:** predicted SPI-3 value
        - **Random Forest output:** drought probability and
          drought classification
        - **Drought watch:** model probability of at least 40%,
          without a positive drought classification

        The dashboard displays historical test predictions.
        It does not claim to use live weather data.
        """
    )
