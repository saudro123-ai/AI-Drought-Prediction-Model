
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Saudi Drought Early Warning Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# ============================================================
# DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 38px;
        font-weight: 750;
        margin-bottom: 3px;
    }

    .subtitle {
        color: #667085;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .status-card {
        background-color: white;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e3e7ec;
        margin-bottom: 15px;
    }

    .normal-card {
        border-left: 8px solid #2e8b57;
    }

    .watch-card {
        border-left: 8px solid #e3a008;
    }

    .warning-card {
        border-left: 8px solid #c0392b;
    }

    .small-description {
        font-size: 13px;
        color: #667085;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

@st.cache_data
def load_dashboard_data():

    file_path = "/content/random_forest_dashboard_data.csv"

    data = pd.read_csv(file_path)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

    numeric_columns = [
        "Drought_Probability",
        "Predicted_Drought",
        "Actual_Drought",
        "SPI_3",
        "Temperature_at_2m",
        "Relative_Humidity_at_2m",
        "Surface_Pressure",
        "Wind_Speed_at_2m",
        "Monthly_Precipitation"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    return data


df = load_dashboard_data()

# ============================================================
# CITY COORDINATES
# ============================================================

city_coordinates = {

    "Riyadh": {
        "latitude": 24.7136,
        "longitude": 46.6753
    },

    "Taif": {
        "latitude": 21.2854,
        "longitude": 40.4248
    }
}

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'AI Drought Early Warning Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Random Forest drought classification using six months '
    'of meteorological data from Riyadh and Taif'
    '</div>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.title("Dashboard Controls")

available_cities = sorted(
    df["City"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not available_cities:

    st.error("No cities were found in the dashboard dataset.")
    st.stop()

selected_city = st.sidebar.selectbox(
    "Select city",
    available_cities
)

city_df = (
    df[df["City"] == selected_city]
    .sort_values("Date")
    .reset_index(drop=True)
)

available_dates = (
    city_df["Date"]
    .dropna()
    .drop_duplicates()
    .tolist()
)

if available_dates:

    selected_date = st.sidebar.selectbox(
        "Select prediction month",
        available_dates,
        index=len(available_dates) - 1,
        format_func=lambda date: date.strftime("%B %Y")
    )

    selected_row = city_df[
        city_df["Date"] == selected_date
    ].iloc[-1]

else:

    selected_date = None
    selected_row = city_df.iloc[-1]

# ============================================================
# SELECTED PREDICTION VALUES
# ============================================================

probability = float(
    selected_row.get(
        "Drought_Probability",
        0
    )
)

prediction = int(
    selected_row.get(
        "Predicted_Drought",
        0
    )
)

actual = int(
    selected_row.get(
        "Actual_Drought",
        0
    )
)

actual_spi = selected_row.get(
    "SPI_3",
    None
)

# ============================================================
# DETERMINE RISK STATUS
# ============================================================

if prediction == 1:

    risk_status = "Drought Warning"
    risk_description = (
        "The Random Forest model detected meteorological "
        "conditions associated with drought."
    )

    status_class = "warning-card"
    status_color = "#c0392b"
    marker_color = "red"

elif probability >= 40:

    risk_status = "Drought Watch"
    risk_description = (
        "The model did not classify this month as drought, "
        "but the probability is elevated."
    )

    status_class = "watch-card"
    status_color = "#d68910"
    marker_color = "orange"

else:

    risk_status = "Normal Conditions"
    risk_description = (
        "The model detected a relatively low probability "
        "of drought."
    )

    status_class = "normal-card"
    status_color = "#247a4d"
    marker_color = "green"

# ============================================================
# INTERACTIVE MAP AND PREDICTION SUMMARY
# ============================================================

map_column, summary_column = st.columns(
    [1.55, 1],
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

        city_records = (
            df[df["City"] == city]
            .sort_values("Date")
        )

        if city == selected_city:

            city_row = selected_row

        else:

            city_row = city_records.iloc[-1]

        city_probability = float(
            city_row.get(
                "Drought_Probability",
                0
            )
        )

        city_prediction = int(
            city_row.get(
                "Predicted_Drought",
                0
            )
        )

        city_spi = city_row.get(
            "SPI_3",
            None
        )

        if city_prediction == 1:

            city_status = "Drought Warning"
            city_marker_color = "red"

        elif city_probability >= 40:

            city_status = "Drought Watch"
            city_marker_color = "orange"

        else:

            city_status = "Normal Conditions"
            city_marker_color = "green"

        if pd.notna(city_spi):

            spi_text = f"{float(city_spi):.2f}"

        else:

            spi_text = "Unavailable"

        popup_text = f"""
        <div style="width:240px">

            <h3 style="margin-bottom:8px;">
                {city}
            </h3>

            <b>Status:</b>
            {city_status}
            <br>

            <b>Drought probability:</b>
            {city_probability:.1f}%
            <br>

            <b>Model prediction:</b>
            {"Drought" if city_prediction == 1 else "No drought"}
            <br>

            <b>Actual SPI-3:</b>
            {spi_text}

        </div>
        """

        folium.CircleMarker(
            location=[
                coordinates["latitude"],
                coordinates["longitude"]
            ],
            radius=15 if city == selected_city else 11,
            color=city_marker_color,
            fill=True,
            fill_color=city_marker_color,
            fill_opacity=0.82,
            weight=4 if city == selected_city else 2,
            tooltip=f"{city}: {city_status}",
            popup=folium.Popup(
                popup_text,
                max_width=300
            )
        ).add_to(drought_map)

    st_folium(
        drought_map,
        height=500,
        use_container_width=True,
        returned_objects=[]
    )

    st.caption(
        "Green: normal conditions | "
        "Orange: drought watch | "
        "Red: drought warning"
    )

with summary_column:

    st.subheader("Prediction Summary")

    date_text = (
        selected_date.strftime("%B %Y")
        if selected_date is not None
        else "Latest available month"
    )

    st.markdown(
        f"""
        <div class="status-card {status_class}">

            <h2 style="
                color:{status_color};
                margin-top:0;
                margin-bottom:8px;
            ">
                {risk_status}
            </h2>

            <b>City:</b>
            {selected_city}
            <br>

            <b>Prediction month:</b>
            {date_text}
            <br><br>

            <span class="small-description">
                {risk_description}
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )

    metric_1, metric_2 = st.columns(2)

    metric_1.metric(
        "Drought Probability",
        f"{probability:.1f}%"
    )

    metric_2.metric(
        "RF Prediction",
        "Drought" if prediction == 1 else "No Drought"
    )

    metric_3, metric_4 = st.columns(2)

    metric_3.metric(
        "Actual Condition",
        "Drought" if actual == 1 else "No Drought"
    )

    if actual_spi is not None and pd.notna(actual_spi):

        metric_4.metric(
            "Actual SPI-3",
            f"{float(actual_spi):.2f}"
        )

    else:

        metric_4.metric(
            "Actual SPI-3",
            "Unavailable"
        )

    if prediction == actual:

        st.success(
            "The Random Forest prediction matched "
            "the actual drought class."
        )

    else:

        st.warning(
            "The Random Forest prediction did not match "
            "the actual drought class for this month."
        )

    st.info(
        "Actual drought labels were defined using SPI-3 ≤ −1."
    )

# ============================================================
# WEATHER CONDITION CARDS
# ============================================================

st.divider()

st.subheader("Meteorological Conditions")

weather_variables = [

    (
        "Temperature_at_2m",
        "Temperature",
        "°C"
    ),

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

available_weather_variables = [

    item
    for item in weather_variables
    if item[0] in selected_row.index
]

if available_weather_variables:

    weather_columns = st.columns(
        len(available_weather_variables)
    )

    for display_column, (
        variable,
        label,
        unit
    ) in zip(
        weather_columns,
        available_weather_variables
    ):

        value = selected_row[variable]

        if pd.isna(value):

            displayed_value = "Unavailable"

        else:

            displayed_value = (
                f"{float(value):.2f} {unit}"
            )

        display_column.metric(
            label,
            displayed_value
        )

else:

    st.info(
        "Weather-variable columns were not included "
        "in the current Random Forest result table."
    )

# ============================================================
# HISTORICAL CHARTS
# ============================================================

st.divider()

st.subheader(
    f"Historical Drought Indicators — {selected_city}"
)

chart_column_1, chart_column_2 = st.columns(
    2,
    gap="large"
)

with chart_column_1:

    st.markdown("#### SPI-3 History")

    if "SPI_3" in city_df.columns:

        spi_history = (
            city_df[
                ["Date", "SPI_3"]
            ]
            .dropna()
            .set_index("Date")
        )

        if not spi_history.empty:

            st.line_chart(
                spi_history,
                use_container_width=True
            )

            st.caption(
                "SPI-3 values at or below −1 "
                "represent drought months."
            )

        else:

            st.info(
                "No valid SPI-3 values are available."
            )

    else:

        st.info(
            "SPI-3 was not included in the dashboard data."
        )

with chart_column_2:

    st.markdown(
        "#### Random Forest Drought Probability"
    )

    probability_history = (
        city_df[
            ["Date", "Drought_Probability"]
        ]
        .dropna()
        .set_index("Date")
    )

    if not probability_history.empty:

        st.line_chart(
            probability_history,
            use_container_width=True
        )

        st.caption(
            "Higher percentages represent stronger "
            "classifier confidence in the drought class."
        )

    else:

        st.info(
            "No valid drought probabilities are available."
        )

# ============================================================
# PRECIPITATION HISTORY
# ============================================================

if "Monthly_Precipitation" in city_df.columns:

    st.divider()

    st.subheader(
        f"Monthly Precipitation — {selected_city}"
    )

    precipitation_history = (
        city_df[
            ["Date", "Monthly_Precipitation"]
        ]
        .dropna()
        .set_index("Date")
    )

    if not precipitation_history.empty:

        st.bar_chart(
            precipitation_history,
            use_container_width=True
        )

# ============================================================
# PREDICTION HISTORY TABLE
# ============================================================

st.divider()

st.subheader("Prediction History")

preferred_history_columns = [

    "Date",
    "City",
    "SPI_3",
    "Drought_Probability",
    "Prediction_Label",
    "Actual_Label",
    "Risk_Status"
]

history_columns = [

    column
    for column in preferred_history_columns
    if column in city_df.columns
]

history_table = (
    city_df[history_columns]
    .sort_values(
        "Date",
        ascending=False
    )
    .copy()
)

if "Drought_Probability" in history_table.columns:

    history_table["Drought_Probability"] = (
        history_table["Drought_Probability"]
        .round(2)
    )

if "SPI_3" in history_table.columns:

    history_table["SPI_3"] = (
        history_table["SPI_3"]
        .round(3)
    )

st.dataframe(
    history_table,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "About the Random Forest model"
):

    st.markdown(
        """
        ### Random Forest classifier

        The classifier uses six months of weather history to
        predict whether the following month will experience
        drought conditions.

        **Input variables**

        - Temperature at 2 metres
        - Relative humidity at 2 metres
        - Surface pressure
        - Wind speed at 2 metres
        - Monthly precipitation

        **Drought definition**

        A month was labelled as drought when:

        **SPI-3 ≤ −1**

        **Final test performance**

        - Accuracy: **92.36%**
        - Precision: **50%**
        - Recall: **75%**
        - F1 score: **0.60**

        **Geographical coverage**

        The current version includes Riyadh and Taif.
        Additional Saudi cities can be added when compatible
        meteorological data becomes available.
        """
    )
