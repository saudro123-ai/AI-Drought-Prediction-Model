
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

    .research-panel {
        background: rgba(255,255,255,.95);
        border: 1px solid #e1e8e4;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(20,70,48,.06);
        min-height: 145px;
    }

    .research-panel h4 {
        margin-top: 0;
        margin-bottom: .4rem;
        color: #183127;
    }

    .research-panel p {
        color: #667085;
        margin-bottom: 0;
        font-size: .92rem;
    }

    .importance-note {
        background: #edf6f1;
        border: 1px solid #d3e7dc;
        border-radius: 12px;
        padding: .9rem 1rem;
        color: #355e4b;
        margin-bottom: 1rem;
    }


    .project-hero {
        background:
            linear-gradient(
                135deg,
                rgba(20, 92, 65, .98),
                rgba(41, 128, 92, .94)
            );
        border-radius: 20px;
        padding: 1.55rem 1.7rem;
        margin: .5rem 0 1.4rem 0;
        color: white;
        box-shadow: 0 12px 30px rgba(20, 70, 48, .16);
    }

    .project-hero h2 {
        color: white;
        margin: 0 0 .45rem 0;
        font-size: 1.65rem;
    }

    .project-hero p {
        color: rgba(255,255,255,.91);
        margin: .2rem 0;
        line-height: 1.55;
    }

    .hero-tags {
        margin-top: .9rem;
        display: flex;
        flex-wrap: wrap;
        gap: .45rem;
    }

    .hero-tag {
        background: rgba(255,255,255,.16);
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 999px;
        padding: .32rem .72rem;
        font-size: .82rem;
        color: white;
    }

    .sidebar-information {
        background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 13px;
        padding: .8rem .9rem;
        margin-top: .65rem;
        line-height: 1.55;
    }

    .sidebar-information strong {
        color: inherit;
    }

    .final-footer {
        text-align: center;
        padding: 1.6rem 1rem .8rem 1rem;
        margin-top: 1rem;
        border-top: 1px solid #dfe7e2;
        color: #667085;
        font-size: .88rem;
        line-height: 1.65;
    }

    .final-footer strong {
        color: #254f3d;
    }

    </style>
    """,
    unsafe_allow_html=True
)

BASE_DIR = Path(__file__).resolve().parent

RF_FILE = BASE_DIR / "random_forest_dashboard_data.csv"
LSTM_FILE = BASE_DIR / "lstm_dashboard_data.csv"

OVERALL_IMPORTANCE_IMAGE = (
    BASE_DIR / "Figure_Overall_Variable_Importance.png"
)

RF_IMPORTANCE_IMAGE = (
    BASE_DIR / "Figure_RF_Feature_Importance.png"
)

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


st.sidebar.divider()

st.sidebar.markdown("### Project Information")

st.sidebar.markdown(
    """
    <div class="sidebar-information">
        <strong>Study area</strong><br>
        Riyadh and Taif<br><br>

        <strong>Data source</strong><br>
        NASA POWER meteorological records<br><br>

        <strong>Prediction target</strong><br>
        Three-month Standardized Precipitation Index
        (SPI-3)<br><br>

        <strong>Models</strong><br>
        LSTM regression<br>
        Random Forest classification<br><br>

        <strong>Prediction horizon</strong><br>
        One month ahead<br><br>

        <strong>Version</strong><br>
        Research Prototype v1.0
    </div>
    """,
    unsafe_allow_html=True
)


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


st.markdown(
    """
    <div class="project-hero">
        <h2>Saudi Arabia AI Drought Early Warning System</h2>

        <p>
            An interactive research prototype for predicting
            meteorological drought severity and drought probability
            using historical weather records.
        </p>

        <p>
            The system combines an LSTM regression model for
            forecasting SPI-3 with a Random Forest classifier for
            estimating drought risk.
        </p>

        <div class="hero-tags">
            <span class="hero-tag">📍 Riyadh and Taif</span>
            <span class="hero-tag">🌧️ NASA POWER Data</span>
            <span class="hero-tag">🧠 LSTM</span>
            <span class="hero-tag">🌲 Random Forest</span>
            <span class="hero-tag">📊 SPI-3</span>
            <span class="hero-tag">Research Prototype v1.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

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
# RESEARCH INTERPRETATION
# ============================================================

st.divider()
st.subheader("Research Interpretation")

interpretation_1, interpretation_2, interpretation_3 = (
    st.columns(3, gap="large")
)

with interpretation_1:
    st.markdown(
        """
        <div class="research-panel">
            <h4>🌧️ Dominant weather driver</h4>
            <p>
                Precipitation is the strongest overall variable in
                the Random Forest model. This agrees with the use
                of accumulated precipitation in calculating SPI-3.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with interpretation_2:
    st.markdown(
        """
        <div class="research-panel">
            <h4>🧠 Complementary AI models</h4>
            <p>
                The LSTM predicts continuous SPI-3 severity, while
                the Random Forest separately estimates drought
                probability and drought or no-drought classification.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with interpretation_3:
    st.markdown(
        """
        <div class="research-panel">
            <h4>📍 Current geographical scope</h4>
            <p>
                The current system evaluates Riyadh and Taif.
                Additional Saudi regions can be added after preparing
                compatible meteorological datasets.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()
st.subheader("Model Performance")

lstm_performance_tab, rf_performance_tab = st.tabs(
    [
        "LSTM Regression",
        "Random Forest Classification"
    ]
)

with lstm_performance_tab:

    lstm_mae_col, lstm_rmse_col, lstm_r2_col = (
        st.columns(3)
    )

    lstm_mae_col.metric(
        "MAE",
        "0.4674",
        help=(
            "The average absolute difference between actual "
            "and predicted SPI-3."
        )
    )

    lstm_rmse_col.metric(
        "RMSE",
        "0.6157",
        help=(
            "A prediction-error measure that gives more "
            "weight to larger errors."
        )
    )

    lstm_r2_col.metric(
        "R²",
        "0.5520",
        help=(
            "The proportion of SPI-3 variation explained "
            "by the model."
        )
    )

    st.info(
        "The final LSTM uses six months of weather history, "
        "48 LSTM units, batch size 16, dropout 0.20, and a "
        "learning rate of 0.001."
    )

with rf_performance_tab:

    rf_accuracy_col, rf_precision_col, rf_recall_col, rf_f1_col = (
        st.columns(4)
    )

    rf_accuracy_col.metric(
        "Accuracy",
        "92.36%"
    )

    rf_precision_col.metric(
        "Precision",
        "50.00%"
    )

    rf_recall_col.metric(
        "Recall",
        "75.00%"
    )

    rf_f1_col.metric(
        "F1 Score",
        "0.60"
    )

    st.warning(
        "Accuracy should not be interpreted alone because "
        "drought months are less common than non-drought months. "
        "Recall and F1 provide important additional context."
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()
st.subheader(
    "What Weather Variables Influence the Model?"
)

st.markdown(
    """
    <div class="importance-note">
        <b>Main result:</b> precipitation is the most influential
        overall weather variable. The detailed analysis also shows
        that precipitation from the previous one and two months
        contributes most strongly to Random Forest classification.
    </div>
    """,
    unsafe_allow_html=True
)

overall_tab, detailed_tab = st.tabs(
    [
        "Overall Variable Importance",
        "Detailed Monthly-Lag Importance"
    ]
)

with overall_tab:

    if OVERALL_IMPORTANCE_IMAGE.exists():

        st.image(
            str(OVERALL_IMPORTANCE_IMAGE),
            caption=(
                "Combined importance of each meteorological "
                "variable across the six-month input period."
            ),
            use_container_width=True
        )

        st.markdown(
            """
            **Interpretation:** precipitation has the largest
            influence, followed by surface pressure and relative
            humidity. Temperature and wind speed contribute less,
            but they remain part of the model input.
            """
        )

    else:

        st.info(
            "Place Figure_Overall_Variable_Importance.png "
            "in the same folder as app.py."
        )

with detailed_tab:

    st.caption(
        "This chart includes all 30 model inputs: five weather "
        "variables measured across six previous months."
    )

    if RF_IMPORTANCE_IMAGE.exists():

        st.image(
            str(RF_IMPORTANCE_IMAGE),
            caption=(
                "Random Forest importance for each weather "
                "variable and each month in the input sequence."
            ),
            use_container_width=True
        )

        st.markdown(
            """
            **Interpretation:** precipitation one and two months
            before the predicted month had the strongest individual
            effects. Recent pressure and humidity values were the
            next most influential inputs.
            """
        )

    else:

        st.info(
            "Place Figure_RF_Feature_Importance.png "
            "in the same folder as app.py."
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



# ============================================================
# LIMITATIONS AND FUTURE DEVELOPMENT
# ============================================================

st.divider()
st.subheader("Limitations and Future Development")

limitation_tab, future_tab = st.tabs(
    ["Current Limitations", "Future Development"]
)

with limitation_tab:
    st.markdown(
        """
        - The current study is limited to **Riyadh and Taif**.
        - Predictions are based on historical meteorological data
          rather than live weather-station measurements.
        - Drought events are less common than non-drought events,
          creating class imbalance in classification.
        - SPI-3 represents meteorological drought and does not
          directly measure groundwater, soil moisture, vegetation
          health, or crop losses.
        - The dashboard is a research prototype and should not be
          treated as an official operational warning service.
        """
    )

with future_tab:
    st.markdown(
        """
        - Expand coverage to additional Saudi regions.
        - Connect the dashboard to regularly updated weather data.
        - Incorporate soil moisture, vegetation, evaporation, and
          satellite-derived indicators.
        - Compare additional forecasting algorithms and ensemble
          methods.
        - Introduce multi-month drought forecasts.
        - Collaborate with agricultural and water-management
          institutions to validate the warning system.
        """
    )




st.markdown(
    """
    <div class="final-footer">
        <strong>
            Saudi Arabia AI Drought Early Warning System
        </strong>
        <br>

        Developed by <strong>Saud Saad Almaymoni</strong>
        <br>

        Mawhiba Research Enrichment Program 2026
        <br><br>

        Built using Python, Streamlit, TensorFlow,
        Scikit-learn and NASA POWER meteorological data.
        <br>

        Research Prototype v1.0
    </div>
    """,
    unsafe_allow_html=True
)
