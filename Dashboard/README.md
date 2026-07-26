# Saudi Arabia AI Drought Early Warning Project

## Overview

This research project uses meteorological data and artificial
intelligence to assess drought conditions in Riyadh and Taif.

The project contains two complementary machine-learning models:

1. An LSTM regression model that predicts the numerical SPI-3
   drought severity value.
2. A Random Forest classifier that predicts drought probability
   and drought/no-drought classification.

## Data

Source:
NASA POWER meteorological data.

Cities:
- Riyadh
- Taif

Variables:
- Temperature at 2 metres
- Relative humidity at 2 metres
- Surface pressure
- Wind speed at 2 metres
- Monthly precipitation

Daily observations were aggregated into monthly observations.

## Drought definition

An actual drought month is defined as:

SPI-3 <= -1

SPI-3 represents the Standardized Precipitation Index calculated
using three-month accumulated precipitation.

## LSTM model

Purpose:
Predict the following month's numerical SPI-3 value.

Input:
Six months of meteorological history.

Final configuration:
- Sequence length: 6 months
- LSTM units: 48
- Batch size: 16
- Dropout: 0.20
- Learning rate: 0.001

Reported final performance:
- MAE: 0.4674
- RMSE: 0.6157
- R²: 0.5520

## Random Forest model

Purpose:
Predict whether the following month will experience drought.

Input:
Six months of five meteorological variables.

6 months x 5 variables = 30 flattened features.

Outputs:
- Drought probability
- Drought or no-drought classification

The corrected dashboard dataset contains both Riyadh and Taif.

## Dashboard interpretation

Actual drought:
The actual SPI-3 value is less than or equal to -1.

LSTM result:
The predicted numerical SPI-3 value.

Random Forest result:
The predicted drought probability and drought/no-drought class.

Drought warning:
The Random Forest predicts drought.

Drought watch:
The Random Forest does not predict drought, but its probability
is at least 40 percent.

Normal conditions:
The Random Forest predicts no drought and the probability is
below 40 percent.

Drought watch is not the same as an actual drought month.

## Included files

- app.py
  Combined Streamlit dashboard for both AI models.

- random_forest_dashboard_data.csv
  Corrected Random Forest test predictions for Riyadh and Taif.

- lstm_dashboard_data.csv
  Actual and LSTM-predicted SPI-3 values.

- random_forest_drought_model.joblib
  Saved Random Forest classifier, when available.

- lstm_spi3_model.keras
  Saved LSTM model, when available.

- Saudi_Drought_AI_Project.ipynb
  Full Colab research notebook.

- requirements.txt
  Required Python packages.

## Dashboard requirements

The website should:

- Include an interactive Saudi Arabia map.
- Include Riyadh and Taif.
- Display LSTM predicted SPI-3.
- Display actual SPI-3.
- Display Random Forest drought probability.
- Display Random Forest drought classification.
- Clearly distinguish predictions from actual results.
- Include actual-versus-predicted SPI-3 charts.
- Include drought-probability history.
- Include meteorological-condition cards.
- Include model-performance information.
- Be responsive and suitable for research judges.
- Never invent live forecasts or new measurements.