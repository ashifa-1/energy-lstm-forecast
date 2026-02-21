# Forecast Energy Consumption Using LSTM Networks

## Overview

This project implements a deep learning–based time series forecasting model using Long Short-Term Memory (LSTM) networks to predict future household energy consumption.

The model uses the past 72 hours of historical data to forecast the next 24 hours of energy usage. The implementation covers the complete end-to-end workflow including data preprocessing, feature engineering, sequence generation, model training, evaluation, and residual analysis.

## Dataset

**Dataset Used:**
UCI Individual Household Electric Power Consumption Dataset

**Source:** UCI Machine Learning Repository

The dataset contains minute-level measurements of household power consumption over multiple years. For this project:

- Minute-level data was resampled to hourly averages.
- Missing values were handled.
- Temporal features were engineered.
- Sequences were created using a sliding window approach.

## Methodology

The project follows a structured time series forecasting pipeline:

1. **Data Preprocessing**
   - Combined Date and Time into a single datetime index.
   - Converted all columns to numeric.
   - Removed missing values.
   - Resampled minute-level data into hourly aggregates.

2. **Exploratory Data Analysis**
   - Identified daily seasonality.
   - Analyzed monthly trends.
   - Visualized consumption patterns over time.

3. **Feature Engineering**
   Additional temporal features were created:
   - Hour of the day
   - Month
   - Day of the week

4. **Sequence Creation (Sliding Window)**
   - Past 72 hours used as input sequence.
   - Next 24 hours used as prediction target.
   - Generated supervised learning samples from time series data.

5. **Chronological Data Splitting**
   - 70% Training
   - 15% Validation
   - 15% Testing
   (Time-based split to avoid data leakage.)

6. **Scaling**
   - Applied MinMaxScaler.
   - Scaler fitted only on training data to prevent leakage.

## Model Architecture

The forecasting model is implemented using TensorFlow/Keras.

Architecture:

- LSTM Layer (64 units)
- Dropout Layer (0.2)
- Dense Output Layer (24 units for 24-hour forecast)

Loss Function: Mean Squared Error (MSE)
Optimizer: Adam

The trained model is saved inside the `model/` directory.

## Model Evaluation

The model was evaluated using standard forecasting metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

Example Results:

- MAE ≈ 2.82
- RMSE ≈ 3.89
- MAPE ≈ 58.5%

The model successfully captured overall consumption trends and seasonal patterns but struggled with sharp spikes, which is common in baseline LSTM implementations.

Residual analysis showed errors centered approximately around zero, indicating no major systematic bias.

## Project Structure

```
energy-lstm-forecast/
│
├── data/
│   └── .gitkeep
│
├── notebook/
│   └── energy_forecast.ipynb
│
├── model/
│   └── lstm_energy_model.h5
│
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ashifa-1/energy-lstm-forecast
   cd energy-lstm-forecast
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Notebook**
   ```bash
   jupyter notebook
   ```

   Open:
   `notebook/energy_forecast.ipynb`

## Key Learning Outcomes

- End-to-end time series forecasting workflow
- Temporal feature engineering
- Sliding window sequence modeling
- Chronological train-validation-test splitting
- LSTM implementation for sequence-to-sequence forecasting
- Residual analysis and performance interpretation

This project demonstrates a complete deep learning pipeline for energy consumption forecasting using LSTM networks.
