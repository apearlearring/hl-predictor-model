# Hyperliquid Price Predictor Model

A comprehensive machine learning system for predicting cryptocurrency prices on the Hyperliquid platform. The system combines multiple models (LSTM, ARIMA, and Prophet) to generate robust price forecasts using ensemble learning techniques.

## Features

- **Multi-Model Ensemble System**
  - LSTM for complex pattern recognition
  - ARIMA for statistical time series analysis
  - Prophet for handling seasonality and missing data
  - Weighted ensemble predictions

- **Advanced Evaluation**
  - Direction accuracy tracking
  - Multiple error metrics (MAE, MSE, MAPE)
  - Confidence interval estimation
  - Step-back testing

- **Data Processing**
  - Automated preprocessing
  - Feature scaling (RobustScaler)
  - Missing data interpolation
  - Time series resampling
  - Stationarity testing

## Installation

```bash
git clone https://github.com/yourusername/hl-predictor-model.git
cd hl-predictor-model
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --data ./data/sets/BTC_metrics.csv
```

Options:
- `--data`: Path to training data for LSTM

### Forecasting

```bash
python forecast.py --data data/sets/BTC_metrics.csv --models lstm --steps 1 --start_time "2024-12-10 00:00:00"
```

Options:
- `--data`: Path to training data
- `--models`: Model names (lstm, arima, prophet)
- `--start_time`: Start time
- `--steps`: Forecast steps

Outputs:
```
Actual price: 2024-12-15 00:00:00 - 101498.0
Actual price: 2024-12-15 00:05:00 - 101575.0
Actual direction: increase
lstm : ✓ 101500.38 (increase)
prophet : ✗ 101060.75 (decrease)
arima : ✓ 101510.38 (increase)
combined : ✗ 101460.42 (decrease)
```


### Evaluation

```bash
python evaluate.py --data data/sets/BTC_metrics.csv --models lstm arima prophet --window 1
```

Options:
- `--data`: Path to historical data
- `--models`: Models to evaluate
- `--window`: Step-back window size

Outputs:
```
EVALUATION SUMMARY
==================
Reference: 2024-12-23 01:00:00 - Price: 94365.00
Target:    2024-12-23 01:05:00 - Price: 94395.00 (increase)
Model and scalers loaded from trained_models\lstm
lstm: ✗ Predicted: 94354.84 (decrease), Error: 40.16
arima: ✓ Predicted: 94397.82 (increase), Error: 2.82
prophet: ✓ Predicted: 94820.53 (increase), Error: 425.53
COMBINED: ✓ Predicted: 94418.60 (increase), Error: 23.60

PROPHET PERFORMANCE
------------------------------
Total Predictions: 10
Correct Directions: 5/10
Direction Accuracy: 50.00%
Average MAE: 526.34

Detailed Prediction History:
--------------------------------------------------------------------------------
Date                 Direction       Predicted    Actual       Error
--------------------------------------------------------------------------------
2024-12-22 20:45:00  ✗ decrease     94662.59     95248.00     585.41
2024-12-22 23:50:00  ✓ increase     95187.29     95288.00     100.71
2024-12-23 01:05:00  ✓ increase     94820.53     94395.00     425.53
2024-12-23 01:35:00  ✓ increase     94458.08     94546.00     87.92
2024-12-23 18:10:00  ✗ decrease     92802.15     93535.00     732.85
2024-12-24 14:05:00  ✓ decrease     94457.04     95922.00     1464.96
2024-12-25 02:35:00  ✗ increase     99030.10     97883.00     1147.10
2024-12-25 05:25:00  ✗ decrease     98173.34     98261.00     87.66
2024-12-25 16:30:00  ✓ increase     98435.22     98400.00     35.22
2024-12-25 21:50:00  ✗ increase     98954.04     98358.00     596.04
--------------------------------------------------------------------------------
```

## Model Configuration

### LSTM
```python
model_config = {
    'dropout': 0.2,
    'simulations': 100
}
```

### ARIMA
```python
config = {
    'order': (5, 1, 1),
    'seasonal_order': (1, 1, 1, 12)
}
```

### Prophet
```python
config = {
    'changepoint_prior_scale': 0.05,
    'seasonality_mode': 'multiplicative'
}
```

### Ensemble Weights
```python
weights = {
    'lstm': 0.5,
    'arima': 0.4,
    'prophet': 0.1
}
```

## Required Data Features

- `time`: Timestamp (YYYY-MM-DD HH:MM:SS)
- `current_price`: BTC price
- `funding`: Funding rate
- `open_interest`: Open interest
- `premium`: Premium rate
- `day_ntl_vlm`: Daily notional volume
- `long_number`: Long positions count
- `short_number`: Short positions count
