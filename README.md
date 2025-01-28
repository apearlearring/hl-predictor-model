# Hyperliquid Price Predictor Model

A comprehensive machine learning system for predicting cryptocurrency prices on the Hyperliquid platform. The system combines multiple models (LSTM, ARIMA, and Prophet) to generate robust price forecasts.

## Features

- Multiple model support:
  - LSTM (Long Short-Term Memory) for complex pattern recognition
  - ARIMA for time series analysis
  - Prophet for trend and seasonality detection
  - Weighted ensemble predictions
- Automated data preprocessing and feature scaling
- Configurable forecast horizons
- Interactive visualization of predictions
- Model performance monitoring
- Flexible model training and evaluation

## Project Structure

```
hl-predictor-model/
├── data/
│   ├── sets/
│   │   └── BTC_metrics.csv    # Historical BTC metrics data
│   └── utils/
│       └── data_preprocessing.py  # Data preprocessing utilities
├── models/
│   ├── arima/                 # ARIMA model implementation
│   │   ├── configs.py         # ARIMA configuration
│   │   ├── model.py          # ARIMA model class
│   │   └── utils.py          # ARIMA utilities
│   ├── lstm/                  # LSTM model implementation
│   │   ├── configs.py         # LSTM configuration
│   │   └── model.py          # LSTM model class
│   ├── prophet/              # Prophet model implementation
│   │   ├── configs.py         # Prophet configuration
│   │   └── model.py          # Prophet model class
│   ├── base_model.py         # Base model architecture
│   └── model_factory.py      # Model creation factory
├── utils/
│   └── common.py             # Common utilities
├── train.py                  # Training pipeline
├── forecast.py               # Prediction/inference logic
├── configs.py                # Global configurations
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hl-predictor-model.git
cd hl-predictor-model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The system uses historical cryptocurrency data with the following features:

- `time`: Timestamp (YYYY-MM-DD HH:MM:SS)
- `current_price`: Current BTC price
- `funding`: Funding rate
- `open_interest`: Open interest
- `premium`: Premium rate
- `day_ntl_vlm`: Daily notional volume
- `long_number`: Number of long positions
- `short_number`: Number of short positions

## Usage

### Training Models

Train individual or multiple models using the training script:

```bash
python train.py --data ./data/sets/BTC_metrics.csv --models [model_numbers]
```

Options:
- `--data`: Path to training data CSV file
- `--models`: Comma-separated list of model numbers or 'all'
  - 1: LSTM
  - 2: ARIMA
  - 3: Prophet

Example:
```bash
# Train ARIMA model only
python train.py --data ./data/sets/BTC_metrics.csv --models 2

# Train all models
python train.py --data ./data/sets/BTC_metrics.csv --models all
```

### Generating Forecasts

Generate price predictions using trained models:

```bash
python forecast.py --data [data_path] --models [model_list] --steps [forecast_steps] --start_time [start_datetime]
```

Options:
- `--data`: Path to historical data
- `--models`: Space-separated list of models (lstm arima prophet)
- `--steps`: Number of steps to forecast ahead
- `--start_time`: Starting point for forecast (format: "YYYY-MM-DD HH:MM:SS")

Example:
```bash
python forecast.py --data data/sets/BTC_metrics.csv --models lstm arima prophet --steps 24 --start_time "2024-03-15 14:00:00"
```

## Model Details

### LSTM Model
- Deep learning model for capturing complex patterns
- Features dropout layers for regularization
- Configurable architecture and hyperparameters
- Supports multi-step forecasting

### ARIMA Model
- Statistical time series model
- Automatic parameter optimization
- Handles trend and seasonality
- Includes confidence intervals

### Prophet Model
- Facebook's time series forecasting tool
- Handles missing data and outliers
- Captures seasonal patterns
- Provides uncertainty estimates

### Ensemble Forecasting
The system combines predictions from all models using weighted averaging:
- LSTM: 50% weight (best at capturing complex patterns)
- ARIMA: 40% weight (good at short-term trends)
- Prophet: 10% weight (captures seasonality)

## Output

The system generates:
1. Individual model predictions
2. Combined ensemble forecast
3. Visualization plots showing:
   - Historical data
   - Individual model forecasts
   - Ensemble prediction
   - Confidence intervals (where applicable)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
