# Hyperliquid Price Predictor Model

A comprehensive machine learning system for predicting cryptocurrency prices on the Hyperliquid platform. The system combines multiple models (LSTM, ARIMA, and Prophet) to generate robust price forecasts using ensemble learning techniques.

## Key Features

### Model Architecture
- **Multi-Model Ensemble System**:
  - LSTM (Long Short-Term Memory) for complex pattern recognition and non-linear relationships
  - ARIMA for statistical time series analysis and trend detection
  - Prophet for handling seasonality and missing data
  - Weighted ensemble predictions with configurable weights

### Prediction Strategy
- **Percentage Change Prediction**:
  - Models predict price percentage changes instead of absolute prices
  - This approach provides several advantages:
    - Better handling of price volatility
    - More stable training process
    - Scale-invariant predictions
    - Improved generalization across different price ranges
  - Final price predictions are derived from:
    1. Predicted percentage changes
    2. Last known prices
    3. Confidence intervals for risk assessment

### Advanced Capabilities
- Real-time price prediction with 5-minute intervals
- Multi-step forecasting up to 24 steps ahead
- Automatic feature engineering and selection
- Dynamic model weight adjustment based on performance
- Confidence interval estimation for risk assessment
- Robust error handling and data validation

### Data Processing
- Automated data preprocessing and cleaning
- Advanced feature scaling using RobustScaler for outlier handling
- Missing data interpolation with linear methods
- Time series resampling and alignment
- Automatic stationarity testing and differencing
- Percentage change calculation with configurable windows

### Model Training
- Configurable hyperparameters via config files
- Early stopping to prevent overfitting
- Model checkpointing for best performance
- Cross-validation for robust evaluation
- Automatic parameter optimization for ARIMA
- Dropout regularization for LSTM

### Visualization & Monitoring
- Interactive forecast visualization
- Historical vs predicted price comparisons
- Confidence interval plotting
- Model performance metrics tracking
- Training progress monitoring
- Error analysis and diagnostics

## Technical Details

### Percentage Change Calculation
```python
# Example of how percentage changes are calculated
percent_change = (current_price - previous_price) / previous_price * 100
```

### Price Reconstruction
```python
# Example of how final prices are reconstructed
final_price = last_known_price * (1 + predicted_percent_change)
```

### Benefits of Percentage Change Prediction
1. **Stationarity**: Percentage changes are more likely to be stationary, improving model performance
2. **Scale Independence**: Models can learn patterns regardless of absolute price levels
3. **Error Interpretation**: Errors in percentage terms are more meaningful for trading
4. **Risk Assessment**: Easier to estimate prediction confidence intervals
5. **Model Stability**: More stable training process and better convergence

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
- **Monte Carlo Simulation**:
  - Performs multiple forward passes with dropout enabled
  - Generates probability distribution of predictions
  - Provides robust confidence intervals
  - Configurable number of simulations
  - Handles uncertainty estimation

#### LSTM Simulation Process
```python
# Example of how LSTM simulations work
predictions = []
for i in range(config.simulations):
    # Generate prediction with dropout enabled
    pred = model.predict(sequence)
    # Convert to percentage changes
    pred = scaler.inverse_transform(pred)
    # Calculate prices
    pred = prices * (pred + 1)
    predictions.append(pred)

# Calculate statistics from simulations
forecast_mean = predictions.mean(axis=0)
forecast_std = predictions.std(axis=0)

# Generate confidence intervals
z_norm = st.norm.ppf(config.conf_int)
forecast_upper = forecast_mean + z_norm * forecast_std
forecast_lower = forecast_mean - z_norm * forecast_std
```

### Benefits of Simulation Approach
1. **Robust Predictions**:
   - Multiple forward passes capture model uncertainty
   - Reduces impact of random initialization
   - Better handling of edge cases

2. **Uncertainty Quantification**:
   - Confidence intervals based on actual predictions
   - More reliable risk assessment
   - Better understanding of model confidence

3. **Improved Decision Making**:
   - Range of possible outcomes
   - Probability-based forecasts
   - Risk-aware predictions

### Configuration Options
```python
class LstmConfig:
    # Simulation parameters
    simulations = 10    # Number of Monte Carlo simulations
    conf_int = 0.8     # Confidence interval (80%)
    dropout = 0.2      # Dropout rate for uncertainty
```

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
