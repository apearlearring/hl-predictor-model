# Hyperliquid Price Predictor Model

A machine learning model for predicting cryptocurrency prices on Hyperliquid platform using PyTorch.

## Project Structure

```
hl-predictor-model/
├── data/
│   └── sets/
│       └── BTC_metrics.csv    # Historical BTC metrics data
├── scripts/
│   └── convert_json_to_csv.py # Data conversion utility
├── core/
│   ├── train.py              # Training pipeline
│   └── forecast.py           # Prediction/inference logic
├── models/
│   └── base_model.py         # Base model architecture
├── .env.example              # Example environment variables
├── .gitignore
├── pyproject.toml            # Poetry dependency management
└── README.md
```

## Install

```bash
poetry install
```

## Data Format

The BTC metrics data is stored in CSV format with the following columns:

- `time`: Timestamp of the data point
- `coin`: Cryptocurrency symbol (BTC)
- `funding`: Funding rate
- `open_interest`: Open interest
- `premium`: Premium rate
- `day_ntl_vlm`: Daily notional volume
- `current_price`: Current BTC price
- `long_number`: Number of long positions
- `short_number`: Number of short positions
- `long_short_ratio`: Ratio of long to short positions

## Usage

To convert JSON data to CSV format:

```bash
poetry run python scripts/convert_json_to_csv.py
```