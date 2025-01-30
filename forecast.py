import numpy as np
from configs import models
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse
from utils.common import print_colored
from models.model_factory import ModelFactory


def ensure_forecast_dir():
    """Create forecasts directory if it doesn't exist"""
    forecast_dir = Path('forecasts')
    forecast_dir.mkdir(exist_ok=True)
    return forecast_dir


def plot_forecasts(historical_data: pd.DataFrame, forecasts: dict):
    """Plot historical data and forecasts from all models"""
    try:
        plt.figure(figsize=(15, 7))

        # Create a copy of historical data
        hist_data = historical_data.copy()
        hist_data['time'] = pd.to_datetime(hist_data['time'])
        
        # Plot historical data
        plt.plot(hist_data['time'][-100:], hist_data['current_price'][-100:], 
                label='Historical', color='blue', alpha=0.7)
        
        # Plot forecasts with different colors
        colors = ['red', 'green', 'purple', 'orange', 'brown']
        for (model_name, forecast_data), color in zip(forecasts.items(), colors):
            plt.plot(forecast_data['forecast'], label=f'{model_name}', color=color, linestyle='--')
        
        plt.title('BTC Price Forecasts Comparison', fontsize=16, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot first
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = ensure_forecast_dir() / f'forecast_comparison_{timestamp}.png'
        plt.savefig(filename)
        print_colored(f"Saved comparison plot to {filename}", "success")
        
        # Show plot after saving
        plt.show()
        
    except Exception as e:
        print_colored(f"Error plotting comparison: {str(e)}", "error")
        raise
    finally:
        plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Forecasting')
    parser.add_argument('--data', type=str, default="data/sets/BTC_metrics.csv",
                      help='Path to the CSV data file')
    parser.add_argument('--models', type=str, nargs='+', choices=models,
                      default=models, help='Models to use for forecasting')
    parser.add_argument('--steps', type=int, default=12,
                      help='Number of steps to forecast')
    parser.add_argument('--start_time', type=str,
                      help='Start time for forecasting (format: YYYY-MM-DD HH:MM:SS)')
    return parser.parse_args()


def get_historical_data(data: pd.DataFrame, start_time: str = None, lookback: int = 1024, steps: int = 24) -> pd.DataFrame:
    """Get historical data with optional start time filtering"""
    required_features = ['time', 'current_price', 'funding', 'open_interest', 
                        'premium', 'day_ntl_vlm', 'long_number', 'short_number']
    
    # Create a copy of the data to avoid the warning
    df = data[required_features].copy()
    
    if start_time:
        df['time'] = pd.to_datetime(df['time'])
        # Use .loc for proper indexing
        df = df.loc[df['time'] <= pd.to_datetime(start_time) + pd.Timedelta(minutes=5 * steps)]
    
    return df.tail(lookback)


def main():
    try:
        args = parse_args()
        data = pd.read_csv(args.data)
        
        # Get historical data
        historical_data = get_historical_data(
            data=data, 
            start_time=args.start_time, 
            steps=args.steps
        )
        
        print_colored(f"Using historical data from {historical_data['time'].min()} "
                     f"to {historical_data['time'].max()}", "info")
        print_colored(f"Forecasting {args.steps} steps ahead", "info")
        
        # Generate forecasts
        forecasts = {}
        factory = ModelFactory()
        
        for model_type in args.models:
            try:
                print_colored(f"Forecasting using {model_type} model...", "info")
                model = factory.create_model(model_type)
                
                # Create a copy of the data for forecasting
                forecast_data = historical_data.iloc[:-args.steps].copy()
                
                forecast = model.forecast(
                    steps=args.steps, 
                    last_known_data=forecast_data
                )
                
                if forecast is not None:
                    forecasts[model_type] = forecast
                    
            except Exception as e:
                print_colored(f"Error with {model_type} model: {str(e)}", "error")
        
        # Calculate weighted ensemble if all models are available
        weights = [0.4, 0.1, 0.5]
        combined_df = pd.DataFrame()
        
        for model_name in ["arima", "prophet", "lstm"]:
            if model_name in forecasts:
                combined_df[model_name] = forecasts[model_name]["forecast"]
        
        if len(combined_df.columns) == 3:
            combined_forecasts = np.average(combined_df, weights=weights, axis=1)
            forecasts['combined'] = pd.DataFrame(
                {'forecast': combined_forecasts}, 
                index=combined_df.index
            )
            
        if forecasts:
            plot_forecasts(historical_data, forecasts)
            
        if forecasts:
            plot_forecasts(historical_data, forecasts)
    except Exception as e:
        print_colored(f"An error occurred: {str(e)}", "error")


if __name__ == "__main__":
    main()
