from configs import models
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
from pathlib import Path
import argparse
from data.loader import Loader
from utils.common import print_colored
from models.model_factory import ModelFactory


def ensure_forecast_dir():
    """Create forecasts directory if it doesn't exist"""
    forecast_dir = Path('forecasts')
    forecast_dir.mkdir(exist_ok=True)
    return forecast_dir

def plot_forecast(historical_data: pd.DataFrame, forecast_data: pd.DataFrame, model_name: str):
    """Plot historical and forecasted prices"""
    try:
        plt.figure(figsize=(15, 7))
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['lines.linewidth'] = 2
        
        # Plot historical data
        historical_data['time'] = pd.to_datetime(historical_data['time'])
        plt.plot(historical_data['time'], historical_data['current_price'], 
                label='Historical Price', color='blue', alpha=0.7)
        
        # Plot forecasted data
        if 'ds' in forecast_data.columns:
            forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
            plt.plot(forecast_data['ds'], forecast_data['yhat'], 
                    label=f'{model_name} Forecast', color='red')
        else:
            forecast_data['time'] = pd.to_datetime(forecast_data['time'])
            plt.plot(forecast_data['time'], forecast_data['prediction'], 
                    label=f'{model_name} Forecast', color='red', alpha=0.7)
        
        # Customize the plot
        plt.title(f'BTC Price Forecast using {model_name}', fontsize=16, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Ensure directory exists and save plot
        forecast_dir = ensure_forecast_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = forecast_dir / f'{model_name.lower()}_forecast_{timestamp}.png'
        plt.savefig(filename)
        print_colored(f"Saved forecast plot to {filename}", "success")
        
    except Exception as e:
        print_colored(f"Error plotting forecast for {model_name}: {str(e)}", "error")
        print(e)
    finally:
        plt.close()

def plot_all_forecasts(historical_data: pd.DataFrame, forecasts: dict):
    """Plot historical data and forecasts from all models in a single graph"""
    try:
        plt.figure(figsize=(15, 7))
        sns.set_style("whitegrid")
        
        # Plot historical data
        historical_data['time'] = pd.to_datetime(historical_data['time'])
        plt.plot(historical_data['time'], historical_data['current_price'], 
                label='Historical Price', color='blue', alpha=0.7)
        
        # Plot each model's forecast with different colors
        colors = ['red', 'green', 'purple', 'orange', 'brown']
        for (model_name, forecast_data), color in zip(forecasts.items(), colors):
            if 'ds' in forecast_data.columns:
                forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
                plt.plot(forecast_data['ds'], forecast_data['yhat'], 
                        label=f'{model_name} Forecast', color=color, linestyle='--')
            else:
                forecast_data['time'] = pd.to_datetime(forecast_data['time'])
                plt.plot(forecast_data['time'], forecast_data['prediction'], 
                        label=f'{model_name} Forecast', color=color, linestyle='--')
        
        plt.title('BTC Price Forecasts Comparison', fontsize=16, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.show()
        
        # Ensure directory exists and save plot
        forecast_dir = ensure_forecast_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = forecast_dir / f'all_models_comparison_{timestamp}.png'
        plt.savefig(filename)
        print_colored(f"Saved comparison plot to {filename}", "success")
        
    except Exception as e:
        print_colored(f"Error plotting comparison: {str(e)}", "error")
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
    
    parser.add_argument('--lookback', type=int, default=100,
                      help='Number of historical data points to use')
    
    return parser.parse_args()

def get_data_from_time(data: pd.DataFrame, start_time: str, lookback: int, steps: int) -> pd.DataFrame:
    """
    Get data from specified start time with lookback period
    
    Args:
        data (pd.DataFrame): Full dataset
        start_time (str): Start time for forecasting
        lookback (int): Number of historical data points to use
    
    Returns:
        pd.DataFrame: Filtered dataset
    """
    data['time'] = pd.to_datetime(data['time'])
    start_datetime = pd.to_datetime(start_time)
    
    # Get data up to start_time
    mask = data['time'] <= (start_datetime + pd.Timedelta(minutes=5 * steps))
    filtered_data = data[mask]
    
    # Get last lookback points
    return filtered_data.tail(lookback)

def main():
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Ensure forecast directory exists
        ensure_forecast_dir()
        
        # Load data
        data = Loader.load_csv(args.data)
        
        required_features = ['time', 'current_price', 'funding', 'open_interest', 'premium', 
                           'day_ntl_vlm',  'long_number', 'short_number']
        
        # Filter data based on start_time if provided
        if args.start_time:
            historical_data = get_data_from_time(
                data[required_features], 
                args.start_time, 
                args.lookback,
                args.steps
            )
        else:
            historical_data = data[required_features].tail(args.lookback)
        
        print_colored(f"Using historical data from {historical_data['time'].min()} to {historical_data['time'].max()}", "info")
        print_colored(f"Forecasting {args.steps} steps ahead", "info")
        
        # Store all forecasts for comparison
        all_forecasts = {}
        factory = ModelFactory()
        
        for model_type in args.models:
            try:
                print_colored(f"Forecasting using {model_type} model...", "info")
                model = factory.create_model(model_type)
                model.load()
                
                if model_type == 'lstm':    
                    forecasted_data = model.forecast(args.steps, last_known_data=historical_data[:-args.steps])
                else:
                    forecasted_data = model.forecast(args.steps)
                
                print(forecasted_data)
                all_forecasts[model_type] = forecasted_data
                
                # Plot individual forecast
                # plot_forecast(historical_data, forecasted_data, model_type)
                
            except Exception as e:
                print_colored(f"Error processing {model_type} model: {str(e)}", "error")
        
        # Plot comparison if multiple models were used
        if len(all_forecasts) > 0:
            plot_all_forecasts(historical_data, all_forecasts)
            
    except Exception as e:
        print_colored(f"An error occurred: {str(e)}", "error")

if __name__ == "__main__":
    main()
