import pandas as pd
import argparse
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import numpy as np
import warnings
import os

# Suppress warnings and configure TensorFlow
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Enable deterministic operations

# Import TensorFlow and configure
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Other imports
import matplotlib.pyplot as plt
from configs import models
from utils.common import print_colored
from models.model_factory import ModelFactory

class ModelForecaster:
    """Handles forecasting using trained cryptocurrency prediction models"""
    
    def __init__(self):
        # Initialize paths and components
        self.forecast_dir = Path('forecasts')
        self.forecast_dir.mkdir(exist_ok=True)
        self.factory = ModelFactory()
        # Define model weights for ensemble
        self.weights = {
            'lstm': 0.5,    # Best at capturing complex patterns
            'arima': 0.4,   # Good at short-term trends
            'prophet': 0.1  # Captures seasonality
        }
        
    def generate_forecasts(self, data: pd.DataFrame, selected_models: List[str], 
                         steps: int, start_time: Optional[str] = None) -> Dict:
        """
        Generate forecasts using selected models
        
        Args:
            data: Historical price data
            selected_models: List of models to use
            steps: Number of steps to forecast
            start_time: Optional starting time for forecast
            
        Returns:
            Dictionary containing forecasts from each model
        """
        forecasts = {}
        
        # Generate individual model forecasts
        for model_name in selected_models:
            try:
                print_colored(f"Forecasting using {model_name} model...", "info")
                model = self.factory.create_model(model_name)
                
                # Use data up to forecast point
                forecast_data = data.iloc[:-steps].copy() if steps else data.copy()
                forecast = model.forecast(steps=steps, last_known_data=forecast_data)
                
                if forecast is not None:
                    forecasts[model_name] = forecast
                    
            except Exception as e:
                print_colored(f"Error with {model_name}: {str(e)}", "error")
                
        return forecasts
    
    def combine_forecasts(self, forecasts: Dict) -> Optional[pd.DataFrame]:
        """
        Combine forecasts using weighted ensemble
        
        Args:
            forecasts: Dictionary of individual model forecasts
            
        Returns:
            Combined forecast DataFrame with confidence intervals
        """
        # Check if we have all required models
        if not all(model in forecasts for model in self.weights):
            return None

        # Combine predictions
        combined_df = pd.DataFrame()
        for model_name in self.weights:
            combined_df[model_name] = forecasts[model_name]["forecast"]
        
        # Calculate weighted average and confidence intervals
        combined_forecasts = np.average(combined_df, weights=list(self.weights.values()), axis=1)
        std_dev = np.std(combined_df.values, axis=1)
        conf_interval = 1.96 * std_dev  # 95% confidence interval
        
        return pd.DataFrame({
            'forecast': combined_forecasts,
            'lower_bound': combined_forecasts - conf_interval,
            'upper_bound': combined_forecasts + conf_interval
        }, index=combined_df.index)
    
    def plot_forecasts(self, historical_data: pd.DataFrame, forecasts: Dict):
        """
        Plot historical data and forecasts
        
        Args:
            historical_data: Historical price data
            forecasts: Dictionary of forecasts to plot
        """
        try:
            plt.figure(figsize=(15, 7))
            
            # Plot recent historical data
            hist_data = historical_data.copy()
            hist_data['time'] = pd.to_datetime(hist_data['time'])
            hist_data = hist_data.tail(min(30, len(hist_data)))  # Show last 30 points
            plt.plot(hist_data['time'].dt.strftime('%Y-%m-%d %H:%M:%S'), 
                    hist_data['current_price'], 
                    label='Historical', color='blue', alpha=0.7)
            
            # Plot individual model forecasts
            colors = ['red', 'green', 'purple', 'orange', 'brown']
            for (model_name, forecast_data), color in zip(forecasts.items(), colors):
                plt.plot(forecast_data.index.strftime('%Y-%m-%d %H:%M:%S'), 
                        forecast_data['forecast'], 
                        label=f'{model_name}', color=color, linestyle='--', 
                        marker='o', markersize=6, markerfacecolor='white')
            
            # Customize plot appearance
            plt.title('Cryptocurrency Price Forecasts', fontsize=16, pad=20)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Price (USD)', fontsize=12)
            plt.legend(fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path('forecasts') / f'forecast_comparison_{timestamp}.png'
            plt.savefig(filename)
            print_colored(f"Saved comparison plot to {filename}", "success")
            plt.show()
            
        except Exception as e:
            print_colored(f"Error plotting comparison: {str(e)}", "error")
        finally:
            plt.close()

def get_historical_data(data: pd.DataFrame, start_time: Optional[str] = None, 
                       steps: int = 1, lookback: int = 1024) -> pd.DataFrame:
    """
    Get historical data for forecasting
    
    Args:
        data: Raw data DataFrame
        start_time: Optional starting time
        steps: Number of forecast steps
        lookback: Number of historical points to use
        
    Returns:
        Processed historical DataFrame
    """
    required_features = ['time', 'current_price', 'funding', 'open_interest', 
                        'premium', 'day_ntl_vlm', 'long_number', 'short_number']
    
    df = data[required_features].copy()
    
    if start_time:
        df['time'] = pd.to_datetime(df['time'])
        df = df.loc[df['time'] <= pd.to_datetime(start_time) + pd.Timedelta(minutes=5 * steps)]
    
    return df.tail(lookback)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate Cryptocurrency Price Forecasts')
    parser.add_argument('--data', type=str, default="data/sets/BTC_metrics.csv",
                      help='Path to the CSV data file')
    parser.add_argument('--models', type=str, nargs='+', choices=models,
                      default=models, help='Models to use for forecasting')
    parser.add_argument('--steps', type=int, default=1,
                      help='Number of steps to forecast ahead')
    parser.add_argument('--start_time', type=str,
                      help='Starting time for forecast (YYYY-MM-DD HH:MM:SS)')
    return parser.parse_args()

def main():
    """Main execution function"""
    try:
        # Parse arguments and load data
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
        
        # Generate forecasts
        forecaster = ModelForecaster()
        forecasts = forecaster.generate_forecasts(
            data=historical_data,
            selected_models=args.models,
            steps=args.steps,
            start_time=args.start_time
        )
        
        print(historical_data)
        
        # Add combined forecast and plot results
        if forecasts:
            combined = forecaster.combine_forecasts(forecasts)
            if combined is not None:
                forecasts['combined'] = combined
            forecaster.plot_forecasts(historical_data, forecasts)
        
        # Print actual prices and evaluate directions
        print(f'Reference price: {historical_data.iloc[-2]["time"]} - {historical_data.iloc[-2]["current_price"]}')
        print(f'Actual price: {historical_data.iloc[-1]["time"]} - {historical_data.iloc[-1]["current_price"]}')
        
        actual_direction = np.sign(historical_data.iloc[-1]['current_price'] - 
                                 historical_data.iloc[-2]['current_price'])
        actual_direction_str = "increase" if actual_direction > 0 else "decrease"
        print(f'Actual direction: {actual_direction_str}')
        
        # Print model predictions and accuracies
        for model_name, forecast_data in forecasts.items():
            direction_prediction = np.sign(forecast_data['forecast'].iloc[0] - 
                                        historical_data.iloc[-2]['current_price'])
            direction_str = "increase" if direction_prediction > 0 else "decrease"
            
            mark = "✓" if direction_prediction == actual_direction else "✗"
            color = "blue" if direction_prediction == actual_direction else "red"
            print_colored(
                f'{model_name} : {mark} {forecast_data["forecast"].iloc[0]:.2f} ({direction_str})', 
                color
            )
            
    except Exception as e:
        print_colored(f"An error occurred: {str(e)}", "error")

if __name__ == "__main__":
    main()
