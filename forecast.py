from configs import models
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import os
from pathlib import Path
from data.loader import Loader
from utils.common import print_colored
from models.model_factory import ModelFactory


def select_data(file_path=None):
    if file_path is None:
        file_path = input("Enter the csv file path: ").strip()
    return Loader.load_csv(file_path)

def model_selection_input():
    print("Select the models to forecast:")
    print("1. All models")
    print("2. Custom selection")

    model_selection = input("Enter your choice (1/2): ").strip()

    if model_selection == "1":
        model_types = models
    elif model_selection == "2":
        available_models = {str(i + 1): model for i, model in enumerate(models)}
        print("Available models to train:")
        for key, value in available_models.items():
            print(f"{key}. {value}")
            
        selected_models = input(
            "Enter the numbers of the models to train (e.g., 1,3,5): "
        ).strip()
        model_types = [
            available_models[num.strip()]
            for num in selected_models.split(",")
            if num.strip() in available_models
        ]
    else:
        print_colored("Invalid choice, defaulting to all models.", "error")
        model_types = models

    return model_types

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
                    label=f'{model_name} Forecast', color='red', linestyle='--')
        else:
            forecast_data['time'] = pd.to_datetime(forecast_data['time'])
            plt.plot(forecast_data['time'], forecast_data['prediction'], 
                    label=f'{model_name} Forecast', color='red', linestyle='--')
        
        # Customize the plot
        plt.title(f'BTC Price Forecast using {model_name}', fontsize=16, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
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

def main():
    try:
        # Ensure forecast directory exists
        ensure_forecast_dir()
        
        factory = ModelFactory()
        data = select_data("data/sets/BTC_metrics.csv")
        model_types = model_selection_input()
        steps = int(input("Enter the number of the steps for forecasting: "))

        required_features = ['time', 'funding', 'open_interest', 'premium', 'day_ntl_vlm', 
                           'current_price', 'long_number', 'short_number']
        last_known_data = data[required_features].tail(100)
        
        print(last_known_data)
        
        # Store all forecasts for comparison
        all_forecasts = {}
        
        for model_type in model_types:
            try:
                print_colored(f"Forecasting {model_type} model...", "info")
                model = factory.create_model(model_type)
                model.load()
                
                if model_type == 'lstm':    
                    forecasted_data = model.forecast(steps, last_known_data=last_known_data)
                else:
                    forecasted_data = model.forecast(steps)
                
                print(forecasted_data)
                
                # Store forecast for comparison
                all_forecasts[model_type] = forecasted_data
                
                # Plot individual forecast
                plot_forecast(last_known_data, forecasted_data, model_type)
                
            except Exception as e:
                print_colored(f"Error processing {model_type} model: {str(e)}", "error")
        
        # Plot comparison of all models if more than one model was used
        if len(all_forecasts) > 1:
            plot_all_forecasts(last_known_data, all_forecasts)
            
    except Exception as e:
        print_colored(f"An error occurred: {str(e)}", "error")

if __name__ == "__main__":
    main()
