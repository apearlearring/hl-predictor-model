from configs import models
import pandas as pd
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

def main():
    
    factory = ModelFactory()
    data = select_data("data/sets/BTC_metrics.csv")
    model_types = model_selection_input()
    steps = int(input("Enter the number of the steps for forecasting"))


    required_features = ['time', 'funding', 'open_interest', 'premium', 'day_ntl_vlm', 'current_price', 'long_number', 'short_number']
    last_known_data = data[required_features].tail(100)
    
    print(last_known_data)
    
    
    for model_type in model_types:
        print(f"Forecasting {model_type} model...")
        model = factory.create_model(model_type)
        
        if model_type == 'lstm':    
            # Convert Series to DataFrame with a single row
            forcasted_data = model.forecast(steps, last_known_data=last_known_data)
        else:
            forcasted_data = model.forecast(steps)
        
        print(forcasted_data)

if __name__ == "__main__":
    main()
