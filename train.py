import numpy as np
import pandas as pd

# pylint: disable=no-name-in-module
from configs import models

from data.loader import Loader
from data.utils.data_preprocessing import preprocess_data
from models.model_factory import ModelFactory
from utils.common import print_colored


def select_data(file_path=None):

    print("You selected to load data from a csv file.")
    if file_path is None:
        file_path = input("Enter the csv file path: ").strip()
    return Loader.load_csv(file_path)


def model_selection_input():
    print("Select the models to train:")
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

    # Select data dynamically based on user input
    data = select_data("data/sets/BTC_metrics.csv")  # example testing defaults , "4", "data/sets/eth.csv"


    data = preprocess_data(data) 
    
    
    # Convert date column to datetime if it's not already
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])

    # Initialize ModelFactory
    factory = ModelFactory()

    # Select models to train
    model_types = model_selection_input()

    # Train and save the selected models
    for model_type in model_types:
        print(f"Training {model_type} model...")
        model = factory.create_model(model_type)
        model.train(data)

    print_colored("Model training and saving complete!", "success")


if __name__ == "__main__":
    main()
