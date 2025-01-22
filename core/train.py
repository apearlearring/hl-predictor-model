import numpy as np
import pandas as pd

# pylint: disable=no-name-in-module
from data.csv_loader import CSVLoader
from models.model_factory import ModelFactory
from utils.common import print_colored
from configs import models


def select_data(file_path=None):

    print("You selected to load data from a CSV file.")
    if file_path is None:
        file_path = input("Enter the CSV file path: ").strip()
    return CSVLoader.load_csv(file_path)


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


def prepare_lstm_data(data, time_steps):
    """Prepare data for LSTM model with correct dimensions."""
    # Select required features
    features = ['open', 'high', 'low', 'close', 'volume', 'marketCap']

    # Ensure all required columns exist
    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Missing required feature: {feature}")

    # Normalize each feature independently
    normalized_data = data[features].copy()
    for feature in features:
        mean = normalized_data[feature].mean()
        std = normalized_data[feature].std()
        normalized_data[feature] = (normalized_data[feature] - mean) / std

    # Create sequences
    X = []
    for i in range(len(normalized_data) - time_steps):
        X.append(normalized_data.iloc[i:(i + time_steps)].values)

    return np.array(X)  # Shape will be [samples, time_steps, features]


def main():

    # Select data dynamically based on user input
    data = select_data("data/sets/eth.csv")  # example testing defaults , "4", "data/sets/eth.csv"

    # Convert date column to datetime if it's not already
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])

    # Initialize ModelFactory
    factory = ModelFactory()

    # Select models to train
    model_types = model_selection_input()

    # Train and save the selected models
    for model_type in model_types:
        print(f"Training {model_type} model...")
        model = factory.create_model(model_type)

        if model_type == 'lstm':
            # Special handling for LSTM data preparation
            X = prepare_lstm_data(data, model.config.time_steps)
            model.train(X)
        else:
            model.train(data)

    print_colored("Model training and saving complete!", "success")


if __name__ == "__main__":
    main()
