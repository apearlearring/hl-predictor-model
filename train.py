import numpy as np
import pandas as pd
import argparse
from typing import List

# pylint: disable=no-name-in-module
from configs import models

from data.loader import Loader
from data.utils.data_preprocessing import preprocess_data
from models.model_factory import ModelFactory
from utils.common import print_colored


def parse_args():
    parser = argparse.ArgumentParser(description='Train cryptocurrency prediction models')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the CSV file containing training data')
    parser.add_argument('--models', type=str, default='all',
                       help='Comma-separated list of model numbers to train, or "all" for all models')
    return parser.parse_args()

def select_data(file_path: str):
    print_colored(f"Loading data from: {file_path}", "info")
    return Loader.load_csv(file_path)

def get_selected_models(model_arg: str) -> List[str]:
    available_models = {str(i + 1): model for i, model in enumerate(models)}
    
    if model_arg.lower() == 'all':
        return list(models)
    
    selected_models = []
    for num in model_arg.split(','):
        num = num.strip()
        if num in available_models:
            selected_models.append(available_models[num])
        else:
            print_colored(f"Warning: Invalid model number {num}, skipping", "warning")
    
    if not selected_models:
        print_colored("No valid models selected, defaulting to all models.", "warning")
        return list(models)
    
    return selected_models

def main():
    args = parse_args()
    
    # Load data from specified file path
    data = select_data(args.data)
    data = preprocess_data(data)
    
    # Convert date column to datetime if it's not already
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'])

    # Initialize ModelFactory
    factory = ModelFactory()

    # Get selected models from command line arguments
    model_types = get_selected_models(args.models)

    # Train and save the selected models
    for model_type in model_types:
        print_colored(f"Training {model_type} model...", "info")
        model = factory.create_model(model_type)
        model.train(data)

    print_colored("Model training and saving complete!", "success")

if __name__ == "__main__":
    main()
