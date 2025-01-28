import pandas as pd
import argparse
from typing import List, Optional
from pathlib import Path

from configs import models
from data.utils.data_preprocessing import preprocess_data
from models.model_factory import ModelFactory
from utils.common import print_colored


class ModelTrainer:
    """Handles the training of cryptocurrency prediction models"""
    
    def __init__(self):
        self.factory = ModelFactory()
    
    @staticmethod
    def load_data(file_path: Path) -> pd.DataFrame:
        """
        Load and preprocess data from CSV file
        
        Args:
            file_path: Path to the CSV data file
            
        Returns:
            Preprocessed DataFrame
        """
        print_colored(f"Loading data from: {file_path}", "info")
        data = pd.read_csv(file_path)
        data = preprocess_data(data)
        
        # Convert time column to datetime
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
            
        return data
    
    @staticmethod
    def parse_model_selection(model_arg: str) -> List[str]:
        """
        Parse model selection argument
        
        Args:
            model_arg: Comma-separated list of model numbers or 'all'
            
        Returns:
            List of selected model names
        """
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
    
    def train_models(self, data: pd.DataFrame, selected_models: List[str]) -> None:
        """
        Train and save selected models
        
        Args:
            data: Training data
            selected_models: List of model types to train
        """
        for model_type in selected_models:
            try:
                print_colored(f"Training {model_type} model...", "info")
                model = self.factory.create_model(model_type)
                model.train(data)
            except Exception as e:
                print_colored(f"Error training {model_type} model: {str(e)}", "error")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train cryptocurrency prediction models')
    parser.add_argument(
        '--data', 
        type=Path, 
        required=True,
        help='Path to the CSV file containing training data'
    )
    parser.add_argument(
        '--models', 
        type=str, 
        default='all',
        help='Comma-separated list of model numbers to train, or "all" for all models'
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    try:
        args = parse_args()
        trainer = ModelTrainer()
        
        # Load and preprocess data
        data = trainer.load_data(args.data)
        
        # Get selected models
        selected_models = trainer.parse_model_selection(args.models)
        
        # Train models
        trainer.train_models(data, selected_models)
        
        print_colored("Model training and saving complete!", "success")
        
    except Exception as e:
        print_colored(f"Training failed: {str(e)}", "error")
        raise


if __name__ == "__main__":
    main()
