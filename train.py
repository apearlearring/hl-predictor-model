import pandas as pd
import argparse
from pathlib import Path

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
    
    def train_models(self, data: pd.DataFrame) -> None:
        """
        Train and save selected models
        
        Args:
            data: Training data
        """
        try:
            print_colored(f"Training LSTM model...", "info")
            model = self.factory.create_model('lstm')
            model.train(data)
        except Exception as e:
            print_colored(f"Error training lstm model: {str(e)}", "error")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train cryptocurrency prediction models')
    parser.add_argument(
        '--data', 
        type=Path, 
        required=True,
        help='Path to the CSV file containing training data'
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    try:
        args = parse_args()
        trainer = ModelTrainer()
        
        # Load and preprocess data
        data = trainer.load_data(args.data)
        
        # Train models
        trainer.train_models(data)
        
        print_colored("Model training and saving complete!", "success")
        
    except Exception as e:
        print_colored(f"Training failed: {str(e)}", "error")
        raise


if __name__ == "__main__":
    main()
