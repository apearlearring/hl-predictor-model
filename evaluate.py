import pandas as pd
import argparse
from typing import Dict, List
import numpy as np

from configs import models
from utils.common import print_colored
from models.model_factory import ModelFactory
from utils.tf_settings import setup_tensorflow

setup_tensorflow()

time_steps = 60

class ModelEvaluator:
    """Handles evaluation of cryptocurrency prediction models"""
    
    def __init__(self):
        self.factory = ModelFactory()
        self.weights = {'lstm': 0.5, 'arima': 0.4, 'prophet': 0.1}
    
    def _evaluate_single_point(self, models: dict, historical_slice: pd.DataFrame, 
                             actual_data: dict, selected_models: List[str]) -> Dict:
        """Evaluate models for a single point"""
        point_predictions = {}
        point_metrics = {model: {} for model in selected_models + ['combined']}
        
        # Get predictions from each model
        for model_name in selected_models:
            try:
                forecast_df = models[model_name].forecast(
                    steps=actual_data['window'], 
                    last_known_data=historical_slice
                )
                
                if forecast_df is not None:
                    predicted_price = forecast_df['forecast'].iloc[0]
                    point_predictions[model_name] = predicted_price
                    
                    # Calculate metrics
                    direction = np.sign(predicted_price - actual_data['last_price'])
                    metrics = self._calculate_prediction_metrics(
                        predicted_price, direction,
                        actual_data, model_name
                    )
                    point_metrics[model_name] = metrics
                    
            except Exception as e:
                print_colored(f"Error with {model_name}: {str(e)}", "error")
        
        # Calculate combined prediction if all models succeeded
        if len(point_predictions) == len(selected_models):
            combined_price = sum(point_predictions[model] * self.weights[model] 
                               for model in point_predictions)
            combined_direction = np.sign(combined_price - actual_data['last_price'])
            
            point_metrics['combined'] = self._calculate_prediction_metrics(
                combined_price, combined_direction,
                actual_data, 'COMBINED'
            )
            
        return point_metrics
    
    def _calculate_prediction_metrics(self, pred_price: float, pred_direction: float, 
                                   actual_data: dict, model_name: str) -> Dict:
        """Calculate and display metrics for a single prediction"""
        mae = abs(pred_price - actual_data['value'])
        direction_str = "increase" if pred_direction > 0 else "decrease"
        correct_direction = pred_direction == actual_data['direction']
        
        # Print prediction result
        result_color = "blue" if correct_direction else "red"
        mark = "✓" if correct_direction else "✗"
        print_colored(
            f"{model_name}: {mark} Predicted: {pred_price:.2f} ({direction_str}), "
            f"Error: {mae:.2f}", 
            result_color
        )
        
        return {
            'predicted_price': pred_price,
            'mae': mae,
            'correct_direction': correct_direction,
            'direction_str': direction_str
        }
    
    def evaluate_models(self, data: pd.DataFrame, selected_models: List[str], 
                       window: int = 24) -> Dict:
        """Evaluate selected models and combined forecast"""
        
        # Initialize metrics tracking
        evaluation_metrics = {
            model: {'total_predictions': 0, 'correct_predictions': 0,
                   'total_mae': 0, 'predictions': []} 
            for model in selected_models + ['combined']
        }
        
        # Create evaluation points and models
        test_indices = np.random.choice(
            range(window + time_steps, len(data) - window), 
            size=10, replace=False
        )
        test_indices.sort()
        
        models = {name: self.factory.create_model(name) for name in selected_models}
        
        # Evaluate each point
        for i, start_idx in enumerate(test_indices):
            print_colored(f"\nEvaluation Point {i+1}/10:", "info")
            
            # Prepare data for this evaluation point
            actual_data = {
                'value': data.iloc[start_idx + 1]['current_price'],
                'date': data.iloc[start_idx + 1]['time'],
                'last_price': data.iloc[start_idx]['current_price'],
                'last_date': data.iloc[start_idx]['time'],
                'direction': np.sign(data.iloc[start_idx + 1]['current_price'] - 
                                   data.iloc[start_idx]['current_price']),
                'window': window
            }
            actual_data['direction_str'] = "increase" if actual_data['direction'] > 0 else "decrease"
            
            print(f'Reference: {actual_data["last_date"]} - Price: {actual_data["last_price"]:.2f}')
            print(f'Target:    {actual_data["date"]} - Price: {actual_data["value"]:.2f} '
                  f'({actual_data["direction_str"]})')
            
            # Evaluate models for this point
            point_metrics = self._evaluate_single_point(
                models, data.iloc[:start_idx].copy(),
                actual_data, selected_models
            )
            
            # Update overall metrics
            for model_name, metrics in point_metrics.items():
                if metrics:  # Skip if model failed
                    self._update_overall_metrics(
                        evaluation_metrics[model_name],
                        metrics, actual_data
                    )
        
        self._print_evaluation_summary(evaluation_metrics)
        return evaluation_metrics
    
    def _update_overall_metrics(self, overall_metrics: Dict, 
                              point_metrics: Dict, actual_data: Dict):
        """Update overall metrics with single point results"""
        overall_metrics['total_predictions'] += 1
        overall_metrics['total_mae'] += point_metrics['mae']
        if point_metrics['correct_direction']:
            overall_metrics['correct_predictions'] += 1
            
        overall_metrics['predictions'].append({
            'date': actual_data['date'],
            'reference_price': actual_data['last_price'],
            'predicted_price': point_metrics['predicted_price'],
            'actual_price': actual_data['value'],
            'mae': point_metrics['mae'],
            'correct_direction': point_metrics['correct_direction'],
            'predicted_direction': point_metrics['direction_str'],
            'actual_direction': actual_data['direction_str']
        })
    
    def _print_evaluation_summary(self, metrics: Dict):
        """Print comprehensive evaluation summary for all models"""
        print_colored("\n" + "="*50, "info")
        print_colored("EVALUATION SUMMARY", "info")
        print("="*50)
        
        # Calculate and display summary for each model
        for model_name, model_metrics in metrics.items():
            if model_metrics['total_predictions'] > 0:
                total_pred = model_metrics['total_predictions']
                correct_pred = model_metrics['correct_predictions']
                accuracy = (correct_pred / total_pred) * 100
                avg_mae = model_metrics['total_mae'] / total_pred
                
                print_colored(f"\n{model_name.upper()} PERFORMANCE", "info")
                print("-"*30)
                print(f"Total Predictions: {total_pred}")
                print(f"Correct Directions: {correct_pred}/{total_pred}")
                print(f"Direction Accuracy: {accuracy:.2f}%")
                print(f"Average MAE: {avg_mae:.2f}")
                
                # Print prediction history
                print("\nDetailed Prediction History:")
                print("-"*80)
                print(f"{'Date':<20} {'Direction':<15} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
                print("-"*80)
                
                for pred in model_metrics['predictions']:
                    mark = "✓" if pred['correct_direction'] else "✗"
                    print(f"{str(pred['date']):<20} "
                          f"{mark} {pred['predicted_direction']:<12} "
                          f"{pred['predicted_price']:<12.2f} "
                          f"{pred['actual_price']:<12.2f} "
                          f"{pred['mae']:<10.2f}")
                
                print("-"*80)
                print()

def get_historical_data(data: pd.DataFrame, lookback: int = 1024) -> pd.DataFrame:
    """Get historical data for evaluation"""
    required_features = ['time', 'current_price', 'funding', 'open_interest', 
                        'premium', 'day_ntl_vlm', 'long_number', 'short_number']
    data['time'] = pd.to_datetime(data['time'])
    return data[required_features].copy()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Cryptocurrency Prediction Models')
    parser.add_argument('--data', type=str, default="data/sets/BTC_metrics.csv",
                      help='Path to the CSV data file')
    parser.add_argument('--models', type=str, nargs='+', choices=models,
                      default=models, help='Models to evaluate')
    parser.add_argument('--window', type=int, default=24,
                      help='Window size for step-back testing')
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        data = pd.read_csv(args.data)
        
        historical_data = get_historical_data(data)
        print_colored(f"Using historical data from {historical_data['time'].min()} "
                     f"to {historical_data['time'].max()}", "info")
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_models(
            data=historical_data,
            selected_models=args.models,
            window=args.window
        )
            
    except Exception as e:
        print_colored(f"An error occurred: {str(e)}", "error")

if __name__ == "__main__":
    main()