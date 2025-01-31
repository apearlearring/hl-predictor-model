import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional, Tuple

from models.arima.configs import ArimaConfig
from models.arima.utils import (
    grid_search_arima,
)
from models.base_model import Model


class ArimaModel(Model):
    """ARIMA model for time series forecasting"""
    
    MIN_DATA_POINTS = 30
    FREQ = '5T'  # 5-minute frequency
    
    def __init__(self, model_name: str = "arima", config: Optional[ArimaConfig] = None, debug: bool = False):
        super().__init__(model_name=model_name, debug=debug)
        self.model = None
        self.config = config or ArimaConfig()
        self.best_params = self.config.best_params
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data requirements"""
        if 'time' not in data.columns:
            raise ValueError("Time column not found in data")
        if 'current_price' not in data.columns:
            raise ValueError("Current price column not found in data")
        if len(data) < self.MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data points. Minimum required: {self.MIN_DATA_POINTS}")
    
    def _prepare_time_series(self, data: pd.DataFrame) -> pd.Series:
        """
        Prepare time series data for ARIMA model
        
        Args:
            data: Input data with 'time' and 'current_price' columns
            
        Returns:
            Cleaned and preprocessed time series
        """
        self._validate_data(data)
        
        prices = data["current_price"].copy()
        prices.index = pd.to_datetime(data["time"])
        
        # Process time series
        prices = (prices
                 .sort_index()
                 .asfreq(self.FREQ)
                 .replace([np.inf, -np.inf], np.nan)
                 .interpolate(method='linear')
                 .dropna())
        
        return prices
    
    def _fit_model(self, data: pd.Series) -> None:
        """
        Fit ARIMA model to the data
        
        Args:
            data: Preprocessed time series data
        """
        if self.config.use_grid_search:
            print("Performing grid search for ARIMA parameters...")
            self._grid_search(data)
        else:
            print(f"Using default ARIMA parameters: {self.best_params}")
        
        self.model = ARIMA(data, order=self.best_params)
        self.model = self.model.fit(method_kwargs={"maxiter": self.config.max_iter})
    
    def _grid_search(self, data: pd.Series) -> None:
        """
        Perform grid search to find optimal ARIMA parameters
        
        Args:
            data: Time series data for parameter search
        """
        grid_search_arima(
            self,
            data=data,
            p_values=self.config.p_values,
            d_values=self.config.d_values,
            q_values=self.config.q_values,
        )
    
    def _generate_forecast(self, model_fit: ARIMA, steps: int) -> Tuple[pd.Series, pd.DataFrame]:
        """Generate forecast and confidence intervals"""
        forecast_result = model_fit.get_forecast(steps=steps)
        
        # Get predictions and confidence intervals
        predictions = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=self.config.alpha)
        
        return predictions, conf_int
    
    def forecast(self, steps: int, last_known_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate forecasts from the last known data point"""
        try:
            if len(last_known_data) <= self.MIN_DATA_POINTS:
                print(f"Warning: Insufficient data (minimum {self.MIN_DATA_POINTS} points required)")
                return None
            
            # Prepare data
            prices = self._prepare_time_series(last_known_data)
            
            # Generate forecasts
            model = ARIMA(prices, order=self.config.best_params)
            model_fit = model.fit(method_kwargs={"warn_convergence": False})
            predictions, conf_int = self._generate_forecast(model_fit, steps)
            
            # Create forecast DataFrame
            forecast = pd.DataFrame({
                'time': predictions.index,
                'forecast': predictions.values,
                'forecast_lower': conf_int.iloc[:, 0].values,
                'forecast_upper': conf_int.iloc[:, 1].values
            })
            
            return forecast.set_index('time').sort_index()
            
        except Exception as e:
            print(f"Error in ARIMA forecast: {str(e)}")
            raise

    