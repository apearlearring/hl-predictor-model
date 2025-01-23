import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from models.arima.configs import ArimaConfig
from models.arima.utils import (
    adf_test,
    differencing,
    grid_search_arima,
    resample_data,
    reverse_differencing,
)
from models.base_model import Model


class ArimaModel(Model):
    """ARIMA model for time series forecasting"""

    def __init__(self, model_name="arima", config=ArimaConfig(), debug=False):
        super().__init__(model_name=model_name, debug=debug)
        self.model = None
        self.config = config  # Configuration instance passed to the model
        self.best_params = self.config.best_params
        
    def train(self, data: pd.DataFrame):
        """Train ARIMA model on the 'close' prices"""
        current_prices = data["current_price"]

        # Set proper datetime index or reset the index if dates are available
        if "time" in data.columns:
            current_prices = pd.Series(
                current_prices.values, index=pd.to_datetime(data["time"])
            )  # Set the index with a datetime index

            # Resample the data to the configured interval
            current_prices = resample_data(self, current_prices)

        # Perform stationarity check and differencing if necessary
        print("current_prices")
        print(current_prices)
        p_value = adf_test(current_prices)
        if p_value > 0.05:
            print("Data is not stationary, applying differencing...")
            current_prices = differencing(current_prices)

        # Perform enhanced grid search if enabled in the configuration
        if self.config.use_grid_search:
            print("Performing grid search for ARIMA parameters...")
            grid_search_arima(
                self,
                data=current_prices,
                p_values=self.config.p_values,
                d_values=self.config.d_values,
                q_values=self.config.q_values,
            )
        else:
            print("Using default ARIMA parameters...")

        # Fit ARIMA model with the configured or best-found parameters
        print("best_params")
        print(self.best_params)
        self.model = ARIMA(current_prices, order=self.best_params)
        self.model = self.model.fit(method_kwargs={"maxiter": self.config.max_iter})

        # Save the model
        self.save()

    def inference(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Predict based on existing model and input data."""
        if self.model is None:
            raise ValueError(
                "Model is not trained. Please train the model before calling forecast."
            )

        current_prices = input_data["current_price"]

        # Resample the input data to the desired interval
        current_prices = pd.Series(
            current_prices.values, index=pd.to_datetime(input_data["time"])
        )
        current_prices = resample_data(self, current_prices)

        # Forecast the number of steps equal to the length of the input data
        predictions = self.model.forecast(steps=len(current_prices))

        # If differencing was applied, reverse the differencing to get price predictions
        if self.config.best_params[1] > 0:  # If d > 0, reverse differencing
            predictions = reverse_differencing(current_prices, predictions)

        # Replace unreasonable negative values with NaN
        predictions[predictions < 0] = np.nan

        # Convert the date to string and return results
        predictions = pd.Series(
            predictions.values, index=current_prices.index.astype(str)
        )
        return pd.DataFrame(
            {"time": predictions.index, "prediction": predictions.values.ravel()}
        )

    def forecast(self, steps: int) -> pd.DataFrame:
        """Forecast future values"""
        # pylint: disable=no-member
        predictions = self.model.forecast(steps=steps)
        return predictions
