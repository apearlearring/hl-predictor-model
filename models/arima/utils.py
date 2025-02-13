import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test with proper data validation
    """
    try:
        # Ensure series is clean before testing
        series = series.replace([np.inf, -np.inf], np.nan)
        series = series.interpolate(method='linear')
        series = series.dropna()
        
        if len(series) < 20:  # Minimum length for meaningful test
            print("Warning: Series too short for reliable ADF test")
            return 1.0  # Return value indicating non-stationarity
            
        result = adfuller(series, maxlag=None)
        return result[1]  # Return p-value
        
    except Exception as e:
        print(f"Warning: ADF test failed - {str(e)}")
        return 1.0  # Return value indicating non-stationarity


def differencing(series):
    """Apply differencing if series is not stationary"""
    return series.diff().dropna()


def resample_data(self, data: pd.Series) -> pd.Series:
    """Resample the data to the desired interval"""
    return data.resample(self.config.interval).mean()


def grid_search_arima(self, data: pd.Series, p_values, d_values, q_values):
    """Find the best ARIMA(p, d, q) model using AIC."""
    best_aic = np.inf
    best_order = None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    print(p)
                    model = ARIMA(data, order=(p, d, q)).fit()
                    aic = model.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                # pylint: disable=bare-except
                except:
                    continue
    self.best_params = best_order
    print(f"Best ARIMA params: {self.best_params}")


def reverse_differencing(original_data: pd.Series, predictions: pd.Series) -> pd.Series:
    """Reverse differencing by adding the previous values to the predictions."""
    last_observed_value = original_data.iloc[-1]  # Use .iloc for position-based access
    for i in range(len(predictions)):
        predictions.iloc[i] = (
            predictions.iloc[i] + last_observed_value
        )  # Use .iloc for position-based setting
        last_observed_value = predictions.iloc[
            i
        ]  # Update last observed value using .iloc
    return predictions
