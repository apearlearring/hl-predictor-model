import os
import pandas as pd
from prophet import Prophet

from models.base_model import Model
from models.prophet.configs import ProphetConfig

class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class ProphetModel(Model):
    """Prophet model for time series forecasting"""

    def __init__(self, model_name="prophet", config=ProphetConfig(), debug=True):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config  # Use the configuration class
        self.cap = None

    def forecast(self, steps: int, last_known_data: pd.DataFrame) -> pd.DataFrame:
        
        try: 
            if len(last_known_data) <= 30:
                return None
            
            # Create a copy of the required columns
            price_df = pd.DataFrame({
                'ds': pd.to_datetime(last_known_data['time']),
                'y': last_known_data['current_price'].values
            })
            
            price_df = price_df.sort_values(by='ds')
            
            with suppress_stdout_stderr():
                model = Prophet(
                    growth=self.config.growth,
                    changepoint_prior_scale=self.config.changepoint_prior_scale,
                    yearly_seasonality=self.config.yearly_seasonality,  # type: ignore
                    weekly_seasonality=self.config.weekly_seasonality,  # type: ignore
                    daily_seasonality=self.config.daily_seasonality,  # type: ignore
                    seasonality_mode=self.config.seasonality_mode,
                )
                
                model.fit(price_df)
                
            data_forecast = model.make_future_dataframe(
                periods=steps,
                include_history=False,
                freq='5min'
            )
            
            forecast = model.predict(data_forecast)
            
            # Select and rename columns
            forecast_df = pd.DataFrame({
                'time': forecast['ds'],
                'forecast': forecast['yhat'],
                'forecast_lower': forecast['yhat_lower'],
                'forecast_upper': forecast['yhat_upper']
            })
            
            # Set index and sort
            forecast_df = forecast_df.set_index('time')
            forecast_df = forecast_df.sort_index()
            
            return forecast_df
    
        except Exception as e:
            print(f"Error in Prophet forecast: {str(e)}")
            raise
