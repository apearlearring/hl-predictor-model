#  Description: Configuration class for Prophet model.


# pylint: disable=too-many-instance-attributes
class ProphetConfig:
    """
    Configuration class for the Prophet model. This stores settings for model parameters,
    forecast parameters, and data preprocessing.
    """

    def __init__(self):
        # Prophet model configuration
        self.growth = "linear"  # Changed from "logistic" to "linear"
        self.changepoint_prior_scale = 0.05
        self.interval = "5min"
        self.yearly_seasonality = False
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.seasonality_mode = "multiplicative"
        self.remove_timezone = True

        # Forecast parameters
        self.periods = 365  # Default number of periods for future forecasts (trading days for stocks usually 252)

    def display(self):
        """Prints out the current configuration."""
        print("Prophet Configuration:")
        print(f"  Growth: {self.growth}")
        print(f"  Changepoint Prior Scale: {self.changepoint_prior_scale}")
        print(f"  Seasonality Mode: {self.seasonality_mode}")
        print(f"  Yearly Seasonality: {self.yearly_seasonality}")
        print(f"  Weekly Seasonality: {self.weekly_seasonality}")
        print(f"  Daily Seasonality: {self.daily_seasonality}")
        print(f"  Periods: {self.periods}")
        print(f"  Remove Timezone: {self.remove_timezone}")
