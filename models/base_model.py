import os
from abc import ABC, abstractmethod

import joblib
import pandas as pd

from utils.common import print_colored


class Model(ABC):
    """Base class for all financial models with integrated save/load functionality."""

    SUPPORTED_MODEL_TYPES = [
        "pytorch",
        "pkl",
    ]  # Custom model types: 'pytorch' and 'pkl'

    def __init__(
        self, model_name, model_type="pkl", save_dir="trained_models", debug=True
    ):
        self.debug = debug
        self.model_name = model_name
        self.model_type = model_type  # Default to 'pkl' since most are not PyTorch
        self.save_dir = save_dir
        self.scaler = None  # Placeholder for models that use a scaler
        self.model = None  # Initialize with None

        # Validate the model type
        if self.model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. Supported types: {self.SUPPORTED_MODEL_TYPES}"
            )

    @abstractmethod
    def forecast(self, steps: int) -> pd.DataFrame:
        """Forecast the future steps based on the trained model."""

    def load(self):
        """Load the model and scaler (if applicable) from disk."""
        model_dir = os.path.join(self.save_dir, self.model_name)
        print(model_dir)
        try:
            # Load the model (joblib or pickle) and scaler (if applicable)
            model_path = os.path.join(model_dir, "model.pkl")
            scaler_path = os.path.join(model_dir, "scaler.pkl")

            self.model = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            if self.debug:
                print_colored(f"Model loaded from {model_path}", "success")
                if os.path.exists(scaler_path):
                    print_colored(f"Scaler loaded from {scaler_path}", "success")
        except FileNotFoundError as e:
            if self.debug:
                print_colored(f"Error: {str(e)}", "error")
        # pylint: disable=broad-except
        except Exception as e:
            if self.debug:
                print_colored(f"Failed to load model due to: {str(e)}", "error")
