from abc import ABC

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
