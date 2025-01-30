#  Description: Configuration class for LSTM model.


# pylint: disable=too-many-instance-attributes
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

class LstmConfig:
    """
    Configuration class for the LSTM model. This stores settings for model parameters,
    training, and data preprocessing.
    """

    def __init__(self):
        # Model architecture parameters
        self.features = 7  # Input size (number of features)
        self.n_steps_out = 24  # Number of time steps used for LSTM output
        self.hidden_size = 64  # Number of LSTM units per layer
        self.hidden_size2 = 32  # Number of LSTM units per layer
        self.output_size = 1  # Output size (prediction dimension)
        self.num_layers = 2  # Number of stacked LSTM layers
        self.dropout = 0.5  # Dropout probability for regularization
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = MeanSquaredError()

        # Training parameters
        self.learning_rate = 0.0001  # Learning rate for the optimizer
        self.batch_size = 32  # Batch size for training
        self.epochs = 1  # Number of training epochs
        self.early_stopping_patience = 10  # Early stopping patience in epochs

        # Data processing
        self.validation_split = 0.2  # Proportion of data used for validation
        self.time_steps = 60  # Number of time steps used for LSTM input
        self.interval = '5T'  # Default to daily interval (e.g., 'D', '5M', 'H', 'W', 'M')
        self.window = 24
        self.simulations = 10
        self.conf_int = 0.8

    def display(self):
        """Prints out the current configuration."""
        print("LSTM Configuration:")
        print(f"  Input Size: {self.features}")
        print(f"  Hidden Size: {self.hidden_size}")
        print(f"  Output Size: {self.output_size}")
        print(f"  Num Layers: {self.num_layers}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Early Stopping Patience: {self.early_stopping_patience}")
        print(f"  Validation Split: {self.validation_split}")
        print(f"  Time Steps: {self.time_steps}")
        print(f"  Interval: {self.interval}")
