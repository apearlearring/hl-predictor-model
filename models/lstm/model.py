import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset
import joblib
from pathlib import Path

from models.base_model import Model
from models.lstm.configs import LstmConfig


# Define the LSTM architecture
class LSTM(nn.Module):
    """LSTM based model for time series forecasting"""

    # pylint: disable=too-many-arguments
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define the LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

    def forward(self, x, hidden_state=None):
        # Initialize hidden and cell states if not provided
        if hidden_state is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden_state = (h0, c0)

        # Forward pass through LSTM
        out, hidden_state = self.lstm(x, hidden_state)

        # Apply batch normalization and dropout on the output
        out = self.batch_norm(out[:, -1, :])  # Normalize across the last time step
        out = self.dropout(out)

        # Fully connected layer to map the hidden state to output
        out = self.fc(out)
        return out, hidden_state


# Define the LSTM model class that integrates with the base model
class LstmModel(Model):
    """LSTM model for time series forecasting"""

    def __init__(self, model_name="lstm", config=LstmConfig(), debug=True):
        super().__init__(model_name=model_name, model_type="pytorch", debug=debug)
        self.config = config  # Use the configuration class
        self.model = LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            output_size=self.config.output_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.criterion = nn.MSELoss()

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Initialize scalers dictionary
        self.scalers = {}
        self.feature_names = [
            'current_price', 'funding', 'open_interest', 'premium',
            'day_ntl_vlm', 'long_number', 'short_number'
        ]

    def _initialize_scalers(self):
        """Initialize scalers for each feature"""
        for feature in self.feature_names:
            if feature in ['current_price', 'day_ntl_vlm']:
                # Use RobustScaler for price and volume data to handle outliers
                self.scalers[feature] = RobustScaler()
            else:
                # Use StandardScaler for other features
                self.scalers[feature] = StandardScaler()

    def _save_scalers(self):
        """Save scalers to disk"""
        scaler_dir = Path('trained_models') / self.model_name / 'scalers'
        scaler_dir.mkdir(parents=True, exist_ok=True)
        
        for feature, scaler in self.scalers.items():
            scaler_path = scaler_dir / f'{feature}_scaler.pkl'
            joblib.dump(scaler, scaler_path)

    def _load_scalers(self):
        """Load scalers from disk"""
        scaler_dir = Path('trained_models') / self.model_name / 'scalers'
        
        for feature in self.feature_names:
            scaler_path = scaler_dir / f'{feature}_scaler.pkl'
            if scaler_path.exists():
                self.scalers[feature] = joblib.load(scaler_path)
            else:
                raise FileNotFoundError(f"Scaler not found for feature: {feature}")

    def _scale_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Scale features using appropriate scalers
        
        Args:
            data: DataFrame containing features to scale
            
        Returns:
            Scaled features as numpy array
        """
        scaled_features = []
        
        # Initialize scalers if not already done
        if not self.scalers:
            self._initialize_scalers()
        
        for feature in self.feature_names:
            if feature not in data.columns:
                raise ValueError(f"Required feature '{feature}' not found in data")
                
            values = data[feature].values.reshape(-1, 1)
            
            # Fit the scaler if it hasn't been fitted yet
            if not hasattr(self.scalers[feature], 'mean_') or not hasattr(self.scalers[feature], 'scale_'):
                self.scalers[feature].fit(values)
            
            scaled_values = self.scalers[feature].transform(values)
            scaled_features.append(scaled_values)
        
        return np.hstack(scaled_features)

    def _create_sequences(self, scaled_data: np.ndarray, 
                         sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for training/inference"""
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length):
            sequence = scaled_data[i:(i + sequence_length)]
            target = scaled_data[i + sequence_length, 0]  # First feature is price
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)

    def train(self, data):
        """Train the LSTM model with improved feature scaling"""
        try:
            # Handle both DataFrame and numpy array inputs
            if isinstance(data, pd.DataFrame):
                if "time" in data.columns:
                    data["time"] = pd.to_datetime(data["time"])
                    data = data.set_index("time")
                data = data.resample(self.config.interval).mean().dropna()
                scaled_data = self._scale_features(data)
            else:
                # If input is numpy array, assume it's already properly formatted
                # and contains all required features in the correct order
                scaled_data = np.array(data)
                if scaled_data.ndim == 1:
                    scaled_data = scaled_data.reshape(-1, 1)
                
                # Initialize and fit scalers if not already done
                if not self.scalers:
                    self._initialize_scalers()
                    for i, feature in enumerate(self.feature_names):
                        feature_data = scaled_data[:, i].reshape(-1, 1)
                        self.scalers[feature].fit(feature_data)
                
                # Scale each feature
                scaled_features = []
                for i, feature in enumerate(self.feature_names):
                    feature_data = scaled_data[:, i].reshape(-1, 1)
                    scaled_values = self.scalers[feature].transform(feature_data)
                    scaled_features.append(scaled_values)
                scaled_data = np.hstack(scaled_features)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, self.config.time_steps)
            
            # Convert to tensors
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)
            
            # Create dataset and dataloader
            dataset = TensorDataset(X, y)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            # Training loop
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                self.model.train()
                epoch_loss = 0
                
                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    output, _ = self.model(batch_X)
                    loss = self.criterion(output.squeeze(), batch_y)

                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                
                if self.debug and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.config.epochs}], Loss: {avg_loss:.4f}")
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model
                    self.save()
                    self._save_scalers()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break
                        
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def inference(self, input_data: pd.DataFrame, time_steps=None) -> pd.DataFrame:
        self.model.eval()

        # Ensure the 'date' column is present and set it as the index for resampling
        if "time" in input_data.columns:
            input_data["time"] = pd.to_datetime(input_data["time"])
            input_data = input_data.set_index("time")  # Set 'date' as index

        # Resample based on the specified interval in the config
        input_data = input_data.resample(self.config.interval).mean().dropna()

        # Initialize the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Set the time_steps to the configuration value if not provided
        time_steps = self.config.time_steps if time_steps is None else time_steps

        # Scale the close prices using the scaler
        close_prices_scaled = scaler.fit_transform(
            input_data["close"].values.astype(float).reshape(-1, 1)
        )

        # Dynamically adjust time_steps if necessary
        time_steps = min(time_steps, len(close_prices_scaled))

        # Prepare the scaled data into time step sequences using a sliding window approach
        x_test = []
        if len(close_prices_scaled) <= time_steps:
            x_test.append(close_prices_scaled[:time_steps, 0])
        else:
            for i in range(time_steps, len(close_prices_scaled)):
                x_test.append(
                    close_prices_scaled[i - time_steps : i, 0]
                )  # Create sequences using sliding windows

        x_test = np.array(x_test)
        if self.debug:
            print(f"Prepared {len(x_test)} sequences for testing")

        # Check if any sequences were created for prediction
        if len(x_test) == 0:
            raise ValueError(
                "No sequences were generated for testing. Check if input data is sufficient."
            )

        # Reshape inputs to be 3D: [batch_size, sequence_length, input_size]
        x_test = np.expand_dims(
            x_test, axis=-1
        )  # Adding input_size dimension to make it [batch_size, sequence_length, 1]

        inputs = torch.tensor(
            x_test, dtype=torch.float32
        )  # Convert input data to torch tensor

        predictions = []
        hidden_state = (
            None  # Start with no hidden state, it will be initialized on the first pass
        )

        # Forward pass through the model for each step, using the recursive approach
        with torch.no_grad():
            # If we have only one sequence, start prediction from there
            if len(inputs) == 1:
                for i in range(
                    len(input_data)
                ):  # Predict for every day (or interval) in the input_data
                    predicted_scaled, hidden_state = self.model(
                        inputs[-1:], hidden_state
                    )  # Pass the hidden state
                    predictions.append(predicted_scaled.cpu().numpy()[0])
                    if self.debug:
                        print(
                            f"Generated prediction {i + 1}: {predicted_scaled.cpu().numpy()[0]}"
                        )

                    # Update the input for the next prediction: shift the window and append the prediction
                    new_input_sequence = np.append(
                        inputs[-1, 1:, :],
                        [[predicted_scaled.cpu().numpy()[0][0]]],
                        axis=0,
                    )  # Slide the window
                    new_input_sequence = np.array(
                        new_input_sequence
                    )  # Ensure it's a NumPy array first
                    new_input_tensor = torch.tensor(
                        new_input_sequence, dtype=torch.float32
                    ).unsqueeze(
                        0
                    )  # Convert to tensor and add batch dimension
                    inputs = torch.cat((inputs, new_input_tensor), dim=0)

            # If we have more than one sequence, use the loop
            else:
                for i in range(len(close_prices_scaled) - time_steps):
                    predicted_scaled, hidden_state = self.model(
                        inputs[-1:], hidden_state
                    )  # Pass the hidden state
                    predictions.append(predicted_scaled.cpu().numpy()[0])
                    if self.debug:
                        print(
                            f"Generated prediction {i + 1}: {predicted_scaled.cpu().numpy()[0]}"
                        )

                    new_input_sequence = np.append(
                        inputs[-1, 1:, :],
                        [[predicted_scaled.cpu().numpy()[0][0]]],
                        axis=0,
                    )
                    inputs = torch.cat(
                        (
                            inputs,
                            torch.tensor([new_input_sequence], dtype=torch.float32),
                        ),
                        dim=0,
                    )

        # Convert predictions to 2D array for inverse transform, if predictions exist
        if predictions:
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)
            if self.debug:
                print(f"Inverse transformed predictions: {predictions}")
        else:
            raise ValueError("No predictions were generated.")

        # Ensure we have predictions for each input date (handle small datasets)
        if len(predictions) < len(input_data):
            predictions = np.pad(
                predictions.flatten(), (0, len(input_data) - len(predictions)), "edge"
            )

        predictions = predictions.flatten()

        # Reset index to ensure the 'date' column is available
        input_data = input_data.reset_index()

        # Now create the DataFrame with the 'date' and predictions columns
        df_predictions = pd.DataFrame(
            {
                "time": input_data["time"][
                    : len(predictions)
                ],  # Ensure the date column matches the length of predictions
                "prediction": predictions,
            }
        ).reset_index(drop=True)

        return df_predictions

    def forecast(self, steps: int, last_known_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast future values with improved scaling"""
        try:
            self.model.eval()
            self._load_scalers()  # Load saved scalers
            
            # Prepare input data
            if "time" in last_known_data.columns:
                last_known_data = last_known_data.set_index("time")
            last_known_data = last_known_data.resample(self.config.interval).mean().dropna()
            
            # Scale features
            scaled_data = self._scale_features(last_known_data)
            
            print(scaled_data)
            
            # Prepare last sequence
            last_sequence = scaled_data[-self.config.time_steps:].reshape(
                1, self.config.time_steps, len(self.feature_names)
            )
            
            # Generate predictions
            predictions = []
            current_sequence = torch.FloatTensor(last_sequence)
            
            with torch.no_grad():
                for _ in range(steps):
                    output, _ = self.model(current_sequence)
                    predicted_scaled = output.numpy()[0]
                    
                    # Inverse transform only the price prediction
                    predicted_price = self.scalers['current_price'].inverse_transform(
                        predicted_scaled.reshape(-1, 1)
                    )[0][0]
                    
                    predictions.append(predicted_price)
                    
                    # Update sequence for next prediction
                    new_row = np.copy(scaled_data[-1])  # Copy last known features
                    new_row[0] = predicted_scaled  # Update only the price
                    new_sequence = np.concatenate([
                        current_sequence[0, 1:].numpy(),
                        new_row.reshape(1, -1)
                    ])
                    current_sequence = torch.FloatTensor(new_sequence).unsqueeze(0)
            
            # Create forecast dates
            forecast_dates = pd.date_range(
                start=last_known_data.index[-1],
                periods=steps + 1,
                freq=self.config.interval
            )[1:]
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'time': forecast_dates,
                'prediction': predictions
            })
            
            return forecast_df
            
        except Exception as e:
            print(f"Error during forecasting: {str(e)}")
            raise

    def _prepare_data(self, data, time_steps=None):
        """Prepare data into sequences for the LSTM."""
        if time_steps is None:
            time_steps = self.config.time_steps
        result = []
        if len(data) <= time_steps:
            time_steps = len(data) - 1
        for i in range(time_steps, len(data)):
            result.append(data[i - time_steps : i])
        return np.array(result)
