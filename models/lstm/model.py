import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

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

    def __init__(self, model_name="lstm", config=LstmConfig(), debug=False):
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

    # pylint: disable=too-many-locals,too-many-statements
    def train(self, data):
        """Train the LSTM model with either DataFrame or preprocessed numpy array."""
        if isinstance(data, pd.DataFrame):
            # Process DataFrame input
            # Initialize the scaler
            scaler_funding   = MinMaxScaler(feature_range=(0, 1))
            scaler_open_interest = MinMaxScaler(feature_range=(0, 1))
            scaler_premium = MinMaxScaler(feature_range=(0, 1))
            scaler_day_ntl_vlm = MinMaxScaler(feature_range=(0, 1))
            scaler_current_price = MinMaxScaler(feature_range=(0, 1))
            scaler_long_number = MinMaxScaler(feature_range=(0, 1))
            scaler_short_number = MinMaxScaler(feature_range=(0, 1))

            # Ensure the 'date' column is a DatetimeIndex for resampling
            if "time" in data.columns:
                data["time"] = pd.to_datetime(data["time"])
                data = data.set_index("time")

            # Select only numeric columns for resampling
            numeric_data = data.select_dtypes(include=[np.number])  # type: ignore

            # Resample the data based on the configured interval
            numeric_data = numeric_data.resample(self.config.interval).mean().dropna()

            # Reattach the non-numeric data (e.g., 'asset' column) after resampling if needed
            if "asset" in data.columns:
                asset_data = data[["asset"]].resample(self.config.interval).first()
                data = numeric_data.join(asset_data)
            else:
                data = numeric_data

            # Normalize the data
            funding = data["funding"].values.astype(float).reshape(-1, 1)
            open_interest = data["open_interest"].values.astype(float).reshape(-1, 1)
            premium = data["premium"].values.astype(float).reshape(-1, 1)
            day_ntl_vlm = data["day_ntl_vlm"].values.astype(float).reshape(-1, 1)
            current_price = data["current_price"].values.astype(float).reshape(-1, 1)
            long_number = data["long_number"].values.astype(float).reshape(-1, 1)
            short_number = data["short_number"].values.astype(float).reshape(-1, 1)
            

            # Scale each feature
            scaled_funding = scaler_funding.fit_transform(funding)
            scaled_open_interest = scaler_open_interest.fit_transform(open_interest)
            scaled_premium = scaler_premium.fit_transform(premium)
            scaled_day_ntl_vlm = scaler_day_ntl_vlm.fit_transform(day_ntl_vlm)
            scaled_current_price = scaler_current_price.fit_transform(current_price)
            scaled_long_number = scaler_long_number.fit_transform(long_number)
            scaled_short_number = scaler_short_number.fit_transform(short_number)

            # Stack the scaled features horizontally into a single array
            scaled_data = np.hstack((
                scaled_current_price,
                scaled_funding,
                scaled_open_interest,
                scaled_premium,
                scaled_day_ntl_vlm,
                scaled_long_number,
                scaled_short_number
            ))

            # Prepare sequences
            train_data = self._prepare_data(scaled_data)
        else:
            # Input is already preprocessed numpy array
            train_data = data

        # Split into x and y
        x_train, y_train = train_data[:-1], train_data[1:]

        # Split data into training and validation sets
        val_size = int(len(x_train) * self.config.validation_split)
        x_train, x_val = x_train[:-val_size], x_train[-val_size:]
        y_train, y_val = y_train[:-val_size], y_train[-val_size:]

        # Convert to tensors with correct dimensions [batch_size, sequence_length, features]
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train[:, -1, 0], dtype=torch.float32).unsqueeze(-1)  # Only use close price for target
        x_val = torch.tensor(x_val, dtype=torch.float32)
        y_val = torch.tensor(y_val[:, -1, 0], dtype=torch.float32).unsqueeze(-1)

        # Create DataLoader for mini-batching
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        best_val_loss = float("inf")
        patience_counter = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:

                outputs, _ = self.model(inputs)  # Get only the output
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )  # Gradient clipping
                self.optimizer.step()
                epoch_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs, _ = self.model(x_val)  # Get only the output
                val_loss = self.criterion(val_outputs, y_val).item()
            self.model.train()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.config.epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
                )

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Save the model
        self.save()

    # pylint: disable=too-many-branches,too-many-statements
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

    # pylint: disable=arguments-differ
    def forecast(self, steps: int, last_known_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast future values based on the last known data."""
        self.model.eval()

        # Initialize scalers for each feature
        scaler_funding = MinMaxScaler(feature_range=(0, 1))
        scaler_open_interest = MinMaxScaler(feature_range=(0, 1))
        scaler_premium = MinMaxScaler(feature_range=(0, 1))
        scaler_day_ntl_vlm = MinMaxScaler(feature_range=(0, 1))
        scaler_current_price = MinMaxScaler(feature_range=(0, 1))
        scaler_long_number = MinMaxScaler(feature_range=(0, 1))
        scaler_short_number = MinMaxScaler(feature_range=(0, 1))

        # Ensure 'date' column exists and set it as index for resampling
        if "time" in last_known_data.columns:
            last_known_data["time"] = pd.to_datetime(last_known_data["time"])
            last_known_data = last_known_data.set_index("time")

        # Resample based on the specified interval in the config
        last_known_data = last_known_data.resample(self.config.interval).mean().dropna()

        # Ensure there is enough data to use for forecasting
        if len(last_known_data) < self.config.time_steps:
            raise ValueError(
                f"Not enough data to generate a forecast. Required at least {self.config.time_steps} data points."
            )

        # Scale each feature
        funding = last_known_data["funding"].values.astype(float).reshape(-1, 1)
        open_interest = last_known_data["open_interest"].values.astype(float).reshape(-1, 1)
        premium = last_known_data["premium"].values.astype(float).reshape(-1, 1)
        day_ntl_vlm = last_known_data["volume"].values.astype(float).reshape(-1, 1)
        current_price = last_known_data["current_price"].values.astype(float).reshape(-1, 1)
        long_number = last_known_data["long_number"].values.astype(float).reshape(-1, 1)
        short_number = last_known_data["short_number"].values.astype(float).reshape(-1, 1)

        # Scale each feature
        scaled_funding = scaler_funding.fit_transform(funding)
        scaled_open_interest = scaler_open_interest.fit_transform(open_interest)
        scaled_premium = scaler_premium.fit_transform(premium)
        scaled_day_ntl_vlm = scaler_day_ntl_vlm.fit_transform(day_ntl_vlm)
        scaled_current_price = scaler_current_price.fit_transform(current_price)
        scaled_long_number = scaler_long_number.fit_transform(long_number)
        scaled_short_number = scaler_short_number.fit_transform(short_number)

        # Stack the scaled features horizontally
        scaled_data = np.hstack((
            scaled_current_price,
            scaled_funding,
            scaled_open_interest,
            scaled_premium,
            scaled_day_ntl_vlm,
            scaled_long_number,
            scaled_short_number
        ))

        # Prepare the last sequence of data for forecasting
        last_sequence = scaled_data[-self.config.time_steps:].reshape(1, self.config.time_steps, 7)
        predictions = []

        with torch.no_grad():
            for step in range(steps):
                inputs = torch.tensor(last_sequence, dtype=torch.float32)
                predicted_scaled, _ = self.model(inputs)  # Pass the inputs through the model
                predicted_scaled = predicted_scaled.cpu().numpy()

                # Inverse transform the predicted value (only for close price)
                predicted = scaler_current_price.inverse_transform(predicted_scaled)
                predictions.append(predicted.flatten()[0])

                # Create next sequence input (shift window and append prediction)
                # We'll repeat the last known values for other features
                new_row = np.array([
                    predicted_scaled[0, 0],  # current_price
                    scaled_data[-1, 1],      # funding
                    scaled_data[-1, 2],      # open_interest
                    scaled_data[-1, 3],      # premium
                    scaled_data[-1, 4],      # day_ntl_vlm
                    scaled_data[-1, 5],      # long_number
                    scaled_data[-1, 6],      # short_number
                ]).reshape(1, -1)

                last_sequence = np.concatenate([
                    last_sequence[:, 1:, :],
                    new_row.reshape(1, 1, 7)
                ], axis=1)

                if self.debug:
                    print(f"Step {step + 1}/{steps}, Predicted: {predicted.flatten()[0]}")

        # Create forecast dates
        forecast_dates = pd.date_range(
            start=last_known_data.index[-1],
            periods=steps + 1,
            freq=self.config.interval,
        )[1:]  # Exclude the starting date

        # Create a DataFrame for the forecasted values
        df_forecast = pd.DataFrame({
            "time": forecast_dates,
            "Forecasted Close": predictions,
        })

        return df_forecast

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
