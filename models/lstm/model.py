import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Lambda, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

import joblib
from pathlib import Path

from models.base_model import Model
from models.lstm.configs import LstmConfig
from utils.common import print_colored


# PermaDropout is a dropout layer that is enables even for inference, allowing to perform Bayesian estimation
# def PermaDropout(rate):
#     return Lambda(lambda x: K.dropout(x, level=rate))

# Define the LSTM model class that integrates with the base model
class LstmModel(Model):
    """LSTM model for time series forecasting"""

    def __init__(self, model_name="lstm", config=LstmConfig(), debug=True):
        super().__init__(model_name=model_name, model_type="pytorch", debug=debug)
        self.config = config  # Use the configuration class

        #Initialize the model
        self.model = Sequential()
        
        #Input layer
        self.model.add(LSTM(self.config.hidden_size, activation='relu', return_sequences=True, input_shape=(self.config.time_steps, self.config.features)))
        self.model.add(Dropout(rate=0.2))

        self.model.add(LSTM(self.config.hidden_size, activation="relu", return_sequences=False))
        self.model.add(Dropout(rate=0.2))
        
        # deep layers
        self.model.add(Dense(self.config.hidden_size2))
        # model.add(PermaDropout(rate=0.2))
        
        # output layers
        self.model.add(Dense(self.config.n_steps_out))
        
        # compiling the model
        self.model.compile(optimizer=self.config.optimizer, loss=self.config.loss)

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

    def _save(self):
        """Save the model and scaler (if applicable) to disk."""
        print("asdfandfsdfsdfsdfsdf")
        os.makedirs(self.save_dir, exist_ok=True)
        model_dir = os.path.join(self.save_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model (joblib or pickle) and scaler (if applicable)
        joblib.dump(self.model, os.path.join(model_dir, "model.pkl"))
        if self.scaler:
            joblib.dump(
                self.scaler,
                os.path.join(model_dir, "scaler.pkl"),
            )
        if self.debug:
            print_colored(
                f"Model and scaler saved as {model_dir}/model.pkl and {model_dir}/scaler.pkl",
                "success",
            )

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
        
        self._save_scalers()
        return np.hstack(scaled_features)

    def _create_sequences(self, scaled_data: np.ndarray, 
                         sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for training/inference"""
        sequences = []
        targets = []
        
        # Ensure we have enough data for both sequence and target
        for i in range(len(scaled_data) - sequence_length - self.config.n_steps_out + 1):
            # Input sequence
            sequence = scaled_data[i:(i + sequence_length)]
            # Target sequence - ensure it's the correct shape
            target = scaled_data[i + sequence_length:i + sequence_length + self.config.n_steps_out, 0]
            
            # Only append if we have complete sequences
            if len(target) == self.config.n_steps_out:
                sequences.append(sequence)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def _plot_loss(self, history):
        plt.figure(figsize=(14, 4))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def train(self, data):
        """Train the LSTM model with improved feature scaling"""
        try:
            # Handle both DataFrame and numpy array inputs
            # if isinstance(data, pd.DataFrame):
            if "time" in data.columns:
                data["time"] = pd.to_datetime(data["time"])
                data = data.set_index("time")
            data = data.resample(self.config.interval).mean().dropna()
            scaled_data = self._scale_features(data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, self.config.time_steps)
            print(X.shape)
            print(y.shape)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=False)

            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            # Training loop

            earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')

            # decaying lr on every 2nd epoch
            def scheduler(epoch, lr):
                if epoch % 2 == 0:
                    return lr * .999
                return lr

            LRscheduler = LearningRateScheduler(scheduler)

            # storing all the callbacks
            callbacks_list = [earlystop, LRscheduler]
            
            history = self.model.fit(X_train, y_train, batch_size=self.config.batch_size, epochs=self.config.epochs,
                    verbose=1, callbacks=callbacks_list,
                    validation_data=(X_test, y_test))
                        
            self._save()
            
            self._plot_loss(history)
            
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def forecast(self, steps: int, last_known_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast future values with improved scaling"""
        try:
            self._load_scalers()  # Load saved scalers
            
            # Prepare input data
            if "time" in last_known_data.columns:
                last_known_data = last_known_data.set_index("time")
            last_known_data = last_known_data.resample(self.config.interval).mean().dropna()
            
            # Scale features
            scaled_data = self._scale_features(last_known_data[-self.config.time_steps:])


            # Prepare last sequence
            last_sequence = scaled_data.reshape(
                1, self.config.time_steps, len(self.feature_names)
            )            
                        
            # Generate predictions
            predictions = self.model.predict(last_sequence)
            
            predictions = self.scalers['current_price'].inverse_transform(predictions)

            
            # Create forecast dates
            forecast_dates = pd.date_range(
                start=last_known_data.index[-1],
                periods=self.config.n_steps_out + 1,
                freq=self.config.interval
            )[1:]

            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'time': forecast_dates,
                'forecast': predictions.flatten()
            })
            
            forecast_df = forecast_df.set_index('time')
            
            return forecast_df
            
        except Exception as e:
            print(f"Error during forecasting: {str(e)}")
            raise

