from ast import Lambda
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM,  Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import scipy.stats as st
import tensorflow as tf

import joblib
from pathlib import Path

from models.base_model import Model
from models.lstm.configs import LstmConfig
from utils.common import print_colored


# PermaDropout is a dropout layer that is enables even for inference, allowing to perform Bayesian estimation
def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))

# Define the LSTM model class that integrates with the base model
class LstmModel(Model):
    """LSTM model for time series forecasting"""

    def __init__(self, model_name="lstm", config=None, debug=False):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config or LstmConfig()
        self.model = None
        self.scalers = {}
        
        self.feature_names = [
            'percent_change', 'funding', 'open_interest', 'premium',
            'day_ntl_vlm', 'long_number', 'short_number'
        ]
        
        # Disable TensorFlow warnings
        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.model = self._build_model()

    def _build_model(self):
        """Build LSTM model architecture with proper input shape"""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=self.config.hidden_size,
                      return_sequences=True,
                      input_shape=(self.config.time_steps, self.config.features)))
        model.add(PermaDropout(self.config.dropout))
        
        # Second LSTM layer
        model.add(LSTM(units=self.config.hidden_size2,
                      return_sequences=False))
        model.add(PermaDropout(self.config.dropout))
        
        # Output layer
        model.add(Dense(units=self.config.output_size))
        
        # Compile model
        model.compile(optimizer=self.config.optimizer,
                     loss=self.config.loss)
        
        return model

    def _initialize_scalers(self):
        """Initialize scalers for each feature"""
        for feature in self.feature_names:
            self.scalers[feature] = RobustScaler() if feature in ['percent_change', 'day_ntl_vlm'] else StandardScaler()

    def _scale_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Scale features using appropriate scalers
        
        Args:
            data: DataFrame containing features to scale
            
        Returns:
            Scaled features as numpy array
        """
        if not self.scalers:
            self._initialize_scalers()
        
        scaled_features = []
        for feature in self.feature_names:
            values = data[feature].values.reshape(-1, 1)
            if not hasattr(self.scalers[feature], 'mean_'):
                self.scalers[feature].fit(values)
            scaled_features.append(self.scalers[feature].transform(values))
        
        # self._save_scalers()
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

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for training/forecasting"""
        if "time" in data.columns:
            data = data.set_index("time")
        data = data.resample(self.config.interval).mean().dropna()
        data['percent_change'] = data['current_price'].pct_change(self.config.window)
        return data.dropna()

    def _save(self):
        """Save the model and scaler (if applicable) to disk."""
        os.makedirs(self.save_dir, exist_ok=True)
        model_dir = os.path.join(self.save_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save the model (joblib or pickle) and scaler (if applicable)
        # self.model.save(os.path.join(model_dir, "model.h5"))
        joblib.dump(self.scalers, os.path.join(model_dir, "scalers.pkl"))
        if self.debug:
            print_colored(
                f"Model and scaler saved as {model_dir}/model.h5 and {model_dir}/scalers.pkl",
                "success",
            )
            
    def _load(self):
        """Load model and scalers"""
        try:
            model_dir = Path(f'trained_models/{self.model_name}')
            
            # Instead of loading the model directly, load its weights
            model_path = model_dir / 'model.weights.h5'
            if not model_path.exists():
                raise FileNotFoundError("Model file not found")
            
            # Create a new model with the same architecture
            self.model = self._build_model()
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss=self.config.loss
            )
            # Load the weights
            self.model.load_weights(model_path)
            
            # Load scalers
            scalers_path = model_dir / 'scalers.pkl'
            if scalers_path.exists():
                self.scalers = joblib.load(scalers_path)
            else:
                raise FileNotFoundError("Scalers file not found")
                
            print_colored(f"Model and scalers loaded from {model_dir}", "success")
            
        except Exception as e:
            print_colored(f"Error loading model: {str(e)}", "error")
            raise

    def train(self, data: pd.DataFrame) -> None:
        """Train the LSTM model"""
        try:
            data = self._prepare_data(data)
            scaled_data = self._scale_features(data)
            
            X, y = self._create_sequences(scaled_data, self.config.time_steps)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, mode='min'),
                ModelCheckpoint(
                    f'trained_models/{self.model_name}/model.weights.h5',
                    save_best_only=True,
                    monitor='val_loss',
                    save_weights_only=True
                )
            ]
            
            self.model.fit(
                X_train, y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            self._save()
            
        except Exception as e:
            print_colored(f"Training error: {str(e)}", "error")
            raise

    def forecast(self, steps: int, last_known_data: pd.DataFrame) -> pd.DataFrame:
        """Generate forecasts"""
        try:
            self._load()
            data = self._prepare_data(last_known_data)
            scaled_data = self._scale_features(data[-self.config.time_steps:])
            sequence = scaled_data.reshape(1, self.config.time_steps, len(self.feature_names))

            prices = last_known_data['current_price'].values[-self.config.window:(len(last_known_data) - (self.config.window - steps))]
            
            predictions = []
            for i in range(self.config.simulations):
                pred = self.model.predict(sequence, verbose=0)
                pred = self.scalers['percent_change'].inverse_transform(pred)
                pred = prices * (pred + 1)
                predictions.append(pd.DataFrame(pred))
            predictions = pd.concat(predictions, axis=0)

            forecast_mean = predictions.mean(axis=0)
            forecast_mean = np.maximum(forecast_mean, 0)
            forecast_mean.name = 'forecast'
            forecast_std = predictions.std(axis=0)
            
            z_norm = st.norm.ppf(self.config.conf_int)
            forecast_upper = forecast_mean + z_norm * forecast_std
            forecast_upper = np.maximum(forecast_upper, 0)
            forecast_upper.name = 'forecast_upper'
            
            forecast_lower = forecast_mean - z_norm * forecast_std
            forecast_lower = np.maximum(forecast_lower, 0)
            forecast_lower.name = 'forecast_lower'         
            
            forecast_dates = pd.date_range(
                start=data.index[-1],
                periods=len(forecast_mean) + 1,
                freq=self.config.interval
            )[1:]

            
            forecast_df = pd.DataFrame(
                {'forecast': forecast_mean.values,
                 'forecast_lower': forecast_lower.values,
                 'forecast_upper': forecast_upper.values,
                 },
                index=forecast_dates
            )
                        
            return forecast_df
            
        except Exception as e:
            print_colored(f"Forecasting error: {str(e)}", "error")
            raise