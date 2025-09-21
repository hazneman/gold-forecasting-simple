"""Machine Learning Models Module

This module contains various machine learning models for gold price forecasting including:
- Linear models (Linear Regression, Ridge, Lasso)
- Tree-based models (Random Forest, XGBoost, LightGBM)
- Neural networks (LSTM, GRU, Transformer)
- Time series models (ARIMA, Prophet)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import joblib
from abc import ABC, abstractmethod
import logging

# ML imports (will show import errors until packages are installed)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import lightgbm as lgb

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_importance = None
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'mape': np.mean(np.abs((y - predictions) / y)) * 100
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class LinearModel(BaseModel):
    """Linear regression models."""
    
    def __init__(self, model_type: str = 'linear', **kwargs):
        super().__init__(f"{model_type}_regression")
        
        if model_type == 'linear':
            self.model = LinearRegression(**kwargs)
        elif model_type == 'ridge':
            self.model = Ridge(**kwargs)
        elif model_type == 'lasso':
            self.model = Lasso(**kwargs)
        else:
            raise ValueError(f"Unsupported linear model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the linear model."""
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"{self.name} model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the linear model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)


class TreeModel(BaseModel):
    """Tree-based models."""
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        super().__init__(f"{model_type}")
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(**kwargs)
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(**kwargs)
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(**kwargs)
        else:
            raise ValueError(f"Unsupported tree model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the tree model."""
        self.model.fit(X, y)
        self.is_trained = True
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X.columns
            ).sort_values(ascending=False)
        
        logger.info(f"{self.name} model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the tree model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        if self.feature_importance is None:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.feature_importance


class LSTMModel(BaseModel):
    """LSTM neural network model."""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 units: int = 50,
                 dropout: float = 0.2,
                 layers: int = 2,
                 **kwargs):
        super().__init__("lstm")
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.layers = layers
        self.scaler_X = None
        self.scaler_y = None
        
    def prepare_sequences(self, X: pd.DataFrame, y: pd.Series = None) -> Tuple:
        """Prepare sequences for LSTM training."""
        from sklearn.preprocessing import StandardScaler
        
        # Scale features
        if self.scaler_X is None:
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(X)
        else:
            X_scaled = self.scaler_X.transform(X)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            if y is not None:
                y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            
            # Scale target if needed
            if self.scaler_y is None:
                self.scaler_y = StandardScaler()
                y_sequences = self.scaler_y.fit_transform(y_sequences.reshape(-1, 1)).ravel()
            else:
                y_sequences = self.scaler_y.transform(y_sequences.reshape(-1, 1)).ravel()
            
            return X_sequences, y_sequences
        
        return X_sequences
    
    def build_model(self, input_shape: Tuple) -> None:
        """Build the LSTM model architecture."""
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(units=self.units, 
                           return_sequences=True if self.layers > 1 else False,
                           input_shape=input_shape))
        self.model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, self.layers):
            return_seq = i < self.layers - 1
            self.model.add(LSTM(units=self.units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout))
        
        # Output layer
        self.model.add(Dense(1))
        
        # Compile model
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='mse',
                          metrics=['mae'])
        
        logger.info("LSTM model architecture built")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the LSTM model."""
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.build_model((X_seq.shape[1], X_seq.shape[2]))
        
        # Split data
        split_idx = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        # Callbacks
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.is_trained = True
        logger.info("LSTM model trained successfully")
        
        return history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the LSTM model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_seq = self.prepare_sequences(X)
        predictions = self.model.predict(X_seq)
        
        # Inverse transform predictions
        if self.scaler_y is not None:
            predictions = self.scaler_y.inverse_transform(predictions.reshape(-1, 1)).ravel()
        
        return predictions


class ModelEnsemble:
    """Ensemble of multiple models."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)
        self.is_trained = False
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all models in the ensemble."""
        for model in self.models:
            logger.info(f"Training {model.name}...")
            model.fit(X, y)
        
        self.is_trained = True
        logger.info("Ensemble training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate ensemble and individual model performance."""
        results = {}
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X)
        results['ensemble'] = {
            'mse': mean_squared_error(y, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y, ensemble_pred)),
            'mae': mean_absolute_error(y, ensemble_pred),
            'r2': r2_score(y, ensemble_pred),
            'mape': np.mean(np.abs((y - ensemble_pred) / y)) * 100
        }
        
        # Evaluate individual models
        for model in self.models:
            results[model.name] = model.evaluate(X, y)
        
        return results


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Factory function to create models."""
    if model_type in ['linear', 'ridge', 'lasso']:
        return LinearModel(model_type, **kwargs)
    elif model_type in ['random_forest', 'xgboost', 'lightgbm']:
        return TreeModel(model_type, **kwargs)
    elif model_type == 'lstm':
        return LSTMModel(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    """Example usage of the models."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    n_samples = len(dates)
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)],
        index=dates
    )
    y = pd.Series(
        np.random.randn(n_samples).cumsum() + 1800,
        index=dates,
        name='price'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Create and train models
    models = [
        create_model('linear'),
        create_model('random_forest', n_estimators=100),
        create_model('xgboost', n_estimators=100)
    ]
    
    # Train models
    for model in models:
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        print(f"{model.name} - RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.3f}")
    
    # Create ensemble
    ensemble = ModelEnsemble(models)
    ensemble.fit(X_train, y_train)
    
    ensemble_metrics = ensemble.evaluate(X_test, y_test)
    print(f"Ensemble - RMSE: {ensemble_metrics['ensemble']['rmse']:.2f}, "
          f"R2: {ensemble_metrics['ensemble']['r2']:.3f}")


if __name__ == "__main__":
    main()