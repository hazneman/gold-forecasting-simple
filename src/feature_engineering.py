"""Feature Engineering Module

This module handles feature engineering for gold price forecasting including:
- Technical indicators
- Time-based features
- Lag features
- Rolling statistics
- Market sentiment features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import talib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for creating features for gold price forecasting."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler = None
        self.feature_names = []
        
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        data = df.copy()
        
        try:
            # Moving averages
            data['sma_5'] = data['Close'].rolling(window=5).mean()
            data['sma_10'] = data['Close'].rolling(window=10).mean()
            data['sma_20'] = data['Close'].rolling(window=20).mean()
            data['sma_50'] = data['Close'].rolling(window=50).mean()
            
            # Exponential moving averages
            data['ema_12'] = data['Close'].ewm(span=12).mean()
            data['ema_26'] = data['Close'].ewm(span=26).mean()
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_histogram'] = data['macd'] - data['macd_signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['bb_middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            data['bb_width'] = data['bb_upper'] - data['bb_lower']
            data['bb_position'] = (data['Close'] - data['bb_lower']) / data['bb_width']
            
            # Volume indicators
            if 'Volume' in data.columns:
                data['volume_sma'] = data['Volume'].rolling(window=20).mean()
                data['volume_ratio'] = data['Volume'] / data['volume_sma']
            
            logger.info("Technical indicators created successfully")
            
        except Exception as e:
            logger.error(f"Error creating technical indicators: {e}")
            
        return data
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features added
        """
        data = df.copy()
        
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, attempting to convert")
            data.index = pd.to_datetime(data.index)
        
        # Basic time features
        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['dayofweek'] = data.index.dayofweek
        data['dayofyear'] = data.index.dayofyear
        data['quarter'] = data.index.quarter
        
        # Cyclical features
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
        data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
        data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek'] / 7)
        data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek'] / 7)
        
        # Market timing features
        data['is_monday'] = (data['dayofweek'] == 0).astype(int)
        data['is_friday'] = (data['dayofweek'] == 4).astype(int)
        data['is_month_end'] = data.index.is_month_end.astype(int)
        data['is_month_start'] = data.index.is_month_start.astype(int)
        data['is_quarter_end'] = data.index.is_quarter_end.astype(int)
        
        logger.info("Time features created successfully")
        return data
    
    def create_lag_features(self, df: pd.DataFrame, 
                          target_col: str = 'Close', 
                          lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lag features for the target variable.
        
        Args:
            df: DataFrame with target column
            target_col: Name of the target column
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        data = df.copy()
        
        for lag in lags:
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        logger.info(f"Created lag features for lags: {lags}")
        return data
    
    def create_rolling_features(self, df: pd.DataFrame, 
                              target_col: str = 'Close',
                              windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create rolling statistical features.
        
        Args:
            df: DataFrame with target column
            target_col: Name of the target column
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        data = df.copy()
        
        for window in windows:
            data[f'{target_col}_rolling_mean_{window}'] = data[target_col].rolling(window).mean()
            data[f'{target_col}_rolling_std_{window}'] = data[target_col].rolling(window).std()
            data[f'{target_col}_rolling_min_{window}'] = data[target_col].rolling(window).min()
            data[f'{target_col}_rolling_max_{window}'] = data[target_col].rolling(window).max()
            data[f'{target_col}_rolling_skew_{window}'] = data[target_col].rolling(window).skew()
            data[f'{target_col}_rolling_kurt_{window}'] = data[target_col].rolling(window).kurt()
        
        logger.info(f"Created rolling features for windows: {windows}")
        return data
    
    def create_price_change_features(self, df: pd.DataFrame, 
                                   price_col: str = 'Close') -> pd.DataFrame:
        """Create price change and volatility features.
        
        Args:
            df: DataFrame with price column
            price_col: Name of the price column
            
        Returns:
            DataFrame with price change features added
        """
        data = df.copy()
        
        # Price changes
        data[f'{price_col}_pct_change'] = data[price_col].pct_change()
        data[f'{price_col}_diff'] = data[price_col].diff()
        data[f'{price_col}_log_return'] = np.log(data[price_col] / data[price_col].shift(1))
        
        # Volatility measures
        data['volatility_5'] = data[f'{price_col}_log_return'].rolling(5).std()
        data['volatility_10'] = data[f'{price_col}_log_return'].rolling(10).std()
        data['volatility_20'] = data[f'{price_col}_log_return'].rolling(20).std()
        
        # Price momentum
        data['momentum_5'] = data[price_col] / data[price_col].shift(5) - 1
        data['momentum_10'] = data[price_col] / data[price_col].shift(10) - 1
        data['momentum_20'] = data[price_col] / data[price_col].shift(20) - 1
        
        logger.info("Price change features created successfully")
        return data
    
    def scale_features(self, df: pd.DataFrame, 
                      method: str = 'standard',
                      exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: DataFrame to scale
            method: Scaling method ('standard' or 'minmax')
            exclude_cols: Columns to exclude from scaling
            
        Returns:
            DataFrame with scaled features
        """
        data = df.copy()
        exclude_cols = exclude_cols or []
        
        # Select numerical columns to scale
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        data[cols_to_scale] = self.scaler.fit_transform(data[cols_to_scale])
        
        logger.info(f"Features scaled using {method} method")
        return data
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features in sequence.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features created
        """
        logger.info("Starting feature engineering pipeline...")
        
        data = df.copy()
        
        # Create features in sequence
        data = self.create_technical_indicators(data)
        data = self.create_time_features(data)
        data = self.create_lag_features(data)
        data = self.create_rolling_features(data)
        data = self.create_price_change_features(data)
        
        # Store feature names
        self.feature_names = data.columns.tolist()
        
        logger.info(f"Feature engineering completed. Created {len(self.feature_names)} features")
        return data


def main():
    """Example usage of the feature engineer."""
    # This would typically use real data from data_collection module
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(len(dates)).cumsum() + 1800,
        'High': np.random.randn(len(dates)).cumsum() + 1810,
        'Low': np.random.randn(len(dates)).cumsum() + 1790,
        'Close': np.random.randn(len(dates)).cumsum() + 1800,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    fe = FeatureEngineer()
    featured_data = fe.create_all_features(sample_data)
    
    print(f"Original features: {sample_data.shape[1]}")
    print(f"Engineered features: {featured_data.shape[1]}")
    print(f"Feature names: {fe.feature_names[:10]}...")  # Show first 10


if __name__ == "__main__":
    main()