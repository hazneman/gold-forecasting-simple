"""
Configuration settings for Gold Price Forecasting System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application"""
    
    def __init__(self):
        # API Keys (store in .env file for security)
        self.METALS_API_KEY = os.getenv('METALS_API_KEY', '')
        self.FRED_API_KEY = os.getenv('FRED_API_KEY', '')
        
        # Data settings
        self.DATA_START_DATE = '2019-01-01'
        self.DATA_END_DATE = 'today'
        
        # Model parameters
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        
        # Random Forest parameters
        self.RF_N_ESTIMATORS = 100
        self.RF_MAX_DEPTH = 10
        self.RF_MIN_SAMPLES_SPLIT = 5
        
        # XGBoost parameters
        self.XGB_N_ESTIMATORS = 100
        self.XGB_MAX_DEPTH = 5
        self.XGB_LEARNING_RATE = 0.01
        
        # LSTM parameters
        self.LSTM_SEQUENCE_LENGTH = 30
        self.LSTM_EPOCHS = 50
        self.LSTM_BATCH_SIZE = 32
        
        # Technical indicators
        self.MA_PERIODS = [10, 20, 50, 200]
        self.RSI_PERIOD = 14
        self.BOLLINGER_PERIOD = 20
        
        # Risk management
        self.CONFIDENCE_LEVEL = 0.95
        self.RISK_PER_TRADE = 0.02
        self.STOP_LOSS_PERCENT = 0.05
        
        # Backtesting
        self.INITIAL_CAPITAL = 10000
        self.TRANSACTION_COST = 0.001  # 0.1%
        
        # API settings
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        
        # File paths
        self.MODEL_PATH = "data/models/"
        self.DATA_PATH = "data/"
        self.OUTPUT_PATH = "output/"
        
        # Create directories if they don't exist
        os.makedirs(self.MODEL_PATH, exist_ok=True)
        os.makedirs(self.DATA_PATH, exist_ok=True)
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        
    def get_api_headers(self, api_name='metals'):
        """Get API headers based on API name"""
        if api_name == 'metals':
            return {'api_key': self.METALS_API_KEY}
        elif api_name == 'fred':
            return {'api_key': self.FRED_API_KEY}
        return {}
    
    def validate_config(self):
        """Validate configuration settings"""
        missing_configs = []
        
        if not self.METALS_API_KEY:
            missing_configs.append("METALS_API_KEY")
        
        if not self.FRED_API_KEY:
            missing_configs.append("FRED_API_KEY")
        
        if missing_configs:
            print("Warning: Missing API keys:", ", ".join(missing_configs))
            print("Please add them to your .env file")
            return False
        
        return True