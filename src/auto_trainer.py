"""
Enhanced Auto ML Trainer for Gold Price Prediction
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import os

# ML imports with graceful fallback
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class AutoMLTrainer:
    """Automatic ML trainer for gold price prediction"""
    
    def __init__(self):
        self.training_data = None
        self.models = {}
        self.feature_names = []
        self.model_dir = "models"
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
    def collect_training_data(self, years=2):
        """Collect and prepare training data with technical indicators"""
        
        logger.info(f"📊 Collecting {years} years of gold price data...")
        
        try:
            # Get gold price data - use multiple fallbacks for reliability
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Try gold futures first (most accurate price representation)
            gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
            data_source = 'GC=F (Gold Futures)'
            
            if gold.empty:
                logger.warning("No data found for GC=F, trying XAUUSD=X")
                gold = yf.download('XAUUSD=X', start=start_date, end=end_date, progress=False)
                data_source = 'XAUUSD=X (Gold Spot)'
            
            if gold.empty:
                logger.error("❌ Could not fetch gold price data from any source")
                raise ValueError("Could not fetch gold price data - all sources failed")
            
            logger.info(f"📈 Downloaded {len(gold)} days of data from {data_source}")
            
            # Handle multi-level columns from yfinance
            if gold.columns.nlevels > 1:
                # Flatten multi-level columns
                gold.columns = [col[0] for col in gold.columns]
            
            # Create a clean copy with basic columns
            df = gold[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Log current price to verify data source correctness
            current_price = df['Close'].iloc[-1]
            logger.info(f"💰 Current gold price from {data_source}: ${current_price:.2f}")
            
            # Validate price range (gold should be > $1000/oz)
            if current_price < 1000:
                logger.error(f"❌ Suspicious price data: ${current_price:.2f} - likely wrong data source")
                raise ValueError(f"Gold price ${current_price:.2f} seems too low - possible data source error")
            
            logger.info(f"✅ Price validation passed: ${current_price:.2f} is reasonable for gold")
            
            # Forward fill any missing values
            df = df.ffill()
            
            logger.info("🔧 Calculating technical indicators...")
            
            # Simple moving averages
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # Price ratios
            df['price_to_sma_20'] = df['Close'] / df['sma_20']
            df['price_to_sma_50'] = df['Close'] / df['sma_50']
            
            # RSI
            df['rsi'] = self._calculate_rsi(df['Close'])
            
            # MACD
            macd_line = df['ema_12'] - df['ema_26']
            df['macd'] = macd_line
            df['macd_signal'] = macd_line.ewm(span=9).mean()
            
            # Returns
            df['returns_1d'] = df['Close'].pct_change()
            df['returns_5d'] = df['Close'].pct_change(5)
            
            # Volatility
            df['volatility_10d'] = df['returns_1d'].rolling(10).std()
            
            # Momentum
            df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
            
            # Volume
            df['volume_sma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
            
            # Time features
            df['month'] = df.index.month
            df['day_of_week'] = df.index.dayofweek
            
            # Simple trend indicator
            df['is_uptrend'] = (df['Close'] > df['sma_20']).astype(int)
            
            # Clean data
            logger.info("🧹 Cleaning data...")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            self.training_data = df
            logger.info(f"✅ Prepared {len(self.training_data)} training samples with {len(self.training_data.columns)} features")
            
            return self.training_data
            
        except Exception as e:
            logger.error(f"❌ Error collecting training data: {e}")
            raise
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI
        except:
            # Fallback - return neutral RSI
            return pd.Series([50] * len(prices), index=prices.index)
    
    def train_prediction_models(self):
        """Train models for different time horizons"""
        
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available")
            return {"error": "ML libraries not available"}
        
        # Collect data if not already done
        if self.training_data is None:
            self.collect_training_data()
        
        if self.training_data is None or len(self.training_data) < 100:
            logger.error("Insufficient training data")
            return {"error": "Insufficient training data"}
        
        logger.info("🚀 Training ML models...")
        
        model_paths = {}
        time_horizons = [1, 3, 7, 14]
        
        # Prepare features (exclude target and non-feature columns)
        feature_cols = [col for col in self.training_data.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = self.training_data[feature_cols].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        for horizon in time_horizons:
            try:
                logger.info(f"Training {horizon}-day model...")
                
                # Create target variable (future price)
                y = self.training_data['Close'].shift(-horizon).dropna()
                
                # Align X and y
                X_aligned = X.loc[y.index]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_aligned, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"✅ {horizon}-day model: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
                
                # Save model and scaler
                model_path = os.path.join(self.model_dir, f"gold_model_{horizon}d.pkl")
                scaler_path = os.path.join(self.model_dir, f"scaler_{horizon}d.pkl")
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                
                self.models[f"{horizon}d"] = {
                    'model': model,
                    'scaler': scaler,
                    'features': feature_cols,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                }
                
                model_paths[f"{horizon}d"] = model_path
                
            except Exception as e:
                logger.error(f"❌ Error training {horizon}-day model: {e}")
                
        logger.info(f"🎯 Training completed for {len(model_paths)} models")
        return model_paths
    
    def predict_with_trained_model(self, horizon_days=7):
        """Make prediction using trained model"""
        
        try:
            # Load model if not in memory
            model_key = f"{horizon_days}d"
            
            if model_key not in self.models:
                model_path = os.path.join(self.model_dir, f"gold_model_{horizon_days}d.pkl")
                scaler_path = os.path.join(self.model_dir, f"scaler_{horizon_days}d.pkl")
                
                if not os.path.exists(model_path):
                    logger.warning(f"Model for {horizon_days} days not found, training...")
                    self.train_prediction_models()
                    
                    if not os.path.exists(model_path):
                        raise ValueError(f"Could not create model for {horizon_days} days")
                
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                self.models[model_key] = {
                    'model': model,
                    'scaler': scaler
                }
            
            # Get latest data
            if self.training_data is None:
                self.collect_training_data()
            
            # Prepare features from latest data
            latest_data = self.training_data.iloc[-1:].copy()
            
            # Get feature columns (same as training)
            feature_cols = [col for col in latest_data.columns 
                           if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            X_latest = latest_data[feature_cols].values
            
            # Handle NaN values
            X_latest = np.nan_to_num(X_latest, nan=0.0)
            
            # Scale features
            X_scaled = self.models[model_key]['scaler'].transform(X_latest)
            
            # Make prediction
            prediction = self.models[model_key]['model'].predict(X_scaled)[0]
            
            logger.info(f"🔮 {horizon_days}-day prediction: ${prediction:.2f}")
            return float(prediction)
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            # Return current price as fallback
            try:
                if self.training_data is not None:
                    return float(self.training_data['Close'].iloc[-1])
                else:
                    return 2000.0  # Default fallback
            except:
                return 2000.0
    
    def get_training_status(self):
        """Get current training status"""
        
        status = {
            'status': 'not_trained',
            'message': 'No trained models found',
            'models': {},
            'data_samples': 0,
            'last_updated': None,
            'recommendation': 'Call the training endpoint to train models'
        }
        
        # Check for existing models
        model_files = []
        if os.path.exists(self.model_dir):
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl') and 'gold_model' in f]
        
        if model_files:
            status['status'] = 'trained'
            status['message'] = f'Found {len(model_files)} trained models'
            
            for model_file in model_files:
                horizon = model_file.replace('gold_model_', '').replace('.pkl', '')
                model_path = os.path.join(self.model_dir, model_file)
                
                # Get file modification time
                mtime = os.path.getmtime(model_path)
                last_updated = datetime.fromtimestamp(mtime).isoformat()
                
                status['models'][horizon] = {
                    'path': model_path,
                    'last_updated': last_updated
                }
                
                if status['last_updated'] is None or last_updated > status['last_updated']:
                    status['last_updated'] = last_updated
        
        # Check training data
        if self.training_data is not None:
            status['data_samples'] = len(self.training_data)
        
        return status


# Create global instance
auto_trainer = AutoMLTrainer()