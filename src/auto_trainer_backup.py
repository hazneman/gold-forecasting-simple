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
            # Get gold price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
            
            if gold.empty:
                logger.warning("No data found for GC=F, trying GOLD")
                gold = yf.download('GOLD', start=start_date, end=end_date, progress=False)
            
            if gold.empty:
                raise ValueError("Could not fetch gold price data")
            
            logger.info(f"📈 Downloaded {len(gold)} days of data")
            
            # Handle multi-level columns from yfinance
            if gold.columns.nlevels > 1:
                # Flatten multi-level columns
                gold.columns = [col[0] for col in gold.columns]
            
            # Create a clean copy with basic columns
            df = gold[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
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

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import logging
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class AutoMLTrainer:
    """Automatically train ML models on historical gold price data"""
    
    def __init__(self):
        self.models = {}
        self.training_data = None
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        
    def train_prediction_models(self):
        """Collect comprehensive training data"""
        logger.info("📊 Collecting training data for ML models...")
        
        try:
            # Gold price data
            gold = yf.download('GC=F', period=period, interval='1d', progress=False)
            
            if gold.empty:
                raise ValueError("Unable to fetch gold price data")
            
            # Economic indicators
            symbols = {
                'DXY': 'DX-Y.NYB',
                'VIX': '^VIX',
                'SP500': '^GSPC',
                'TNX': '^TNX',
                'OIL': 'CL=F'
            }
            
            # Collect economic data
            economic_data = {}
            for name, symbol in symbols.items():
                try:
                    data = yf.download(symbol, period=period, interval='1d', progress=False)
                    if not data.empty:
                        economic_data[name] = data['Close']
                except:
                    logger.warning(f"Could not fetch {name} data")
                    continue
            
            # Create features DataFrame
            df = gold.copy()
            
            # Add economic indicators
            for name, data in economic_data.items():
                df[f'{name}_price'] = data
                df[f'{name}_change_1d'] = data.pct_change(1)
                df[f'{name}_change_5d'] = data.pct_change(5)
            
            # Calculate technical indicators
            logger.info("Calculating moving averages...")
            df['sma_5'] = df['Close'].rolling(5).mean()
            df['sma_10'] = df['Close'].rolling(10).mean()
            df['sma_20'] = df['Close'].rolling(20).mean()
            df['sma_50'] = df['Close'].rolling(50).mean()
            df['sma_200'] = df['Close'].rolling(200).mean()
            
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # Price ratios
            df['price_to_sma_20'] = df['Close'] / df['sma_20']
            df['price_to_sma_50'] = df['Close'] / df['sma_50']
            
            # RSI
            logger.info("Calculating RSI...")
            df['rsi'] = self._calculate_rsi(df['Close'])
            
            # MACD
            macd_line = df['ema_12'] - df['ema_26']
            df['macd'] = macd_line
            df['macd_signal'] = macd_line.ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb_sma = df['Close'].rolling(20).mean()
            bb_std = df['Close'].rolling(20).std()
            df['bb_upper'] = bb_sma + (bb_std * 2)
            df['bb_lower'] = bb_sma - (bb_std * 2)
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Returns and momentum
            df['returns_1d'] = df['Close'].pct_change()
            df['returns_3d'] = df['Close'].pct_change(3)
            df['returns_5d'] = df['Close'].pct_change(5)
            df['returns_10d'] = df['Close'].pct_change(10)
            
            # Volatility
            df['volatility_5d'] = df['returns_1d'].rolling(5).std()
            df['volatility_20d'] = df['returns_1d'].rolling(20).std()
            
            # Volume features
            df['volume_sma'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            
            # Time features
            df['month'] = df.index.month
            df['day_of_week'] = df.index.dayofweek
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Market regime indicators
            df['bull_market'] = (df['Close'] > df['sma_200']).astype(int)
            df['uptrend_strength'] = (
                (df['sma_5'] > df['sma_20']).astype(int) +
                (df['sma_20'] > df['sma_50']).astype(int) +
                (df['Close'] > df['sma_20']).astype(int)
            )
            
            # Clean data
            self.training_data = df.dropna()
            logger.info(f"✅ Collected {len(self.training_data)} training samples with {len(self.training_data.columns)} features")
            
            return self.training_data
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            raise
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_prediction_models(self):
        """Train multiple prediction models for different time horizons"""
        
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available, skipping training")
            return {"error": "ML libraries not available"}
        
        if self.training_data is None:
            self.collect_training_data()
        
        logger.info("🤖 Training ML models for gold price prediction...")
        
        # Prepare feature columns (exclude target and non-numeric)
        exclude_cols = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        feature_cols = [col for col in self.training_data.columns 
                       if col not in exclude_cols 
                       and self.training_data[col].dtype in ['float64', 'int64']]
        
        X = self.training_data[feature_cols].fillna(method='ffill')
        
        # Train models for different time horizons
        horizons = [1, 3, 5, 7, 14]
        results = {}
        
        for horizon in horizons:
            try:
                logger.info(f"  Training {horizon}-day prediction model...")
                
                # Create target (future price change percentage)
                y = (self.training_data['Close'].shift(-horizon) / self.training_data['Close'] - 1) * 100
                
                # Align data and remove NaN
                mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 100:
                    logger.warning(f"  ⚠️ Insufficient data for {horizon}-day model")
                    continue
                
                # Split train/test (80/20)
                split_idx = int(len(X_clean) * 0.8)
                X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
                y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Random Forest model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Feature importance
                feature_importance = pd.Series(
                    model.feature_importances_, 
                    index=feature_cols
                ).sort_values(ascending=False)
                
                # Save model
                model_path = self.model_dir / f"gold_prediction_{horizon}d.pkl"
                joblib.dump({
                    'model': model,
                    'scaler': scaler,
                    'features': feature_cols,
                    'mae': mae,
                    'r2': r2,
                    'feature_importance': feature_importance.head(10).to_dict(),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'trained_at': datetime.now().isoformat(),
                    'horizon_days': horizon
                }, model_path)
                
                self.models[f'{horizon}d'] = {
                    'model': model,
                    'scaler': scaler,
                    'features': feature_cols
                }
                
                results[f'{horizon}d'] = {
                    'mae': round(mae, 3),
                    'r2_score': round(r2, 3),
                    'accuracy_pct': round((1 - abs(mae)/100)*100, 1),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test),
                    'top_features': feature_importance.head(5).to_dict()
                }
                
                logger.info(f"    ✅ {horizon}-day model: MAE={mae:.2f}%, R²={r2:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {horizon}-day model: {e}")
                continue
        
        # Save training summary
        summary = {
            'training_completed': datetime.now().isoformat(),
            'data_period': '2 years',
            'total_samples': len(self.training_data),
            'features_used': len(feature_cols),
            'models_trained': list(results.keys()),
            'model_performance': results
        }
        
        summary_path = self.model_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"🎯 Training completed! {len(results)} models saved.")
        return summary
    
    def load_trained_models(self):
        """Load previously trained models"""
        models = {}
        
        if not self.model_dir.exists():
            return models
        
        for model_file in self.model_dir.glob("gold_prediction_*.pkl"):
            try:
                model_data = joblib.load(model_file)
                horizon = model_file.stem.split('_')[-1]  # Extract "1d", "3d", etc.
                models[horizon] = model_data
                logger.info(f"✅ Loaded {horizon} model (MAE: {model_data['mae']:.2f}%)")
            except Exception as e:
                logger.error(f"❌ Failed to load {model_file}: {e}")
        
        return models
    
    def predict_with_trained_model(self, horizon_days=7):
        """Make prediction using trained models"""
        try:
            # Load models
            models = self.load_trained_models()
            
            if not models:
                logger.info("⚠️ No trained models found. Training new models...")
                training_results = self.train_prediction_models()
                if 'error' in training_results:
                    return training_results
                models = self.load_trained_models()
            
            # Get fresh data for prediction
            fresh_data = self.collect_training_data(period="6mo")
            
            # Find best matching model
            available_horizons = [int(k.replace('d', '')) for k in models.keys() if k.endswith('d')]
            if not available_horizons:
                return {"error": "No trained models available"}
            
            best_horizon = min(available_horizons, key=lambda x: abs(x - horizon_days))
            best_model_key = f"{best_horizon}d"
            
            model_data = models[best_model_key]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_cols = model_data['features']
            
            # Prepare features for prediction
            feature_data = fresh_data[feature_cols].fillna(method='ffill')
            if feature_data.empty:
                return {"error": "No feature data available"}
            
            # Use the most recent data point
            X_current = feature_data.iloc[-1:].values
            X_scaled = scaler.transform(X_current)
            
            # Make prediction
            prediction_pct = model.predict(X_scaled)[0]
            current_price = fresh_data['Close'].iloc[-1]
            predicted_price = current_price * (1 + prediction_pct / 100)
            
            # Calculate confidence based on model performance
            mae = model_data['mae']
            confidence = max(0.3, min(0.95, 1 - (mae / 100)))
            
            return {
                'status': 'success',
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'predicted_change_pct': float(prediction_pct),
                'prediction_horizon': f"{best_horizon} days",
                'model_accuracy': f"{(1 - mae/100)*100:.1f}%",
                'confidence': float(confidence),
                'trained_on': model_data.get('trained_at', 'Unknown'),
                'training_samples': model_data.get('training_samples', 0),
                'top_features': model_data.get('feature_importance', {})
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def get_training_status(self):
        """Get status of trained models"""
        try:
            models = self.load_trained_models()
            
            if not models:
                return {
                    "status": "not_trained",
                    "message": "No trained models found",
                    "recommendation": "Call the training endpoint to train models"
                }
            
            # Load training summary if available
            summary_path = self.model_dir / 'training_summary.json'
            training_info = {}
            
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    training_info = json.load(f)
            
            return {
                "status": "trained",
                "models_available": list(models.keys()),
                "training_completed": training_info.get('training_completed', 'Unknown'),
                "total_samples": training_info.get('total_samples', 0),
                "features_used": training_info.get('features_used', 0),
                "model_details": {
                    horizon: {
                        "accuracy": f"{(1 - data['mae']/100)*100:.1f}%",
                        "mae": f"{data['mae']:.2f}%",
                        "r2_score": f"{data.get('r2', 0):.3f}",
                        "training_samples": data.get('training_samples', 0),
                        "trained_at": data.get('trained_at', 'Unknown')
                    }
                    for horizon, data in models.items()
                }
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Global trainer instance
auto_trainer = AutoMLTrainer()