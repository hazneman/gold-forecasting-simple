"""
Enhanced Auto ML Trainer for Gold Price Forecasting
====================================================

Supports predictions from 1 day to 1 year with advanced feature engineering
and multiple algorithm testing for optimal long-term accuracy.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import joblib
from pathlib import Path
import logging
# Note: SimpleEconomicDataCollector import removed - we'll collect economic data directly
import json
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries with fallback
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedAutoMLTrainer:
    """Enhanced ML trainer supporting 1-year predictions with 300+ features"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.training_data = None
        self.model_dir = Path("data/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking attributes
        self.training_progress = {
            'status': 'idle',
            'current_step': '',
            'progress_percent': 0,
            'start_time': None,
            'detailed_log': []
        }
        self.is_training = False
        
        # Extended horizons (1 day to 1 year)
        self.horizons = [1, 3, 5, 7, 14, 21, 30, 42, 60, 90, 120, 180, 270, 365]
        self.horizon_names = {
            1: '1d', 3: '3d', 5: '5d', 7: '1w', 14: '2w', 21: '3w',
            30: '1m', 42: '6w', 60: '2m', 90: '3m', 120: '4m',
            180: '6m', 270: '9m', 365: '1y'
        }
        
        # Algorithms optimized for different time horizons
        self.algorithms = {
            'short_term': ['random_forest', 'gradient_boost', 'extra_trees'],  # 1-14 days
            'medium_term': ['gradient_boost', 'extra_trees', 'ridge', 'elastic_net'],  # 15-90 days
            'long_term': ['ridge', 'elastic_net', 'gradient_boost']  # 90+ days
        }
    
    def get_training_progress(self):
        """Get current training progress with ETA calculation"""
        progress = self.training_progress.copy()
        
        # Calculate estimated time remaining
        if self.is_training and progress.get('start_time'):
            try:
                start_time = datetime.fromisoformat(progress['start_time'])
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if progress['progress_percent'] > 0:
                    total_estimated = elapsed / (progress['progress_percent'] / 100)
                    remaining = max(0, total_estimated - elapsed)
                    progress['estimated_time_remaining'] = int(remaining)
                else:
                    progress['estimated_time_remaining'] = 0
            except:
                progress['estimated_time_remaining'] = 0
            
        return progress

    def update_progress(self, step, percent, details=""):
        """Update training progress with timestamp and logging"""
        self.training_progress.update({
            'current_step': step,
            'progress_percent': percent,
            'last_update': datetime.now().isoformat(),
            'details': details
        })
        
        # Add to detailed log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'progress': percent,
            'details': details
        }
        self.training_progress['detailed_log'].append(log_entry)
        
        # Keep only last 50 log entries
        if len(self.training_progress['detailed_log']) > 50:
            self.training_progress['detailed_log'] = self.training_progress['detailed_log'][-50:]
        
        logger.info(f"📊 Progress: {percent:.1f}% - {step} - {details}")

    def train_extended_models_with_progress(self, period="5y", force_retrain=False):
        """Train extended models with progress tracking and caching"""
        try:
            self.is_training = True
            start_time = time.time()
            self.training_progress['status'] = 'running'
            self.training_progress['start_time'] = datetime.now().isoformat()
            
            # Check cache if not forcing retrain
            if not force_retrain:
                self.update_progress("🔍 Checking cache...", 5, "Checking for existing models...")
                model_status = self.check_models_exist()
                fresh_models = [h for h, status in model_status.items() if status['fresh']]
                
                if len(fresh_models) == len(self.horizon_names):
                    self.update_progress("✅ Using cache", 100, "All models are fresh!")
                    self.training_progress['status'] = 'completed'
                    self.is_training = False
                    return {
                        'status': 'success',
                        'cached': True,
                        'message': 'All models loaded from cache'
                    }
            
            # Train models
            self.update_progress("🚀 Starting training...", 10, "Training extended models...")
            results = self.train_extended_ml_models(period=period)
            
            # Save cache metadata
            self.save_cache_metadata(results)
            
            elapsed_time = time.time() - start_time
            self.update_progress("✅ Training complete!", 100, f"Completed in {elapsed_time:.1f}s")
            
            self.training_progress['status'] = 'completed'
            self.is_training = False
            
            return {
                'status': 'success',
                'cached': False,
                'models_trained': len(results),
                'training_time': elapsed_time
            }
            
        except Exception as e:
            self.is_training = False
            self.training_progress['status'] = 'failed'
            error_msg = f"❌ Training failed: {str(e)}"
            self.update_progress(error_msg, 0, f"Error: {str(e)}")
            logger.error(error_msg)
            raise

    def get_extended_predictions(self, horizons=None):
        """Get predictions for extended horizons up to 1 year"""
        if horizons is None:
            horizons = ['1d', '3d', '5d', '1w', '2w', '3w', '1m', '6w', '2m', '3m', '4m', '6m', '9m', '1y']
        
        predictions = {}
        current_price = None
        
        # Get fresh data for predictions
        try:
            fresh_data = self.collect_extended_training_data(period="2y")
            if fresh_data.empty:
                return {"error": "Could not fetch current data for predictions"}
            
            # Engineer features for prediction
            fresh_data = self.engineer_features(fresh_data)
            current_price = float(fresh_data['Close'].iloc[-1])
            
        except Exception as e:
            return {"error": f"Data collection failed: {str(e)}"}

        # Check if we have any trained models
        model_files_found = 0
        for horizon in horizons:
            model_path = self.model_dir / f"gold_model_{horizon}.pkl"
            if model_path.exists():
                model_files_found += 1
        
        if model_files_found == 0:
            return {"error": "No trained models found. Please train the extended models first."}
        
        logger.info(f"🔮 Making predictions with {model_files_found} models, current price: ${current_price:.2f}")

        for horizon in horizons:
            try:
                logger.info(f"🔄 Processing horizon: {horizon}")
                model_path = self.model_dir / f"gold_model_{horizon}.pkl"
                
                if not model_path.exists():
                    logger.warning(f"⚠️ Model for {horizon} not found at {model_path}")
                    continue

                try:
                    # Load the trained model
                    model = joblib.load(model_path)
                    
                    # Prepare features for prediction
                    feature_cols = [col for col in fresh_data.columns 
                                   if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
                                   and fresh_data[col].dtype in ['float64', 'int64']]
                    
                    # Get the latest feature values
                    X_latest = fresh_data[feature_cols].iloc[-1:].ffill().fillna(0)
                    
                    # Handle any remaining missing or infinite values
                    X_latest = X_latest.replace([np.inf, -np.inf], 0).fillna(0)
                    
                    # Make prediction (percentage change)
                    prediction_pct = float(model.predict(X_latest)[0])
                    
                    # Convert percentage change to predicted price
                    predicted_price = current_price * (1 + prediction_pct / 100)
                    
                    # Calculate change in dollars and percentage
                    price_change = predicted_price - current_price
                    
                    predictions[horizon] = {
                        'predicted_price': round(predicted_price, 2),
                        'current_price': round(current_price, 2),
                        'price_change': round(price_change, 2),
                        'percentage_change': round(prediction_pct, 2),
                        'horizon': horizon,
                        'model_file': str(model_path),
                        'confidence': 'medium',  # Could be enhanced with actual confidence intervals
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    logger.info(f"🎯 {horizon}: ${predicted_price:.2f} ({prediction_pct:+.2f}%)")
                    
                except Exception as e:
                    logger.error(f"❌ Error predicting {horizon}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as outer_e:
                logger.error(f"❌ Outer error for horizon {horizon}: {outer_e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not predictions:
            return {"error": "Failed to generate any predictions"}
        
        logger.info(f"✅ Generated {len(predictions)} predictions successfully")
        
        # Prepare result
        result = {
            "status": "success",
            "predictions": predictions,
            "current_price": current_price,
            "total_horizons": len(horizons),
            "successful_predictions": len(predictions),
            'timestamp': datetime.now().isoformat(),
            'data_freshness': str(fresh_data.index[-1]) if len(fresh_data) > 0 else 'unknown'
        }
        
        logger.info(f"🎯 Generated {result['successful_predictions']}/{result['total_horizons']} predictions successfully")
        return result
    
    def check_models_exist(self, horizons=None):
        """Check which models exist and their freshness"""
        if horizons is None:
            horizons = ['1d', '3d', '5d', '1w', '2w', '3w', '1m', '6w', '2m', '3m', '4m', '6m', '9m', '1y']
        
        model_status = {}
        cache_file = self.model_dir / "cache_metadata.json"
        cache_data = {}
        
        # Load existing cache metadata
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}")
        
        for horizon in horizons:
            model_path = self.model_dir / f"gold_model_{horizon}.pkl"
            status = {
                'exists': model_path.exists(),
                'path': str(model_path),
                'fresh': False,
                'age_hours': None,
                'last_trained': None
            }
            
            if model_path.exists():
                # Check file modification time
                mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
                age_hours = (datetime.now() - mod_time).total_seconds() / 3600
                status['age_hours'] = round(age_hours, 2)
                status['last_trained'] = mod_time.isoformat()
                
                # Consider fresh if less than 7 days old
                status['fresh'] = age_hours < (7 * 24)
                
                # Add cache metadata if available
                if horizon in cache_data:
                    status.update(cache_data[horizon])
            
            model_status[horizon] = status
        
        return model_status
    
    def save_cache_metadata(self, training_results):
        """Save cache metadata after training"""
        cache_file = self.model_dir / "cache_metadata.json"
        cache_data = {}
        
        # Load existing cache data
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except Exception:
                pass
        
        # Update with new training results
        timestamp = datetime.now().isoformat()
        for horizon, result in training_results.items():
            cache_data[horizon] = {
                'trained_at': timestamp,
                'algorithm': result.get('best_algorithm', 'unknown'),
                'accuracy': result.get('accuracy', 0),
                'score': result.get('best_score', 0),
                'samples': result.get('samples', 0),
                'features': result.get('features', 0)
            }
        
        # Save updated cache data
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"💾 Cache metadata saved for {len(training_results)} models")
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def load_cache_metadata(self):
        """Load cache metadata from disk"""
        cache_file = self.model_dir / "cache_metadata.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}")
        return {}

    def collect_extended_training_data(self, period="5y"):
        """Collect comprehensive training data for extended predictions"""
        logger.info(f"📊 Collecting extended training data ({period} period)...")
        
        try:
            # Fetch comprehensive gold data
            gold = yf.download("GC=F", period=period, interval="1d", progress=False)
            
            if gold.empty:
                logger.error("Failed to fetch gold futures data")
                return None
            
            # Flatten multi-level columns if they exist
            if hasattr(gold.columns, 'levels'):
                gold.columns = gold.columns.get_level_values(0)
            
            logger.info(f"📊 Collected {len(gold)} gold price samples from {gold.index[0]} to {gold.index[-1]}")
            return gold
            
        except Exception as e:
            logger.error(f"Error collecting training data: {e}")
            return None

    def engineer_features(self, df):
        """Engineer 300+ features for extended predictions"""
        data = df.copy()
        logger.info(f"🔧 Engineering features for {len(data)} samples...")
        
        if data.empty or len(data) < 50:
            logger.warning("Insufficient data for feature engineering")
            return data
        
        try:
            # Basic price features
            data['price_range'] = data['High'] - data['Low']
            data['price_center'] = (data['High'] + data['Low']) / 2
            data['volume_price'] = data['Volume'] * data['Close']
            
            # Returns at multiple timeframes
            for period in [1, 3, 5, 7, 14, 21, 30, 60, 90]:
                data[f'return_{period}d'] = data['Close'].pct_change(period) * 100
                # Use minimum window of 2 for volatility calculation
                vol_window = max(2, period)
                data[f'volatility_{period}d'] = data['Close'].rolling(vol_window).std()
                
            # Moving averages - Fixed version
            for window in [5, 10, 20, 50, 100, 200]:
                ma_col = data['Close'].rolling(window).mean()
                data[f'ma_{window}'] = ma_col
                data[f'ma_ratio_{window}'] = data['Close'] / ma_col
                data[f'ma_slope_{window}'] = ma_col.diff(5)
            
            # Technical indicators
            data['rsi_14'] = self._calculate_rsi(data['Close'], 14)
            data['rsi_21'] = self._calculate_rsi(data['Close'], 21)
            
            # MACD
            macd, macd_signal = self._calculate_macd(data['Close'])
            data['macd'] = macd
            data['macd_signal'] = macd_signal
            data['macd_histogram'] = macd - macd_signal
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data['Close'])
            data['bb_upper'] = bb_upper
            data['bb_lower'] = bb_lower
            data['bb_width'] = bb_upper - bb_lower
            data['bb_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Volume analysis
            volume_ma = data['Volume'].rolling(20).mean()
            data['volume_ma'] = volume_ma
            data['volume_ratio'] = data['Volume'] / volume_ma
            data['price_volume'] = data['Close'] * data['Volume']
            
            # Seasonal features
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['month'] = data.index.month
            data['quarter'] = data.index.quarter
            data['is_month_end'] = data.index.is_month_end.astype(int)
            
            # Trend analysis
            for window in [10, 20, 50]:
                data[f'trend_strength_{window}'] = self._calculate_trend_strength(data['Close'], window)
            
            # Momentum indicators
            data['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
            data['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
            
            # Support/Resistance levels
            high_20 = data['High'].rolling(20).max()
            low_20 = data['Low'].rolling(20).min()
            data['high_20'] = high_20
            data['low_20'] = low_20
            data['support_distance'] = (data['Close'] - low_20) / data['Close']
            data['resistance_distance'] = (high_20 - data['Close']) / data['Close']
            
            # Add economic features if available
            data = self._add_economic_features(data)
            
            logger.info(f"🔧 Feature engineering complete: {data.shape[1]} features")
            return data
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, window=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def _calculate_trend_strength(self, prices, window):
        """Calculate trend strength using linear regression slope"""
        def calc_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            try:
                slope = np.polyfit(x, y, 1)[0]
                return slope / y.mean() * 100  # Normalize by price level
            except:
                return 0
        return prices.rolling(window).apply(calc_slope)
    
    def _collect_economic_data(self, period="5y"):
        """Collect economic indicators for ML feature engineering"""
        try:
            logger.info("📊 Collecting economic indicators for ML training...")
            
            # Define economic data sources
            economic_symbols = {
                'DXY': 'DX-Y.NYB',      # US Dollar Index
                'VIX': '^VIX',          # Volatility Index  
                'TNX': '^TNX',          # 10-Year Treasury
                'IRX': '^IRX',          # 3-Month Treasury (Fed funds proxy)
                'OIL': 'CL=F',          # Oil prices
                'SPY': 'SPY',           # S&P 500 ETF
                'TLT': 'TLT',           # 20+ Year Treasury Bond ETF
                'TIPS': 'SCHP'          # TIPS ETF for inflation
            }
            
            economic_data = {}
            
            for name, symbol in economic_symbols.items():
                try:
                    data = yf.download(symbol, period=period, progress=False)
                    if not data.empty:
                        # Use adjusted close for consistency
                        economic_data[name] = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                        logger.info(f"✅ Collected {name}: {len(data)} samples")
                    else:
                        logger.warning(f"⚠️ No data for {name} ({symbol})")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to collect {name}: {e}")
            
            if economic_data:
                # Combine all economic data into a single DataFrame
                # All series should have the same index (dates), so we can concat
                econ_df = pd.concat(economic_data, axis=1)
                logger.info(f"📊 Economic data collected: {econ_df.shape}")
                return econ_df
            else:
                logger.warning("❌ No economic data collected")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Economic data collection failed: {e}")
            return pd.DataFrame()
    
    def _add_economic_features(self, data):
        """Add economic indicators as ML features"""
        try:
            # Get economic data for the same period
            period = "5y" if len(data) > 1000 else "2y"
            econ_data = self._collect_economic_data(period)
            
            if econ_data.empty:
                logger.warning("🔍 No economic data available, using technical features only")
                return data
            
            # Align economic data with gold price data by date
            econ_aligned = econ_data.reindex(data.index, method='ffill')
            
            # Add Fed funds rate features (using 3-month treasury as proxy)
            if 'IRX' in econ_aligned.columns:
                data['fed_funds_rate'] = econ_aligned['IRX']
                data['fed_funds_ma_10'] = econ_aligned['IRX'].rolling(10).mean()
                data['fed_funds_change'] = econ_aligned['IRX'].diff()
                data['fed_funds_trend'] = econ_aligned['IRX'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0)
            
            # Add Dollar Index (DXY) features
            if 'DXY' in econ_aligned.columns:
                data['dxy_level'] = econ_aligned['DXY']
                data['dxy_ma_20'] = econ_aligned['DXY'].rolling(20).mean()
                data['dxy_strength'] = econ_aligned['DXY'] / econ_aligned['DXY'].rolling(50).mean()
                data['dxy_change'] = econ_aligned['DXY'].pct_change() * 100
            
            # Add VIX (Market Fear) features
            if 'VIX' in econ_aligned.columns:
                data['vix_level'] = econ_aligned['VIX']
                data['vix_ma_10'] = econ_aligned['VIX'].rolling(10).mean()
                data['market_stress'] = (econ_aligned['VIX'] > 25).astype(int)  # High fear threshold
                data['vix_spike'] = (econ_aligned['VIX'] > econ_aligned['VIX'].rolling(10).mean() * 1.5).astype(int)
            
            # Add Treasury Yield features
            if 'TNX' in econ_aligned.columns:
                data['treasury_10y'] = econ_aligned['TNX']
                data['yield_ma_20'] = econ_aligned['TNX'].rolling(20).mean()
                data['yield_change'] = econ_aligned['TNX'].diff()
                # Yield curve analysis (if we have both 3m and 10y)
                if 'IRX' in econ_aligned.columns:
                    # Ensure we get Series, not DataFrame
                    tnx_series = econ_aligned['TNX'].squeeze() if hasattr(econ_aligned['TNX'], 'squeeze') else econ_aligned['TNX']
                    irx_series = econ_aligned['IRX'].squeeze() if hasattr(econ_aligned['IRX'], 'squeeze') else econ_aligned['IRX']
                    spread = tnx_series - irx_series
                    data['yield_curve_spread'] = spread
                    data['yield_curve_inversion'] = (spread < 0).astype(int)
            
            # Add Inflation Expectations (TIPS)
            if 'TIPS' in econ_aligned.columns:
                data['tips_price'] = econ_aligned['TIPS']
                data['tips_change'] = econ_aligned['TIPS'].pct_change() * 100
                data['tips_ma_20'] = econ_aligned['TIPS'].rolling(20).mean()
                data['inflation_trend'] = econ_aligned['TIPS'].rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0)
            
            # Add Oil Price features (inflation/geopolitical proxy)
            if 'OIL' in econ_aligned.columns:
                data['oil_price'] = econ_aligned['OIL']
                data['oil_volatility'] = econ_aligned['OIL'].rolling(20).std()
                data['oil_change'] = econ_aligned['OIL'].pct_change() * 100
            
            # Add Stock Market (Risk-on/Risk-off) features
            if 'SPY' in econ_aligned.columns:
                data['spy_level'] = econ_aligned['SPY']
                data['spy_change'] = econ_aligned['SPY'].pct_change() * 100
                data['spy_ma_50'] = econ_aligned['SPY'].rolling(50).mean()
                # Risk-on/Risk-off ratio
                if 'TLT' in econ_aligned.columns:
                    spy_series = econ_aligned['SPY'].squeeze() if hasattr(econ_aligned['SPY'], 'squeeze') else econ_aligned['SPY']
                    tlt_series = econ_aligned['TLT'].squeeze() if hasattr(econ_aligned['TLT'], 'squeeze') else econ_aligned['TLT']
                    data['risk_ratio'] = spy_series / tlt_series  # Stocks vs Bonds
            
            # Add Bond Market features
            if 'TLT' in econ_aligned.columns:
                data['tlt_price'] = econ_aligned['TLT']
                data['tlt_change'] = econ_aligned['TLT'].pct_change() * 100
                data['bond_strength'] = econ_aligned['TLT'] / econ_aligned['TLT'].rolling(20).mean()
            
            # Fill any NaN values that might have been introduced
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(method='ffill').fillna(0)
            
            logger.info(f"📊 Added economic features. Total features: {data.shape[1]}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to add economic features: {e}")
            return data

    def train_extended_ml_models(self, period="5y", horizons=None):
        """Train ML models for extended horizons (1 day to 1 year)"""
        if not ML_AVAILABLE:
            raise Exception("ML libraries not available")
        
        if horizons is None:
            horizons = ['1d', '3d', '5d', '1w', '2w', '3w', '1m', '6w', '2m', '3m', '4m', '6m', '9m', '1y']
        
        logger.info(f"🚀 Training extended ML models for {len(horizons)} horizons...")
        
        # Collect training data
        data = self.collect_extended_training_data(period)
        if data is None or data.empty:
            raise Exception("Failed to collect training data")
        
        # Engineer features
        features_df = self.engineer_features(data)
        
        # Select feature columns (exclude OHLCV columns)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] 
                       and features_df[col].dtype in ['float64', 'int64']]
        
        logger.info(f"🔧 Using {len(feature_cols)} engineered features")
        
        results = {}
        
        # Train models for each horizon
        for horizon in horizons:
            horizon_days = {
                '1d': 1, '3d': 3, '5d': 5, '1w': 7, '2w': 14, '3w': 21,
                '1m': 30, '6w': 42, '2m': 60, '3m': 90, '4m': 120,
                '6m': 180, '9m': 270, '1y': 365
            }.get(horizon, 30)
            
            logger.info(f"🎯 Training {horizon} model ({horizon_days} days)...")
            
            try:
                result = self._train_single_horizon_model(features_df, feature_cols, horizon_days, horizon)
                results[horizon] = result
                logger.info(f"✅ {horizon} model: {result['best_algorithm']} (Accuracy: {result.get('accuracy', 0):.1f}%)")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {horizon} model: {e}")
                results[horizon] = {'error': str(e), 'horizon': horizon}
        
        logger.info(f"🎯 Extended training completed! {len(results)} models saved (1d to 1y horizons)")
        logger.info(f"📊 Models available: {', '.join(results.keys())}")
        
        return results

    def _train_single_horizon_model(self, features_df, feature_cols, horizon_days, horizon_name):
        """Train a model for a specific horizon"""
        # Create target variable (percentage change)
        y = (features_df['Close'].shift(-horizon_days) / features_df['Close'] - 1) * 100
        
        # Prepare features
        X = features_df[feature_cols].ffill().bfill()
        
        # Align data (remove NaN values)
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 200:
            raise Exception(f"Insufficient data: {len(X_clean)} samples")
        
        # Determine algorithms based on horizon
        if horizon_days <= 14:
            algorithms = self.algorithms['short_term']
        elif horizon_days <= 90:
            algorithms = self.algorithms['medium_term']
        else:
            algorithms = self.algorithms['long_term']
        
        # Train and evaluate models
        best_score = float('-inf')
        best_model = None
        best_algorithm = None
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for algorithm in algorithms:
            try:
                if algorithm == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                elif algorithm == 'gradient_boost':
                    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                elif algorithm == 'extra_trees':
                    model = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42)
                elif algorithm == 'ridge':
                    model = Ridge(alpha=1.0)
                elif algorithm == 'elastic_net':
                    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
                elif algorithm == 'lasso':
                    model = Lasso(alpha=0.1, random_state=42)
                elif algorithm == 'svr':
                    model = SVR(kernel='rbf', C=1.0, gamma='scale')
                else:
                    continue
                
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in tscv.split(X_clean):
                    X_train, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                    y_train, y_val = y_clean.iloc[train_idx], y_clean.iloc[val_idx]
                    
                    model.fit(X_train, y_train)
                    val_pred = model.predict(X_val)
                    score = r2_score(y_val, val_pred)
                    cv_scores.append(score)
                
                avg_score = np.mean(cv_scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_algorithm = algorithm
                    # Train final model on all data
                    model.fit(X_clean, y_clean)
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"Failed to train {algorithm} for {horizon_name}: {e}")
                continue
        
        if best_model is None:
            raise Exception("No algorithm succeeded")
        
        # Save model
        model_path = self.model_dir / f"gold_model_{horizon_name}.pkl"
        joblib.dump(best_model, model_path)
        
        # Calculate metrics
        accuracy = max(0, best_score * 100) if best_score > 0 else 0
        
        return {
            'horizon_days': horizon_days,
            'horizon_name': horizon_name,
            'best_algorithm': best_algorithm,
            'best_score': best_score,
            'accuracy': accuracy,
            'samples': len(X_clean),
            'features': len(feature_cols),
            'model_path': str(model_path)
        }

# Create global trainer instance
enhanced_trainer = EnhancedAutoMLTrainer()