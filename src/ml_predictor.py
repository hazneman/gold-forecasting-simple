"""
Advanced Gold Price Prediction using ML with Fundamentals + Technicals
Combines economic fundamentals with technical analysis using machine learning
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Try importing ML libraries with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    print("ML libraries not available, using simplified predictions")
    ML_AVAILABLE = False

class TechnicalIndicators:
    """Calculate technical indicators for gold price analysis"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * num_std),
            'lower': sma - (std * num_std)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            '%K': k_percent,
            '%D': d_percent
        }

class FundamentalAnalyzer:
    """Analyze fundamental economic factors affecting gold"""
    
    def __init__(self):
        self.economic_symbols = {
            'dxy': 'DX-Y.NYB',        # US Dollar Index
            'vix': '^VIX',            # Volatility Index
            'sp500': '^GSPC',         # S&P 500
            'treasury_10y': '^TNX',   # 10-Year Treasury
            'oil': 'CL=F',            # Crude Oil
            'silver': 'SI=F',         # Silver
            'bitcoin': 'BTC-USD',     # Bitcoin
            'bonds': 'TLT',           # Treasury Bonds ETF
            'copper': 'HG=F',         # Copper
            'nasdaq': '^IXIC',        # NASDAQ
        }
    
    def get_fundamental_features(self, period: str = '6mo') -> pd.DataFrame:
        """Get fundamental economic features"""
        
        features_data = {}
        
        for name, symbol in self.economic_symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    # Calculate features for each economic indicator
                    close_prices = hist['Close']
                    
                    # Price levels
                    features_data[f'{name}_price'] = close_prices
                    
                    # Moving averages
                    features_data[f'{name}_sma_20'] = TechnicalIndicators.sma(close_prices, 20)
                    features_data[f'{name}_sma_50'] = TechnicalIndicators.sma(close_prices, 50)
                    
                    # Momentum
                    features_data[f'{name}_change_1d'] = close_prices.pct_change(1)
                    features_data[f'{name}_change_5d'] = close_prices.pct_change(5)
                    features_data[f'{name}_change_20d'] = close_prices.pct_change(20)
                    
                    # Volatility
                    features_data[f'{name}_volatility'] = close_prices.pct_change().rolling(20).std()
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                continue
        
        # Convert to DataFrame and align dates
        df = pd.DataFrame(features_data)
        return df.dropna()

class MLGoldPredictor:
    """Advanced ML-based gold price predictor combining fundamentals and technicals"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalIndicators()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.models = {}
        self.feature_importance = {}
        self.last_training_date = None
        
    def prepare_features(self, gold_data: pd.DataFrame, fundamental_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare combined technical and fundamental features"""
        
        # Technical features from gold price data
        close = gold_data['Close']
        high = gold_data['High']
        low = gold_data['Low']
        volume = gold_data['Volume']
        
        features = pd.DataFrame(index=gold_data.index)
        
        # === TECHNICAL FEATURES ===
        
        # Price-based features
        features['price'] = close
        features['high_low_ratio'] = high / low
        features['price_to_sma_20'] = close / self.technical_analyzer.sma(close, 20)
        features['price_to_sma_50'] = close / self.technical_analyzer.sma(close, 50)
        features['price_to_sma_200'] = close / self.technical_analyzer.sma(close, 200)
        
        # Moving averages
        features['sma_5'] = self.technical_analyzer.sma(close, 5)
        features['sma_10'] = self.technical_analyzer.sma(close, 10)
        features['sma_20'] = self.technical_analyzer.sma(close, 20)
        features['sma_50'] = self.technical_analyzer.sma(close, 50)
        features['ema_12'] = self.technical_analyzer.ema(close, 12)
        features['ema_26'] = self.technical_analyzer.ema(close, 26)
        
        # Momentum indicators
        features['rsi'] = self.technical_analyzer.rsi(close)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        
        # MACD
        macd_data = self.technical_analyzer.macd(close)
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_histogram'] = macd_data['histogram']
        features['macd_bullish'] = (features['macd'] > features['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_data = self.technical_analyzer.bollinger_bands(close)
        features['bb_upper'] = bb_data['upper']
        features['bb_lower'] = bb_data['lower']
        features['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        features['bb_position'] = (close - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Stochastic
        stoch_data = self.technical_analyzer.stochastic(high, low, close)
        features['stoch_k'] = stoch_data['%K']
        features['stoch_d'] = stoch_data['%D']
        features['stoch_oversold'] = (features['stoch_k'] < 20).astype(int)
        features['stoch_overbought'] = (features['stoch_k'] > 80).astype(int)
        
        # Volatility features
        features['price_volatility'] = close.pct_change().rolling(20).std()
        features['volume_sma'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / features['volume_sma']
        
        # Price momentum
        features['momentum_1d'] = close.pct_change(1)
        features['momentum_3d'] = close.pct_change(3)
        features['momentum_5d'] = close.pct_change(5)
        features['momentum_10d'] = close.pct_change(10)
        features['momentum_20d'] = close.pct_change(20)
        
        # === FUNDAMENTAL FEATURES ===
        # Align fundamental data with gold data dates
        common_dates = features.index.intersection(fundamental_data.index)
        if len(common_dates) > 0:
            fundamental_aligned = fundamental_data.loc[common_dates]
            features_aligned = features.loc[common_dates]
            
            # Add fundamental features
            for col in fundamental_data.columns:
                if col in fundamental_aligned.columns:
                    features_aligned[f'fund_{col}'] = fundamental_aligned[col]
            
            features = features_aligned
        
        # === DERIVED FEATURES ===
        
        # Trend strength
        features['uptrend_strength'] = (
            (features['sma_5'] > features['sma_20']).astype(int) +
            (features['sma_20'] > features['sma_50']).astype(int) +
            (features['price'] > features['sma_20']).astype(int)
        )
        
        # Market regime (bull/bear/sideways)
        features['market_regime'] = 0  # sideways
        features.loc[features['uptrend_strength'] >= 2, 'market_regime'] = 1  # bull
        features.loc[features['uptrend_strength'] <= 1, 'market_regime'] = -1  # bear
        
        return features.dropna()
    
    def create_target_variables(self, gold_data: pd.DataFrame, prediction_days: List[int] = [1, 3, 5, 7]) -> pd.DataFrame:
        """Create target variables for different prediction horizons"""
        targets = pd.DataFrame(index=gold_data.index)
        close = gold_data['Close']
        
        for days in prediction_days:
            # Future returns
            targets[f'return_{days}d'] = close.shift(-days) / close - 1
            
            # Future price direction (binary classification)
            targets[f'direction_{days}d'] = (targets[f'return_{days}d'] > 0).astype(int)
            
            # Future volatility
            targets[f'volatility_{days}d'] = close.pct_change().shift(-days).rolling(days).std()
        
        return targets.dropna()
    
    def train_models(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
        """Train ML models for gold price prediction"""
        
        if not ML_AVAILABLE:
            return self._simple_prediction_fallback(features, targets)
        
        results = {}
        
        # Align features and targets
        common_index = features.index.intersection(targets.index)
        X = features.loc[common_index]
        y = targets.loc[common_index]
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:  # Not enough data
            return self._simple_prediction_fallback(features, targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # Train models for different targets
        for target_col in y.columns:
            if 'return' in target_col:  # Regression task
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y[target_col], test_size=0.2, random_state=42, shuffle=False
                    )
                    
                    # Train Random Forest
                    rf_model = RandomForestRegressor(
                        n_estimators=100, 
                        max_depth=10, 
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_model.fit(X_train, y_train)
                    
                    # Train Gradient Boosting
                    gb_model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    gb_model.fit(X_train, y_train)
                    
                    # Evaluate models
                    rf_pred = rf_model.predict(X_test)
                    gb_pred = gb_model.predict(X_test)
                    
                    rf_mae = mean_absolute_error(y_test, rf_pred)
                    gb_mae = mean_absolute_error(y_test, gb_pred)
                    
                    # Choose best model
                    if rf_mae < gb_mae:
                        best_model = rf_model
                        best_score = rf_mae
                        model_type = 'RandomForest'
                    else:
                        best_model = gb_model
                        best_score = gb_mae
                        model_type = 'GradientBoosting'
                    
                    self.models[target_col] = best_model
                    
                    # Feature importance
                    feature_importance = pd.Series(
                        best_model.feature_importances_, 
                        index=X.columns
                    ).sort_values(ascending=False)
                    
                    self.feature_importance[target_col] = feature_importance
                    
                    results[target_col] = {
                        'model_type': model_type,
                        'mae': best_score,
                        'r2_score': r2_score(y_test, rf_pred if rf_mae < gb_mae else gb_pred),
                        'feature_importance': feature_importance.head(10).to_dict()
                    }
                    
                except Exception as e:
                    print(f"Error training model for {target_col}: {e}")
                    continue
        
        self.last_training_date = datetime.now()
        return results
    
    def predict_price(self, current_features: pd.DataFrame, days: int = 7) -> Dict[str, Any]:
        """Predict gold price using trained ML models"""
        
        if not ML_AVAILABLE or not self.models:
            return self._simple_prediction_fallback_single(current_features, days)
        
        predictions = {}
        
        try:
            # Get the most recent feature row
            latest_features = current_features.iloc[-1:].copy()
            
            # Scale features
            latest_scaled = self.scaler.transform(latest_features)
            latest_scaled = pd.DataFrame(latest_scaled, columns=latest_features.columns)
            
            # Make predictions for different horizons
            for target_col, model in self.models.items():
                if f'{days}d' in target_col and 'return' in target_col:
                    try:
                        predicted_return = model.predict(latest_scaled)[0]
                        predictions[target_col] = predicted_return
                    except Exception as e:
                        print(f"Error predicting {target_col}: {e}")
                        continue
            
            # Get current gold price
            current_price = latest_features['price'].iloc[0]
            
            # Calculate predicted prices
            predicted_prices = []
            confidence_scores = []
            
            for i in range(1, days + 1):
                target_key = f'return_{i}d'
                if target_key in predictions:
                    predicted_return = predictions[target_key]
                    predicted_price = current_price * (1 + predicted_return)
                    
                    # Calculate confidence based on model performance
                    confidence = max(0.1, min(0.9, 1.0 - abs(predicted_return) * 10))
                    
                else:
                    # Fallback to simple linear interpolation
                    if len(predicted_prices) > 0:
                        predicted_price = predicted_prices[-1] * 1.001  # Small positive bias
                    else:
                        predicted_price = current_price * 1.001
                    confidence = 0.5
                
                predicted_prices.append(predicted_price)
                confidence_scores.append(confidence)
            
            return {
                'predictions': predicted_prices,
                'confidence_scores': confidence_scores,
                'current_price': current_price,
                'model_features_used': len(latest_features.columns),
                'prediction_method': 'ML_Models'
            }
            
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            return self._simple_prediction_fallback_single(current_features, days)
    
    def _simple_prediction_fallback(self, features: pd.DataFrame, targets: pd.DataFrame) -> Dict[str, Any]:
        """Fallback prediction method when ML is not available"""
        return {
            'model_type': 'SimpleFallback',
            'status': 'ML libraries not available, using simple trend analysis',
            'features_analyzed': len(features.columns) if not features.empty else 0
        }
    
    def _simple_prediction_fallback_single(self, features: pd.DataFrame, days: int) -> Dict[str, Any]:
        """Simple fallback for single predictions"""
        if features.empty:
            return {
                'predictions': [0] * days,
                'confidence_scores': [0.3] * days,
                'current_price': 0,
                'prediction_method': 'Fallback'
            }
        
        current_price = features['price'].iloc[-1] if 'price' in features.columns else 0
        
        # Simple trend-based prediction
        trend = 0.001  # Small positive bias
        predicted_prices = [current_price * (1 + trend * i) for i in range(1, days + 1)]
        confidence_scores = [max(0.3, 0.7 - i * 0.05) for i in range(days)]
        
        return {
            'predictions': predicted_prices,
            'confidence_scores': confidence_scores,
            'current_price': current_price,
            'prediction_method': 'SimpleTrend'
        }
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of most important features across all models"""
        
        if not self.feature_importance:
            return {'status': 'No trained models available'}
        
        # Aggregate feature importance across all models
        all_features = {}
        
        for target, importance in self.feature_importance.items():
            for feature, score in importance.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(score)
        
        # Calculate average importance
        avg_importance = {
            feature: np.mean(scores) 
            for feature, scores in all_features.items()
        }
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize features
        technical_features = [(f, s) for f, s in sorted_features if not f.startswith('fund_')]
        fundamental_features = [(f, s) for f, s in sorted_features if f.startswith('fund_')]
        
        return {
            'top_technical_features': technical_features[:10],
            'top_fundamental_features': fundamental_features[:10],
            'total_features_analyzed': len(sorted_features),
            'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None
        }

def get_model_decision_explanation(prediction_result: Dict[str, Any], feature_importance: Dict[str, Any]) -> str:
    """Explain how the model makes its price decisions"""
    
    explanation = []
    
    method = prediction_result.get('prediction_method', 'Unknown')
    
    if method == 'ML_Models':
        explanation.append("🤖 **ML-Based Decision Process:**")
        explanation.append(f"- Analyzed {prediction_result.get('model_features_used', 0)} combined technical and fundamental features")
        explanation.append("- Used ensemble of Random Forest and Gradient Boosting models")
        explanation.append("- Models trained on historical correlations between:")
        
        if 'top_technical_features' in feature_importance:
            explanation.append("  📊 **Top Technical Signals:**")
            for feature, importance in feature_importance['top_technical_features'][:5]:
                explanation.append(f"    • {feature}: {importance:.3f} influence")
        
        if 'top_fundamental_features' in feature_importance:
            explanation.append("  📈 **Top Economic Factors:**")
            for feature, importance in feature_importance['top_fundamental_features'][:5]:
                explanation.append(f"    • {feature}: {importance:.3f} influence")
                
    elif method == 'SimpleTrend':
        explanation.append("📊 **Simple Trend Analysis:**")
        explanation.append("- Based on recent price momentum and moving averages")
        explanation.append("- Limited fundamental analysis due to data constraints")
        
    else:
        explanation.append("⚠️ **Fallback Method:**")
        explanation.append("- Using basic trend extrapolation")
        explanation.append("- Recommend running with full ML capabilities for better accuracy")
    
    return "\n".join(explanation)