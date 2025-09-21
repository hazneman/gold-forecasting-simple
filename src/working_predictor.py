"""
Working Gold Price Predictor with Full ML Integration
Combines technical analysis, fundamental economic data, and machine learning
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    ML_AVAILABLE = True
    print("✅ ML libraries loaded successfully")
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML libraries not available")

class WorkingGoldPredictor:
    """Working ML-based gold predictor with technical and fundamental analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.model = None
        self.feature_importance = None
        
    def collect_all_data(self, period='6mo'):
        """Collect both gold price data and economic indicators"""
        
        print(f"📊 Collecting data for {period}...")
        
        # Economic symbols to collect
        symbols = {
            'GOLD': 'GC=F',
            'DXY': 'DX-Y.NYB',
            'VIX': '^VIX',
            'SP500': '^GSPC',
            'TNX': '^TNX',
            'OIL': 'CL=F',
            'SILVER': 'SI=F',
            'BTC': 'BTC-USD'
        }
        
        data = {}
        
        for name, symbol in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    data[name] = hist['Close']
                    print(f"✅ {name}: {len(hist)} data points")
                else:
                    print(f"❌ {name}: No data")
                    
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
                
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df = df.dropna()
        
        print(f"📈 Combined dataset: {df.shape}")
        return df
    
    def create_technical_features(self, gold_prices):
        """Create technical analysis features from gold price data"""
        
        features = pd.DataFrame(index=gold_prices.index)
        
        # Price features
        features['price'] = gold_prices
        features['log_price'] = np.log(gold_prices)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = gold_prices.rolling(window).mean()
            features[f'price_to_sma_{window}'] = gold_prices / features[f'sma_{window}']
        
        # Returns and momentum
        for days in [1, 3, 5, 10]:
            features[f'return_{days}d'] = gold_prices.pct_change(days)
            features[f'momentum_{days}d'] = gold_prices / gold_prices.shift(days) - 1
        
        # Volatility
        features['volatility_20d'] = gold_prices.pct_change().rolling(20).std()
        features['volatility_5d'] = gold_prices.pct_change().rolling(5).std()
        
        # RSI
        delta = gold_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = features['sma_20']
        std_20 = gold_prices.rolling(20).std()
        features['bb_upper'] = sma_20 + (std_20 * 2)
        features['bb_lower'] = sma_20 - (std_20 * 2)
        features['bb_position'] = (gold_prices - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Trend strength
        features['uptrend_strength'] = (
            (features['sma_5'] > features['sma_20']).astype(int) +
            (features['sma_20'] > features['sma_50']).astype(int) +
            (features['price'] > features['sma_20']).astype(int)
        )
        
        return features
    
    def create_fundamental_features(self, market_data):
        """Create fundamental economic features"""
        
        features = pd.DataFrame(index=market_data.index)
        
        # Direct economic indicators
        if 'DXY' in market_data.columns:
            features['dxy'] = market_data['DXY']
            features['dxy_change_1d'] = market_data['DXY'].pct_change(1)
            features['dxy_change_5d'] = market_data['DXY'].pct_change(5)
            
        if 'VIX' in market_data.columns:
            features['vix'] = market_data['VIX']
            features['vix_change_1d'] = market_data['VIX'].pct_change(1)
            features['vix_high'] = (market_data['VIX'] > 25).astype(int)
            
        if 'SP500' in market_data.columns:
            features['sp500'] = market_data['SP500']
            features['sp500_change_1d'] = market_data['SP500'].pct_change(1)
            features['sp500_change_5d'] = market_data['SP500'].pct_change(5)
            
        if 'TNX' in market_data.columns:
            features['treasury_yield'] = market_data['TNX']
            features['treasury_change_1d'] = market_data['TNX'].pct_change(1)
            features['high_yield'] = (market_data['TNX'] > 4.0).astype(int)
            
        if 'OIL' in market_data.columns:
            features['oil'] = market_data['OIL']
            features['oil_change_1d'] = market_data['OIL'].pct_change(1)
            
        if 'SILVER' in market_data.columns:
            features['silver'] = market_data['SILVER']
            features['gold_silver_ratio'] = market_data['GOLD'] / market_data['SILVER']
            
        if 'BTC' in market_data.columns:
            features['btc'] = market_data['BTC']
            features['btc_change_1d'] = market_data['BTC'].pct_change(1)
        
        # Cross-asset correlations (rolling 20-day)
        if 'DXY' in market_data.columns and 'GOLD' in market_data.columns:
            features['gold_dxy_corr'] = market_data['GOLD'].rolling(20).corr(market_data['DXY'])
            
        if 'VIX' in market_data.columns and 'GOLD' in market_data.columns:
            features['gold_vix_corr'] = market_data['GOLD'].rolling(20).corr(market_data['VIX'])
        
        # Market regime features
        if 'SP500' in market_data.columns:
            sp500_sma_50 = market_data['SP500'].rolling(50).mean()
            features['bull_market'] = (market_data['SP500'] > sp500_sma_50).astype(int)
            
        return features
    
    def prepare_training_data(self, market_data, prediction_days=5):
        """Prepare data for ML training"""
        
        gold_prices = market_data['GOLD']
        
        # Create features
        technical_features = self.create_technical_features(gold_prices)
        fundamental_features = self.create_fundamental_features(market_data)
        
        # Combine features
        all_features = pd.concat([technical_features, fundamental_features], axis=1)
        all_features = all_features.dropna()
        
        # Create target (future returns)
        target = gold_prices.shift(-prediction_days) / gold_prices - 1
        target = target.loc[all_features.index]
        
        # Align and clean
        mask = ~(all_features.isna().any(axis=1) | target.isna())
        X = all_features[mask]
        y = target[mask]
        
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def train_model(self, X, y):
        """Train ML model for gold price prediction"""
        
        if not ML_AVAILABLE or len(X) < 50:
            print("⚠️ Using simple correlation model")
            return self._create_simple_model(X, y)
        
        print("🤖 Training ML models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        rf_pred = rf_model.predict(X_test_scaled)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        print(f"✅ Model trained - MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
        
        # Feature importance
        feature_importance = pd.Series(
            rf_model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        self.model = rf_model
        self.feature_importance = feature_importance
        
        return {
            'model_type': 'RandomForest',
            'mae': rf_mae,
            'r2': rf_r2,
            'feature_importance': feature_importance.head(10).to_dict()
        }
    
    def _create_simple_model(self, X, y):
        """Create simple correlation-based model"""
        
        # Calculate feature correlations with target
        correlations = {}
        for col in X.columns:
            try:
                corr = X[col].corr(y)
                if not np.isnan(corr):
                    correlations[col] = corr
            except:
                correlations[col] = 0.0
                
        self.feature_correlations = correlations
        print(f"📊 Simple model with {len(correlations)} features")
        
        return {
            'model_type': 'SimpleCorrelation',
            'correlations': correlations
        }
    
    def predict(self, current_features, days=5):
        """Make price predictions"""
        
        current_price = current_features['price'].iloc[-1] if 'price' in current_features.columns else 0
        
        if self.model is not None and ML_AVAILABLE:
            # ML prediction
            try:
                latest_features = current_features.iloc[-1:].fillna(0)
                
                # Ensure we have the same features as training
                model_features = []
                for col in latest_features.columns:
                    if col in self.scaler.feature_names_in_:
                        model_features.append(latest_features[col].iloc[0])
                    else:
                        model_features.append(0.0)
                
                model_features = np.array(model_features).reshape(1, -1)
                scaled_features = self.scaler.transform(model_features)
                
                predicted_return = self.model.predict(scaled_features)[0]
                predicted_price = current_price * (1 + predicted_return)
                
                confidence = min(0.85, max(0.4, 0.7 - abs(predicted_return) * 5))
                
                return {
                    'predicted_price': predicted_price,
                    'predicted_return': predicted_return * 100,
                    'confidence': confidence,
                    'method': 'ML_RandomForest'
                }
                
            except Exception as e:
                print(f"ML prediction error: {e}")
                
        # Fallback to simple prediction
        if hasattr(self, 'feature_correlations'):
            weighted_signal = 0
            total_weight = 0
            
            for feature, corr in self.feature_correlations.items():
                if feature in current_features.columns:
                    feature_change = current_features[feature].pct_change(1).iloc[-1]
                    if not np.isnan(feature_change):
                        weighted_signal += corr * feature_change
                        total_weight += abs(corr)
            
            if total_weight > 0:
                expected_return = weighted_signal / total_weight * 0.1  # Scale down
            else:
                expected_return = 0.001  # Small positive bias
                
        else:
            expected_return = 0.001
        
        predicted_price = current_price * (1 + expected_return)
        confidence = 0.5
        
        return {
            'predicted_price': predicted_price,
            'predicted_return': expected_return * 100,
            'confidence': confidence,
            'method': 'SimpleCorrelation'
        }
    
    def get_current_signals(self, current_features):
        """Get current technical and fundamental signals"""
        
        signals = {}
        
        if current_features.empty:
            return {'status': 'No data available'}
        
        latest = current_features.iloc[-1]
        
        # Technical signals
        if 'rsi' in latest and not np.isnan(latest['rsi']):
            rsi = latest['rsi']
            if rsi > 70:
                signals['rsi'] = 'OVERBOUGHT - Sell signal'
            elif rsi < 30:
                signals['rsi'] = 'OVERSOLD - Buy signal'  
            else:
                signals['rsi'] = f'NEUTRAL ({rsi:.1f})'
        
        if 'uptrend_strength' in latest:
            strength = latest['uptrend_strength']
            if strength >= 2:
                signals['trend'] = 'STRONG UPTREND - Buy signal'
            elif strength <= 1:
                signals['trend'] = 'DOWNTREND - Sell signal'
            else:
                signals['trend'] = 'SIDEWAYS - Hold'
        
        # Fundamental signals
        if 'dxy_change_1d' in latest and not np.isnan(latest['dxy_change_1d']):
            dxy_change = latest['dxy_change_1d'] * 100
            if dxy_change > 0.5:
                signals['dollar'] = f'USD STRENGTHENING ({dxy_change:+.1f}%) - Negative for gold'
            elif dxy_change < -0.5:
                signals['dollar'] = f'USD WEAKENING ({dxy_change:+.1f}%) - Positive for gold'
            else:
                signals['dollar'] = 'USD STABLE - Neutral'
        
        if 'vix' in latest and not np.isnan(latest['vix']):
            vix = latest['vix']
            if vix > 25:
                signals['fear'] = f'HIGH FEAR (VIX {vix:.1f}) - Positive for gold'
            elif vix < 15:
                signals['fear'] = f'LOW FEAR (VIX {vix:.1f}) - Negative for gold'
            else:
                signals['fear'] = f'MODERATE FEAR (VIX {vix:.1f}) - Neutral'
        
        return signals

def comprehensive_gold_analysis(days=5):
    """Run comprehensive gold price analysis"""
    
    print("🏆 COMPREHENSIVE GOLD PRICE ANALYSIS")
    print("="*50)
    
    # Initialize predictor
    predictor = WorkingGoldPredictor()
    
    # Collect data
    market_data = predictor.collect_all_data('6mo')
    
    if market_data.empty or 'GOLD' not in market_data.columns:
        return {"error": "Unable to collect gold price data"}
    
    current_price = market_data['GOLD'].iloc[-1]
    print(f"\n💰 Current Gold Price: ${current_price:.2f}")
    
    # Prepare training data
    X, y = predictor.prepare_training_data(market_data, days)
    
    if len(X) == 0:
        return {"error": "Insufficient data for analysis"}
    
    # Train model
    training_results = predictor.train_model(X, y)
    
    # Make prediction
    current_features = X.iloc[-1:].copy()
    current_features['price'] = current_price
    
    prediction = predictor.predict(current_features, days)
    
    # Get current signals
    signals = predictor.get_current_signals(X)
    
    # Calculate change
    predicted_price = prediction['predicted_price']
    change_pct = ((predicted_price - current_price) / current_price) * 100
    
    print(f"\n🔮 {days}-Day Prediction:")
    print(f"   Predicted Price: ${predicted_price:.2f}")
    print(f"   Expected Change: {change_pct:+.2f}%")
    print(f"   Confidence: {prediction['confidence']*100:.1f}%")
    print(f"   Method: {prediction['method']}")
    
    print(f"\n📊 Current Market Signals:")
    for signal_type, signal_msg in signals.items():
        print(f"   {signal_type.upper()}: {signal_msg}")
    
    if predictor.feature_importance is not None:
        print(f"\n🎯 Top Influential Factors:")
        for feature, importance in predictor.feature_importance.head(5).items():
            print(f"   {feature}: {importance:.3f}")
    
    return {
        'current_price': current_price,
        'predicted_price': predicted_price,
        'change_percent': change_pct,
        'confidence': prediction['confidence'],
        'method': prediction['method'],
        'signals': signals,
        'training_results': training_results,
        'feature_importance': predictor.feature_importance.to_dict() if predictor.feature_importance is not None else {}
    }