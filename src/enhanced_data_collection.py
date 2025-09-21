"""
Enhanced Economic Data Collection for Gold Price Forecasting
Includes comprehensive economic indicators that influence gold prices
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

class EnhancedEconomicDataCollector:
    """
    Comprehensive economic data collector for gold price forecasting
    Includes all major economic indicators that influence gold prices
    """
    
    def __init__(self):
        self.data_sources = {
            'gold': 'GLD',  # Gold ETF
            'dxy': 'DX-Y.NYB',  # US Dollar Index
            'tips_10y': '^TNX',  # 10-Year Treasury (proxy)
            'vix': '^VIX',  # Volatility Index
            'sp500': '^GSPC',  # S&P 500
            'crude_oil': 'CL=F',  # Crude Oil (geopolitical proxy)
            'silver': 'SLV',  # Silver (precious metals correlation)
            'bonds': 'TLT',  # 20+ Year Treasury Bond ETF
            'real_estate': 'VNQ',  # Real Estate (inflation hedge)
            'bitcoin': 'BTC-USD',  # Bitcoin (risk asset correlation)
            'copper': 'HG=F'  # Copper (industrial demand)
        }
        
        # FRED API indicators (requires API key for real implementation)
        self.fred_indicators = {
            'cpi': 'CPIAUCSL',  # Consumer Price Index
            'pce': 'PCE',  # Personal Consumption Expenditures
            'real_rates': 'DFII10',  # 10-Year TIPS
            'fed_funds': 'FEDFUNDS',  # Federal Funds Rate
            'money_supply': 'M2SL',  # M2 Money Supply
            'inflation_expectations': 'T5YIE'  # 5-Year Inflation Expectations
        }
    
    def collect_market_data(self, period: str = "2y") -> pd.DataFrame:
        """Collect comprehensive market data"""
        print("🔄 Collecting comprehensive economic data...")
        
        all_data = {}
        
        # Collect market data from Yahoo Finance
        for name, ticker in self.data_sources.items():
            try:
                data = yf.download(ticker, period=period, interval="1d", progress=False)
                if not data.empty:
                    all_data[f'{name}_close'] = data['Close']
                    all_data[f'{name}_volume'] = data['Volume'] if 'Volume' in data.columns else None
                    print(f"✅ {name}: {len(data)} days")
                else:
                    print(f"⚠️ {name}: No data available")
            except Exception as e:
                print(f"❌ {name}: {str(e)}")
        
        # Combine all data
        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        
        return df.dropna()
    
    def calculate_economic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived economic indicators"""
        print("🧮 Calculating economic indicators...")
        
        # Real interest rates proxy (Treasury yield - inflation proxy)
        if 'tips_10y_close' in df.columns:
            df['real_interest_rate'] = df['tips_10y_close']
        
        # Dollar strength relative to gold
        if 'dxy_close' in df.columns and 'gold_close' in df.columns:
            df['gold_dxy_ratio'] = df['gold_close'] / df['dxy_close']
            df['dxy_strength'] = (df['dxy_close'] / df['dxy_close'].rolling(30).mean() - 1) * 100
        
        # Market stress indicators
        if 'vix_close' in df.columns:
            df['market_stress'] = df['vix_close']
            df['vix_percentile'] = df['vix_close'].rolling(252).rank(pct=True) * 100
        
        # Inflation hedge performance
        if 'gold_close' in df.columns and 'sp500_close' in df.columns:
            df['gold_sp500_ratio'] = df['gold_close'] / df['sp500_close']
            df['gold_outperformance'] = (df['gold_close'].pct_change() - df['sp500_close'].pct_change()) * 100
        
        # Geopolitical stress proxy (oil volatility + VIX)
        if 'crude_oil_close' in df.columns and 'vix_close' in df.columns:
            oil_volatility = df['crude_oil_close'].pct_change().rolling(30).std() * np.sqrt(252) * 100
            df['geopolitical_stress'] = (oil_volatility + df['vix_close']) / 2
        
        # Precious metals correlation
        if 'gold_close' in df.columns and 'silver_close' in df.columns:
            df['gold_silver_ratio'] = df['gold_close'] / df['silver_close']
        
        # Bond market signals
        if 'bonds_close' in df.columns:
            df['bond_momentum'] = df['bonds_close'].pct_change(21) * 100  # 21-day momentum
        
        # Risk-on/Risk-off sentiment
        if 'bitcoin_close' in df.columns and 'gold_close' in df.columns:
            df['risk_sentiment'] = df['bitcoin_close'].pct_change(5) - df['gold_close'].pct_change(5)
        
        # Calculate moving averages for trend analysis
        for col in ['gold_close', 'dxy_close', 'vix_close']:
            if col in df.columns:
                df[f'{col}_ma20'] = df[col].rolling(20).mean()
                df[f'{col}_ma50'] = df[col].rolling(50).mean()
                df[f'{col}_trend'] = (df[col] > df[f'{col}_ma20']).astype(int)
        
        return df
    
    def get_current_market_snapshot(self) -> Dict:
        """Get current market conditions snapshot"""
        print("📸 Getting current market snapshot...")
        
        # Get latest data (5 days to ensure we have data)
        current_data = self.collect_market_data(period="5d")
        if current_data.empty:
            return {}
        
        latest = current_data.iloc[-1]
        previous = current_data.iloc[-2] if len(current_data) > 1 else latest
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'gold_price': float(latest.get('gold_close', 0)),
            'gold_change_pct': float(((latest.get('gold_close', 0) - previous.get('gold_close', 0)) / previous.get('gold_close', 1)) * 100),
            'dxy_index': float(latest.get('dxy_close', 0)),
            'dxy_change_pct': float(((latest.get('dxy_close', 0) - previous.get('dxy_close', 0)) / previous.get('dxy_close', 1)) * 100),
            'vix_level': float(latest.get('vix_close', 0)),
            'market_stress_level': 'Low' if latest.get('vix_close', 0) < 20 else 'High' if latest.get('vix_close', 0) > 30 else 'Medium',
            'sp500_level': float(latest.get('sp500_close', 0)),
            'treasury_yield': float(latest.get('tips_10y_close', 0)),
            'oil_price': float(latest.get('crude_oil_close', 0)),
            'silver_price': float(latest.get('silver_close', 0)),
            'bitcoin_price': float(latest.get('bitcoin_close', 0)),
            'bonds_level': float(latest.get('bonds_close', 0))
        }
        
        # Add derived indicators
        if snapshot['dxy_index'] > 0 and snapshot['gold_price'] > 0:
            snapshot['gold_dxy_ratio'] = snapshot['gold_price'] / snapshot['dxy_index']
        
        if snapshot['silver_price'] > 0:
            snapshot['gold_silver_ratio'] = snapshot['gold_price'] / snapshot['silver_price']
        
        return snapshot

class EnhancedGoldPredictor:
    """
    Enhanced gold price predictor using comprehensive economic data
    """
    
    def __init__(self):
        self.data_collector = EnhancedEconomicDataCollector()
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive feature set for modeling"""
        print("🔧 Preparing enhanced features...")
        
        # Calculate enhanced indicators
        df = self.data_collector.calculate_economic_indicators(df)
        
        # Feature engineering
        features = df.copy()
        
        # Price momentum features
        if 'gold_close' in features.columns:
            features['gold_return_1d'] = features['gold_close'].pct_change()
            features['gold_return_5d'] = features['gold_close'].pct_change(5)
            features['gold_return_21d'] = features['gold_close'].pct_change(21)
            features['gold_volatility'] = features['gold_return_1d'].rolling(30).std() * np.sqrt(252)
        
        # Economic stress composite score
        stress_components = []
        if 'vix_close' in features.columns:
            stress_components.append(features['vix_close'] / 100)  # Normalize VIX
        if 'dxy_strength' in features.columns:
            stress_components.append(features['dxy_strength'] / 100)  # Normalize DXY strength
        
        if stress_components:
            features['economic_stress_score'] = pd.DataFrame(stress_components).T.mean(axis=1)
        
        # Time-based features
        features['month'] = features.index.month
        features['quarter'] = features.index.quarter
        features['day_of_year'] = features.index.dayofyear
        
        # Cyclical encoding for seasonality
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features
    
    def create_enhanced_forecast(self, days: int = 7) -> Dict:
        """Create enhanced forecast using economic fundamentals"""
        print(f"🔮 Creating enhanced {days}-day forecast...")
        
        try:
            # Collect comprehensive data
            df = self.data_collector.collect_market_data(period="1y")
            
            if df.empty:
                return {"error": "No data available"}
            
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Get current market snapshot
            market_snapshot = self.data_collector.get_current_market_snapshot()
            
            # Simple trend-based prediction with economic factors
            latest_gold = features_df['gold_close'].iloc[-1]
            
            # Economic factor adjustments
            adjustments = self._calculate_economic_adjustments(features_df, market_snapshot)
            
            # Generate forecast
            forecast_data = []
            base_price = latest_gold
            
            for i in range(1, days + 1):
                # Trend component
                trend_factor = adjustments['trend_factor'] * (i / days)
                
                # Economic stress component
                stress_adjustment = adjustments['stress_adjustment'] * (1 / i)  # Diminishing effect
                
                # Dollar strength component
                dollar_adjustment = adjustments['dollar_adjustment']
                
                # Market sentiment component
                sentiment_adjustment = adjustments['sentiment_adjustment']
                
                # Combined prediction
                predicted_price = base_price * (1 + trend_factor + stress_adjustment + dollar_adjustment + sentiment_adjustment)
                
                # Add some realistic noise
                noise_factor = np.random.normal(0, 0.005)  # 0.5% daily volatility
                predicted_price *= (1 + noise_factor)
                
                forecast_data.append({
                    'day': i,
                    'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'predicted_price': round(predicted_price, 2),
                    'change_from_today': round(((predicted_price - latest_gold) / latest_gold) * 100, 2),
                    'confidence': max(0.6, 0.95 - (i * 0.05))  # Decreasing confidence
                })
                
                base_price = predicted_price  # Update base for next day
            
            return {
                'current_price': round(latest_gold, 2),
                'forecast': forecast_data,
                'market_conditions': market_snapshot,
                'economic_factors': adjustments,
                'model_info': {
                    'data_points': len(df),
                    'features_used': len([col for col in features_df.columns if not features_df[col].isna().all()]),
                    'forecast_horizon': f"{days} days",
                    'last_updated': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"error": f"Forecast generation failed: {str(e)}"}
    
    def _calculate_economic_adjustments(self, df: pd.DataFrame, market_snapshot: Dict) -> Dict:
        """Calculate economic factor adjustments for forecast"""
        adjustments = {
            'trend_factor': 0.0,
            'stress_adjustment': 0.0,
            'dollar_adjustment': 0.0,
            'sentiment_adjustment': 0.0
        }
        
        # Trend analysis
        if 'gold_close' in df.columns:
            recent_trend = df['gold_close'].pct_change(20).iloc[-1]  # 20-day trend
            adjustments['trend_factor'] = np.clip(recent_trend, -0.05, 0.05)  # Cap at 5%
        
        # Market stress (VIX effect)
        vix_level = market_snapshot.get('vix_level', 20)
        if vix_level > 30:  # High stress
            adjustments['stress_adjustment'] = 0.02  # Gold benefits from stress
        elif vix_level < 15:  # Low stress
            adjustments['stress_adjustment'] = -0.01  # Gold less attractive
        
        # Dollar strength effect
        dxy_change = market_snapshot.get('dxy_change_pct', 0)
        adjustments['dollar_adjustment'] = -dxy_change * 0.001  # Inverse relationship
        
        # Market sentiment (S&P 500 correlation)
        gold_change = market_snapshot.get('gold_change_pct', 0)
        if abs(gold_change) > 2:  # Significant move
            adjustments['sentiment_adjustment'] = gold_change * 0.1  # Momentum effect
        
        return adjustments

    def get_correlation_analysis(self) -> Dict:
        """Get correlation analysis between gold and economic indicators"""
        try:
            df = self.data_collector.collect_market_data(period="1y")
            if df.empty:
                return {"error": "No data available"}
            
            # Calculate correlations
            gold_price = df['gold_close']
            correlations = {}
            
            correlation_pairs = {
                'dxy_close': 'US Dollar Index',
                'vix_close': 'VIX (Market Fear)',
                'sp500_close': 'S&P 500',
                'crude_oil_close': 'Oil Prices',
                'silver_close': 'Silver',
                'bitcoin_close': 'Bitcoin',
                'bonds_close': 'Treasury Bonds'
            }
            
            for col, name in correlation_pairs.items():
                if col in df.columns:
                    corr = gold_price.corr(df[col])
                    correlations[name] = {
                        'correlation': round(corr, 3),
                        'relationship': 'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak',
                        'direction': 'Positive' if corr > 0 else 'Negative'
                    }
            
            return {
                'correlations': correlations,
                'analysis_period': '1 year',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}

# Initialize the enhanced predictor
enhanced_predictor = EnhancedGoldPredictor()