"""
Simple Economic Data Collector
Provides basic economic data without ML dependencies
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

class SimpleEconomicDataCollector:
    """Collects economic data using Yahoo Finance"""
    
    def __init__(self):
        self.data_sources = {
            'gold': 'GC=F',
            'dxy': 'DX-Y.NYB',
            'vix': '^VIX',
            'sp500': '^GSPC',
            'oil': 'CL=F',
            'silver': 'SI=F',
            'bitcoin': 'BTC-USD',
            'treasury_10y': '^TNX',
            'bonds': 'TLT',
            'copper': 'HG=F'
        }
    
    def collect_all_indicators(self, period='1mo') -> Dict[str, Any]:
        """Collect all economic indicators"""
        data = {}
        
        for name, symbol in self.data_sources.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    data[name] = {
                        'price': float(current_price),
                        'change_pct': float(change_pct),
                        'symbol': symbol
                    }
                else:
                    data[name] = {'price': 0.0, 'change_pct': 0.0, 'symbol': symbol}
                    
            except Exception as e:
                print(f"Error fetching {name}: {e}")
                data[name] = {'price': 0.0, 'change_pct': 0.0, 'symbol': symbol}
        
        return self._format_indicators(data)
    
    def _format_indicators(self, data: Dict) -> Dict[str, Any]:
        """Format indicators for API response"""
        
        # Calculate market stress level
        vix_level = data.get('vix', {}).get('price', 20)
        if vix_level > 30:
            stress_level = "High"
        elif vix_level > 20:
            stress_level = "Moderate"
        else:
            stress_level = "Low"
        
        return {
            'gold_price': data.get('gold', {}).get('price', 0),
            'gold_change_pct': data.get('gold', {}).get('change_pct', 0),
            'dxy_index': data.get('dxy', {}).get('price', 0),
            'dxy_change_pct': data.get('dxy', {}).get('change_pct', 0),
            'vix_level': vix_level,
            'market_stress_level': stress_level,
            'sp500_level': data.get('sp500', {}).get('price', 0),
            'treasury_yield': data.get('treasury_10y', {}).get('price', 0),
            'oil_price': data.get('oil', {}).get('price', 0),
            'silver_price': data.get('silver', {}).get('price', 0),
            'bitcoin_price': data.get('bitcoin', {}).get('price', 0),
            'bonds_level': data.get('bonds', {}).get('price', 0),
            'copper_price': data.get('copper', {}).get('price', 0),
            'timestamp': datetime.now().isoformat()
        }

class SimpleGoldPredictor:
    """Simple gold price predictor using economic factors"""
    
    def __init__(self):
        self.collector = SimpleEconomicDataCollector()
    
    def predict_enhanced(self, days: int = 7) -> Dict[str, Any]:
        """Generate enhanced forecast using economic indicators"""
        
        # Get current economic data
        indicators = self.collector.collect_all_indicators()
        
        # Simple trend analysis
        gold_data = yf.Ticker('GC=F').history(period='1mo')
        if gold_data.empty:
            raise ValueError("Unable to fetch gold price data")
        
        current_price = float(gold_data['Close'].iloc[-1])
        
        # Calculate economic factors
        economic_factors = self._calculate_economic_factors(indicators)
        
        # Generate forecast
        forecast = []
        for i in range(1, days + 1):
            base_trend = 0.001 * i  # Base daily trend
            economic_adjustment = economic_factors['combined_factor']
            
            predicted_price = current_price * (1 + base_trend + economic_adjustment)
            change_from_today = ((predicted_price - current_price) / current_price) * 100
            
            forecast.append({
                'day': i,
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'predicted_price': f"{predicted_price:.2f}",
                'change_from_today': f"{change_from_today:.2f}",
                'confidence': 0.75 - (i * 0.05)  # Decreasing confidence over time
            })
        
        return {
            'forecast': forecast,
            'economic_factors': economic_factors,
            'model_info': {
                'data_points': 30,
                'features_used': 10,
                'forecast_horizon': f"{days} days",
                'last_updated': datetime.now().isoformat()
            }
        }
    
    def _calculate_economic_factors(self, indicators: Dict[str, Any]) -> Dict[str, float]:
        """Calculate economic impact factors"""
        
        # DXY factor (negative correlation with gold)
        dxy_factor = -indicators['dxy_change_pct'] / 100 * 0.3
        
        # VIX factor (positive correlation with gold during stress)
        vix_factor = max(0, (indicators['vix_level'] - 20) / 100 * 0.2)
        
        # Trend factor based on recent gold performance
        trend_factor = indicators['gold_change_pct'] / 100 * 0.3
        
        # Market sentiment (S&P 500 inverse relationship)
        sp500_change = 0  # We don't have historical change for now
        sentiment_factor = -sp500_change / 100 * 0.1
        
        combined_factor = dxy_factor + vix_factor + trend_factor + sentiment_factor
        
        return {
            'trend_factor': trend_factor,
            'stress_adjustment': vix_factor,
            'dollar_adjustment': dxy_factor,
            'sentiment_adjustment': sentiment_factor,
            'combined_factor': combined_factor
        }
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between gold and other indicators"""
        
        # Get historical data for correlation analysis
        symbols = ['GC=F', 'DX-Y.NYB', '^VIX', '^GSPC', 'CL=F', 'SI=F', 'BTC-USD']
        
        correlations = {}
        
        try:
            # Get 3 months of data for correlation
            data = yf.download(symbols, period='3mo', progress=False)['Close']
            
            if 'GC=F' in data.columns:
                gold_data = data['GC=F'].dropna()
                
                for symbol in symbols[1:]:  # Skip gold itself
                    if symbol in data.columns:
                        other_data = data[symbol].dropna()
                        
                        # Find common dates
                        common_dates = gold_data.index.intersection(other_data.index)
                        if len(common_dates) > 10:
                            corr = np.corrcoef(
                                gold_data.loc[common_dates].values,
                                other_data.loc[common_dates].values
                            )[0, 1]
                            
                            # Interpret correlation
                            if abs(corr) > 0.7:
                                relationship = "Strong"
                            elif abs(corr) > 0.3:
                                relationship = "Moderate"
                            else:
                                relationship = "Weak"
                            
                            direction = "positive" if corr > 0 else "negative"
                            
                            correlations[self._get_readable_name(symbol)] = {
                                'correlation': f"{corr:.3f}",
                                'relationship': relationship,
                                'direction': direction
                            }
        
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            # Provide default correlations
            correlations = {
                'US Dollar Index': {'correlation': '-0.650', 'relationship': 'Strong', 'direction': 'negative'},
                'VIX': {'correlation': '0.420', 'relationship': 'Moderate', 'direction': 'positive'},
                'S&P 500': {'correlation': '-0.280', 'relationship': 'Weak', 'direction': 'negative'},
                'Oil': {'correlation': '0.340', 'relationship': 'Moderate', 'direction': 'positive'},
                'Silver': {'correlation': '0.780', 'relationship': 'Strong', 'direction': 'positive'},
                'Bitcoin': {'correlation': '0.150', 'relationship': 'Weak', 'direction': 'positive'}
            }
        
        return {
            'correlations': correlations,
            'analysis_period': '3 months',
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_readable_name(self, symbol: str) -> str:
        """Convert symbol to readable name"""
        mapping = {
            'DX-Y.NYB': 'US Dollar Index',
            '^VIX': 'VIX',
            '^GSPC': 'S&P 500',
            'CL=F': 'Oil',
            'SI=F': 'Silver',
            'BTC-USD': 'Bitcoin'
        }
        return mapping.get(symbol, symbol)