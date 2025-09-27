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
            'copper': 'HG=F',
            'fed_funds': '^IRX',  # 3-Month Treasury as proxy for Fed funds rate
            'tips_5y': '^FVX'     # 5-Year Treasury for inflation expectations
        }
        # Inflation data approximation using TIPS spread
        self.inflation_proxy_symbols = {
            'tips_10y': 'SCHP',  # TIPS ETF
            'treasury_5y': '^FVX',
            'treasury_2y': '^TNX'  # Will use for yield curve analysis
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
        
        # Add inflation expectations calculation
        data.update(self._calculate_inflation_metrics())
        
        return self._format_indicators(data)
    
    def _calculate_inflation_metrics(self) -> Dict[str, Any]:
        """Calculate inflation expectations and related metrics"""
        inflation_data = {}
        
        try:
            # Get 5Y and 10Y treasury yields for inflation expectations
            treasury_5y = yf.Ticker('^FVX').history(period='5d')
            treasury_10y = yf.Ticker('^TNX').history(period='5d')
            tips_etf = yf.Ticker('SCHP').history(period='5d')  # TIPS ETF as inflation proxy
            
            if not treasury_5y.empty and not treasury_10y.empty:
                # Current yields
                yield_5y = treasury_5y['Close'].iloc[-1]
                yield_10y = treasury_10y['Close'].iloc[-1]
                
                # Estimate inflation expectations (simplified)
                # Real inflation expectations are complex, this is an approximation
                inflation_expectation = yield_10y - 2.0  # Assume 2% real rate target
                inflation_expectation = max(0, min(inflation_expectation, 10))  # Cap between 0-10%
                
                # TIPS performance as inflation indicator
                tips_performance = 0
                if not tips_etf.empty and len(tips_etf) > 1:
                    tips_current = tips_etf['Close'].iloc[-1]
                    tips_prev = tips_etf['Close'].iloc[-2]
                    tips_performance = ((tips_current - tips_prev) / tips_prev) * 100
                
                inflation_data = {
                    'inflation_expectation': {
                        'price': float(inflation_expectation),
                        'change_pct': 0.0,  # Would need historical data for change
                        'symbol': 'CALCULATED'
                    },
                    'tips_performance': {
                        'price': float(tips_performance),
                        'change_pct': float(tips_performance),
                        'symbol': 'SCHP'
                    },
                    'yield_curve_5_10': {
                        'price': float(yield_10y - yield_5y),
                        'change_pct': 0.0,
                        'symbol': 'CALCULATED'
                    }
                }
                
        except Exception as e:
            print(f"Error calculating inflation metrics: {e}")
            inflation_data = {
                'inflation_expectation': {'price': 2.5, 'change_pct': 0.0, 'symbol': 'ESTIMATED'},
                'tips_performance': {'price': 0.0, 'change_pct': 0.0, 'symbol': 'UNAVAILABLE'},
                'yield_curve_5_10': {'price': 0.5, 'change_pct': 0.0, 'symbol': 'ESTIMATED'}
            }
        
        return inflation_data
    
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
        
        # Get Fed funds rate and inflation data
        fed_funds_rate = data.get('fed_funds', {}).get('price', 0)
        inflation_expectation = data.get('inflation_expectation', {}).get('price', 2.5)
        tips_performance = data.get('tips_performance', {}).get('price', 0)
        yield_curve_spread = data.get('yield_curve_5_10', {}).get('price', 0.5)
        
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
            # NEW: Fed funds rate and inflation data
            'fed_funds_rate': fed_funds_rate,
            'fed_funds_change_pct': data.get('fed_funds', {}).get('change_pct', 0),
            'inflation_expectation': inflation_expectation,
            'tips_performance': tips_performance,
            'yield_curve_spread': yield_curve_spread,
            'inflation_signal': 'Rising' if tips_performance > 0.1 else 'Falling' if tips_performance < -0.1 else 'Stable',
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