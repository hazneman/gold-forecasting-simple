"""FastAPI Application for Gold Price Forecasting

This module provides a REST API for gold price forecasting services including:
- Real-time price predictions
- Historical data analysis
- Model training and evaluation
- Risk metrics calculation
- Backtesting results
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import joblib
from pathlib import Path
import yfinance as yf

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection import GoldDataCollector
from src.feature_engineering import FeatureEngineer
from src.models import create_model, ModelEnsemble
from src.visualization import GoldPriceVisualizer
from src.risk_management import RiskManager
from src.backtesting import Backtester, BacktestConfig, MovingAverageCrossoverStrategy
from src.simple_economic_data import SimpleEconomicDataCollector, SimpleGoldPredictor
from src.ml_predictor import MLGoldPredictor, get_model_decision_explanation
from src.auto_trainer import auto_trainer
from src.enhanced_auto_trainer import enhanced_trainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gold Price Forecasting API",
    description="API for gold price predictions and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files to serve the dashboard
app.mount("/static", StaticFiles(directory="."), name="static")

# Global variables for caching
cached_model = None
cached_data = None
last_update = None


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    """Request model for price predictions."""
    features: Dict[str, float] = Field(..., description="Feature values for prediction")
    model_type: str = Field("ensemble", description="Type of model to use")
    

class PredictionResponse(BaseModel):
    """Response model for price predictions."""
    predicted_price: float = Field(..., description="Predicted gold price")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="Confidence interval")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_used: str = Field(..., description="Model used for prediction")


class HistoricalDataRequest(BaseModel):
    """Request model for historical data."""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    source: str = Field("yahoo", description="Data source")


class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    strategy_type: str = Field("ma_crossover", description="Strategy type")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    initial_capital: float = Field(100000, description="Initial capital")
    commission: float = Field(0.001, description="Commission rate")


class RiskMetricsRequest(BaseModel):
    """Request model for risk metrics."""
    prices: List[float] = Field(..., description="Price series")
    dates: List[str] = Field(..., description="Date series")


# Utility functions
async def load_model(model_type: str = "ensemble"):
    """Load or create a model."""
    global cached_model
    
    if cached_model is None:
        try:
            # Try to load saved model
            model_path = Path(f"data/models/{model_type}_model.pkl")
            if model_path.exists():
                cached_model = joblib.load(model_path)
                logger.info(f"Loaded cached {model_type} model")
            else:
                # Create and train a new model (simplified for demo)
                logger.info(f"Creating new {model_type} model")
                cached_model = create_model("random_forest", n_estimators=100)
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to simple model
            cached_model = create_model("linear")
    
    return cached_model


async def get_latest_data():
    """Get latest gold price data."""
    global cached_data, last_update
    
    # Check if we need to update data (every hour)
    if (cached_data is None or 
        last_update is None or 
        datetime.now() - last_update > timedelta(hours=1)):
        
        try:
            collector = GoldDataCollector()
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            cached_data = collector.get_gold_prices(start_date, end_date)
            last_update = datetime.now()
            
            logger.info(f"Updated data cache with {len(cached_data)} records")
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            # Create dummy data for demo
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            cached_data = pd.DataFrame({
                'Close': np.random.randn(100).cumsum() + 1800,
                'Volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
    
    return cached_data


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Gold Price Forecasting API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/economic_dashboard.html")
async def get_dashboard():
    """Serve the economic dashboard."""
    from fastapi.responses import FileResponse
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "economic_dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path, media_type="text/html")
    else:
        raise HTTPException(status_code=404, detail="Dashboard not found")


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Predict gold price based on features."""
    try:
        model = await load_model(request.model_type)
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([request.features])
        
        # Make prediction
        if hasattr(model, 'predict'):
            prediction = model.predict(feature_df)[0]
        else:
            # Fallback prediction
            prediction = sum(request.features.values()) / len(request.features) * 1800
        
        response = PredictionResponse(
            predicted_price=float(prediction),
            timestamp=datetime.now(),
            model_used=request.model_type
        )
        
        logger.info(f"Made prediction: {prediction:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/current-price")
async def get_current_price():
    """Get current gold price."""
    try:
        data = await get_latest_data()
        current_price = data['Close'].iloc[-1]
        
        return {
            "current_price": float(current_price),
            "timestamp": data.index[-1].isoformat(),
            "change_24h": float(data['Close'].iloc[-1] - data['Close'].iloc[-2]),
            "change_pct_24h": float((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100)
        }
        
    except Exception as e:
        logger.error(f"Error getting current price: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current price: {str(e)}")


@app.get("/simple-forecast")
async def get_simple_forecast(days: int = 7):
    """Get a simple forecast for the next N days using basic trend analysis."""
    try:
        data = await get_latest_data()
        
        # Get recent prices for trend analysis
        recent_prices = data['Close'].tail(30).values
        current_price = recent_prices[-1]
        
        # Calculate simple moving averages for trend
        ma_5 = np.mean(recent_prices[-5:])
        ma_20 = np.mean(recent_prices[-20:])
        
        # Calculate daily returns for volatility
        returns = np.diff(recent_prices) / recent_prices[:-1]
        daily_volatility = np.std(returns)
        
        # Simple trend-based forecast
        trend = (ma_5 - ma_20) / ma_20  # Trend direction
        
        forecasts = []
        for i in range(1, days + 1):
            # Simple random walk with trend
            forecast_price = current_price * (1 + trend * 0.1 * i + np.random.normal(0, daily_volatility * 0.5))
            forecast_date = (datetime.now() + timedelta(days=i)).date()
            
            forecasts.append({
                "date": forecast_date.isoformat(),
                "predicted_price": round(float(forecast_price), 2),
                "confidence": "medium"  # Simple confidence indicator
            })
        
        return {
            "forecasts": forecasts,
            "current_price": float(current_price),
            "trend_direction": "bullish" if trend > 0.01 else "bearish" if trend < -0.01 else "neutral",
            "forecast_method": "trend_analysis",
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating simple forecast: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate forecast: {str(e)}")


@app.get("/enhanced-forecast")
async def enhanced_forecast(days: int = 7):
    """Get enhanced gold price forecast using economic indicators"""
    try:
        predictor = SimpleGoldPredictor()
        forecast_data = predictor.predict_enhanced(days)
        
        return {
            "status": "success",
            "forecast": forecast_data["forecast"],
            "economic_factors": forecast_data["economic_factors"],
            "model_info": forecast_data["model_info"]
        }
        
    except Exception as e:
        logger.error(f"Enhanced forecast error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "forecast": []
        }

@app.get("/correlation-analysis")
async def get_correlation_analysis():
    """Get correlation analysis between gold and various economic indicators"""
    try:
        from src.simple_economic_data import SimpleGoldPredictor
        
        predictor = SimpleGoldPredictor()
        correlations = predictor.analyze_correlations()
        
        return {
            "status": "success",
            **correlations
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/economic-indicators") 
async def get_economic_indicators():
    """Get current economic indicators that influence gold prices"""
    try:
        collector = SimpleEconomicDataCollector()
        indicators = collector.collect_all_indicators()
        
        return {
            "status": "success",
            "data": indicators,
            "indicators_included": [
                "Gold Price & Trend",
                "US Dollar Index (DXY)", 
                "Market Volatility (VIX)",
                "S&P 500 Level",
                "Treasury Yields",
                "Oil Prices (Geopolitical Proxy)",
                "Silver Prices", 
                "Bitcoin (Risk Asset)",
                "Treasury Bonds"
            ]
        }
        
    except Exception as e:
        logger.error(f"Economic indicators error: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/working-forecast")
async def get_working_forecast(days: int = 7):
    """Get working ML-based gold price forecast with comprehensive analysis"""
    try:
        # Import and run the working predictor
        import sys
        sys.path.append('.')
        
        from src.working_predictor import WorkingGoldPredictor
        import yfinance as yf
        import pandas as pd
        
        # Initialize predictor
        predictor = WorkingGoldPredictor()
        
        # Collect market data with better alignment
        symbols = {
            'GOLD': 'GC=F',
            'DXY': 'DX-Y.NYB', 
            'VIX': '^VIX',
            'SP500': '^GSPC',
            'TNX': '^TNX',
            'OIL': 'CL=F',
            'SILVER': 'SI=F'
        }
        
        data = {}
        for name, symbol in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='3mo')
                if not hist.empty:
                    data[name] = hist['Close']
            except:
                continue
        
        # Create DataFrame and handle alignment
        market_data = pd.DataFrame(data)
        market_data = market_data.ffill().dropna()
        
        if market_data.empty or 'GOLD' not in market_data.columns:
            raise ValueError("Unable to collect sufficient market data")
        
        current_price = float(market_data['GOLD'].iloc[-1])
        
        # Prepare features
        technical_features = predictor.create_technical_features(market_data['GOLD'])
        fundamental_features = predictor.create_fundamental_features(market_data)
        
        # Combine features
        all_features = pd.concat([technical_features, fundamental_features], axis=1)
        all_features = all_features.ffill().dropna()
        
        if len(all_features) < 30:
            raise ValueError("Insufficient data for ML analysis")
        
        # Create target and train
        target = market_data['GOLD'].shift(-5) / market_data['GOLD'] - 1
        target = target.loc[all_features.index]
        
        mask = ~(all_features.isna().any(axis=1) | target.isna())
        X = all_features[mask]
        y = target[mask]
        
        # Train model
        training_results = predictor.train_model(X, y)
        
        # Make prediction
        current_features = X.iloc[-1:].copy()
        current_features['price'] = current_price
        
        prediction = predictor.predict(current_features, days)
        
        # Get current signals
        signals = predictor.get_current_signals(X)
        
        # Calculate forecast for each day
        forecast = []
        predicted_price = prediction['predicted_price']
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        for i in range(1, days + 1):
            day_price = current_price + (predicted_price - current_price) * (i / days)
            day_change = ((day_price - current_price) / current_price) * 100
            
            forecast.append({
                'day': i,
                'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'predicted_price': f"{day_price:.2f}",
                'change_from_today': f"{day_change:+.2f}",
                'confidence': prediction['confidence']
            })
        
        return {
            "status": "success",
            "forecast": forecast,
            "analysis": {
                "current_price": f"${current_price:.2f}",
                "predicted_price": f"${predicted_price:.2f}",
                "expected_change": f"{change_pct:+.2f}%",
                "confidence_level": f"{prediction['confidence']*100:.1f}%",
                "prediction_method": prediction['method']
            },
            "market_signals": signals,
            "model_info": {
                "data_points": len(X),
                "features_analyzed": len(X.columns),
                "training_results": training_results,
                "last_updated": datetime.now().isoformat()
            },
            "feature_importance": predictor.feature_importance.head(10).to_dict() if predictor.feature_importance is not None else {}
        }
        
    except Exception as e:
        logger.error(f"Working forecast error: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "fallback_recommendation": "Try /enhanced-forecast for basic analysis"
        }

@app.get("/ml-forecast")
async def get_ml_forecast(days: int = 7, retrain: bool = False):
    """Get ML-based gold price forecast combining technicals and fundamentals"""
    try:
        # Initialize ML predictor
        ml_predictor = MLGoldPredictor()
        
        # Get gold price data
        gold_ticker = yf.Ticker('GC=F')
        gold_data = gold_ticker.history(period='1y')
        
        if gold_data.empty:
            raise HTTPException(status_code=500, detail="Unable to fetch gold price data")
        
        # Get fundamental data
        fundamental_data = ml_predictor.fundamental_analyzer.get_fundamental_features('1y')
        
        # Prepare features
        features = ml_predictor.prepare_features(gold_data, fundamental_data)
        
        if len(features) < 50:
            raise HTTPException(status_code=500, detail="Insufficient data for ML training")
        
        # Train models if needed
        if retrain or not ml_predictor.models:
            logger.info("Training ML models...")
            targets = ml_predictor.create_target_variables(gold_data, [1, 3, 5, 7, 14])
            training_results = ml_predictor.train_models(features, targets)
        else:
            training_results = {"status": "Using existing models"}
        
        # Make predictions
        prediction_result = ml_predictor.predict_price(features, days)
        
        # Get feature importance
        feature_importance = ml_predictor.get_feature_importance_summary()
        
        # Generate explanation
        decision_explanation = get_model_decision_explanation(prediction_result, feature_importance)
        
        # Format forecast output
        current_price = prediction_result['current_price']
        predicted_prices = prediction_result['predictions']
        confidence_scores = prediction_result['confidence_scores']
        
        forecast = []
        for i in range(days):
            if i < len(predicted_prices):
                predicted_price = predicted_prices[i]
                confidence = confidence_scores[i] if i < len(confidence_scores) else 0.5
                change_pct = ((predicted_price - current_price) / current_price) * 100
                
                forecast.append({
                    'day': i + 1,
                    'date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'predicted_price': f"{predicted_price:.2f}",
                    'change_from_today': f"{change_pct:.2f}",
                    'confidence': confidence,
                    'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
                })
        
        return {
            "status": "success",
            "forecast": forecast,
            "model_info": {
                "prediction_method": prediction_result.get('prediction_method', 'ML'),
                "features_analyzed": prediction_result.get('model_features_used', 0),
                "data_points": len(features),
                "training_results": training_results,
                "last_updated": datetime.now().isoformat()
            },
            "feature_importance": feature_importance,
            "decision_explanation": decision_explanation,
            "current_analysis": {
                "current_price": f"${current_price:.2f}",
                "technical_signals": _get_current_technical_signals(features),
                "fundamental_factors": _get_current_fundamental_factors(fundamental_data)
            }
        }
        
    except Exception as e:
        logger.error(f"ML forecast error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "fallback_available": True,
            "recommendation": "Try the /enhanced-forecast endpoint for basic analysis"
        }

def _get_current_technical_signals(features: pd.DataFrame) -> Dict[str, Any]:
    """Extract current technical analysis signals"""
    if features.empty:
        return {"status": "No technical data available"}
    
    latest = features.iloc[-1]
    
    signals = {}
    
    # RSI signals
    if 'rsi' in latest:
        rsi = latest['rsi']
        if rsi > 70:
            signals['rsi'] = {'value': f"{rsi:.1f}", 'signal': 'Overbought', 'action': 'Sell Signal'}
        elif rsi < 30:
            signals['rsi'] = {'value': f"{rsi:.1f}", 'signal': 'Oversold', 'action': 'Buy Signal'}
        else:
            signals['rsi'] = {'value': f"{rsi:.1f}", 'signal': 'Neutral', 'action': 'Hold'}
    
    # MACD signals
    if 'macd_bullish' in latest:
        macd_bullish = latest['macd_bullish']
        signals['macd'] = {
            'signal': 'Bullish' if macd_bullish else 'Bearish',
            'action': 'Buy Signal' if macd_bullish else 'Sell Signal'
        }
    
    # Moving average trends
    if 'uptrend_strength' in latest:
        trend_strength = latest['uptrend_strength']
        if trend_strength >= 2:
            signals['trend'] = {'strength': 'Strong Uptrend', 'action': 'Buy Signal'}
        elif trend_strength <= 1:
            signals['trend'] = {'strength': 'Downtrend', 'action': 'Sell Signal'}
        else:
            signals['trend'] = {'strength': 'Sideways', 'action': 'Hold'}
    
    # Bollinger Bands position
    if 'bb_position' in latest:
        bb_pos = latest['bb_position']
        if bb_pos > 0.8:
            signals['bollinger'] = {'position': 'Upper Band', 'signal': 'Overbought'}
        elif bb_pos < 0.2:
            signals['bollinger'] = {'position': 'Lower Band', 'signal': 'Oversold'}
        else:
            signals['bollinger'] = {'position': 'Middle Range', 'signal': 'Neutral'}
    
    return signals

def _get_current_fundamental_factors(fundamental_data: pd.DataFrame) -> Dict[str, Any]:
    """Extract current fundamental analysis factors"""
    if fundamental_data.empty:
        return {"status": "No fundamental data available"}
    
    latest = fundamental_data.iloc[-1]
    factors = {}
    
    # Check key economic indicators
    key_indicators = ['dxy_price', 'vix_price', 'sp500_price', 'treasury_10y_price', 'oil_price']
    
    for indicator in key_indicators:
        if indicator in latest:
            value = latest[indicator]
            change_col = indicator.replace('_price', '_change_1d')
            
            if change_col in latest:
                change = latest[change_col] * 100  # Convert to percentage
                
                factors[indicator.replace('_price', '')] = {
                    'current_value': f"{value:.2f}",
                    'daily_change': f"{change:+.2f}%",
                    'impact_on_gold': _assess_gold_impact(indicator, change)
                }
    
    return factors

def _assess_gold_impact(indicator: str, change_pct: float) -> str:
    """Assess how economic indicator changes impact gold"""
    
    if 'dxy' in indicator:  # US Dollar - inverse relationship
        if change_pct > 0.5:
            return "Negative (stronger dollar typically pressures gold)"
        elif change_pct < -0.5:
            return "Positive (weaker dollar supports gold)"
        else:
            return "Neutral"
    
    elif 'vix' in indicator:  # Volatility - positive relationship
        if change_pct > 5:
            return "Positive (higher uncertainty supports gold)"
        elif change_pct < -5:
            return "Negative (lower uncertainty reduces gold appeal)"
        else:
            return "Neutral"
    
    elif 'sp500' in indicator:  # S&P 500 - generally inverse
        if change_pct > 1:
            return "Negative (strong equities reduce gold appeal)"
        elif change_pct < -1:
            return "Positive (weak equities support gold)"
        else:
            return "Neutral"
    
    elif 'treasury' in indicator:  # Treasury yields - inverse relationship
        if change_pct > 0.1:
            return "Negative (higher yields compete with gold)"
        elif change_pct < -0.1:
            return "Positive (lower yields support gold)"
        else:
            return "Neutral"
    
    elif 'oil' in indicator:  # Oil - positive correlation
        if change_pct > 2:
            return "Positive (oil strength supports commodities)"
        elif change_pct < -2:
            return "Negative (oil weakness pressures commodities)"
        else:
            return "Neutral"
    
    return "Unknown impact"

@app.get("/enhanced-forecast")
async def get_enhanced_forecast(days: int = 7):
    """Get enhanced gold price forecast using economic indicators"""
    try:
        from src.simple_economic_data import SimpleGoldPredictor
        
        predictor = SimpleGoldPredictor()
        forecast_data = predictor.predict_enhanced(days)
        
        return {
            "status": "success",
            "forecast": forecast_data["forecast"],
            "economic_factors": forecast_data["economic_factors"],
            "model_info": forecast_data["model_info"]
        }
        
    except Exception as e:
        logger.error(f"Enhanced forecast error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "forecast": []
        }

@app.get("/correlation-analysis")
async def get_correlation_analysis():
    """Get correlation analysis between gold and various economic indicators"""
    try:
        from src.simple_economic_data import SimpleGoldPredictor
        
        predictor = SimpleGoldPredictor()
        correlations = predictor.analyze_correlations()
        
        return {
            "status": "success",
            **correlations
        }
        
    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/technical-analysis")
async def get_technical_analysis():
    """Get comprehensive technical analysis for gold price"""
    try:
        # Use the same data collection method as other endpoints
        gold_ticker = yf.Ticker("GC=F")  # Gold futures
        gold_data = gold_ticker.history(period="6mo")
        
        if gold_data.empty:
            return {"status": "error", "error": "No gold price data available"}
        
        # Create technical analyzer and calculate indicators
        from src.ml_predictor import TechnicalIndicators
        tech_analyzer = TechnicalIndicators()
        
        close = gold_data['Close']
        high = gold_data['High']
        low = gold_data['Low']
        
        # Calculate technical indicators
        indicators = pd.DataFrame(index=gold_data.index)
        indicators['rsi'] = tech_analyzer.rsi(close)
        
        # MACD
        macd_data = tech_analyzer.macd(close)
        indicators['macd'] = macd_data['macd']
        indicators['macd_signal'] = macd_data['signal']
        indicators['macd_bullish'] = (indicators['macd'] > indicators['macd_signal']).astype(int)
        
        # Bollinger Bands
        bb_data = tech_analyzer.bollinger_bands(close)
        indicators['bb_upper'] = bb_data['upper']
        indicators['bb_lower'] = bb_data['lower']
        indicators['bb_position'] = (close - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
        
        # Moving averages
        indicators['sma_20'] = tech_analyzer.sma(close, 20)
        indicators['sma_50'] = tech_analyzer.sma(close, 50)
        indicators['uptrend_strength'] = 0  # Simplified calculation
        
        # Check if moving averages are in uptrend
        sma_20 = indicators['sma_20'].iloc[-1]
        sma_50 = indicators['sma_50'].iloc[-1]
        current_price = close.iloc[-1]
        
        uptrend_score = 0
        if current_price > sma_20:
            uptrend_score += 1
        if current_price > sma_50:
            uptrend_score += 1
        if sma_20 > sma_50:
            uptrend_score += 1
            
        indicators.loc[indicators.index[-1], 'uptrend_strength'] = uptrend_score
        
        # Volatility
        indicators['volatility_20'] = close.rolling(20).std() / close.rolling(20).mean()
        
        # Drop NaN values
        indicators = indicators.dropna()
        
        if indicators.empty:
            return {"status": "error", "error": "Unable to calculate technical indicators"}
        
        # Get current technical signals
        current_signals = _get_current_technical_signals(indicators)
        
        # Get latest price data
        latest_price = close.iloc[-1]
        price_change = close.pct_change().iloc[-1] * 100
        
        # Get latest indicators
        latest_indicators = indicators.iloc[-1]
        
        analysis_results = {
            "status": "success",
            "current_price": f"${latest_price:.2f}",
            "price_change_24h": f"{price_change:+.2f}%",
            "technical_signals": current_signals,
            "indicators": {
                "rsi": {
                    "value": f"{latest_indicators.get('rsi', 0):.1f}",
                    "interpretation": "Overbought" if latest_indicators.get('rsi', 50) > 70 else "Oversold" if latest_indicators.get('rsi', 50) < 30 else "Neutral"
                },
                "macd": {
                    "bullish": bool(latest_indicators.get('macd_bullish', False)),
                    "signal": "Bullish" if latest_indicators.get('macd_bullish', False) else "Bearish"
                },
                "bollinger_bands": {
                    "position": f"{latest_indicators.get('bb_position', 0.5):.2f}",
                    "interpretation": "Upper Band" if latest_indicators.get('bb_position', 0.5) > 0.8 else "Lower Band" if latest_indicators.get('bb_position', 0.5) < 0.2 else "Middle Range"
                },
                "trend_strength": {
                    "value": int(latest_indicators.get('uptrend_strength', 0)),
                    "signal": "Strong Uptrend" if latest_indicators.get('uptrend_strength', 0) >= 2 else "Downtrend" if latest_indicators.get('uptrend_strength', 0) <= 1 else "Sideways"
                },
                "volatility": {
                    "value": f"{latest_indicators.get('volatility_20', 0):.4f}",
                    "level": "High" if latest_indicators.get('volatility_20', 0) > 0.02 else "Low"
                }
            },
            "overall_sentiment": _determine_overall_sentiment(current_signals),
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Technical analysis error: {e}")
        return {"status": "error", "error": str(e)}

def _determine_overall_sentiment(signals: Dict[str, Any]) -> str:
    """Determine overall market sentiment from technical signals"""
    if not signals or len(signals) == 0:
        return "Neutral"
    
    bullish_count = 0
    bearish_count = 0
    total_signals = 0
    
    for signal_name, signal_data in signals.items():
        if isinstance(signal_data, dict) and 'action' in signal_data:
            total_signals += 1
            action = signal_data['action'].lower()
            if 'buy' in action:
                bullish_count += 1
            elif 'sell' in action:
                bearish_count += 1
    
    if total_signals == 0:
        return "Neutral"
    
    bullish_ratio = bullish_count / total_signals
    
    if bullish_ratio >= 0.6:
        return "Bullish"
    elif bullish_ratio <= 0.4:
        return "Bearish" 
    else:
        return "Neutral"


@app.post("/historical-data")
async def get_historical_data(request: HistoricalDataRequest):
    """Get historical gold price data."""
    try:
        collector = GoldDataCollector()
        data = collector.get_gold_prices(
            request.start_date, 
            request.end_date, 
            request.source
        )
        
        # Convert to JSON-serializable format
        result = {
            "data": data.reset_index().to_dict('records'),
            "period": f"{request.start_date} to {request.end_date}",
            "total_records": len(data)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting historical data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")


@app.post("/risk-metrics")
async def calculate_risk_metrics(request: RiskMetricsRequest):
    """Calculate risk metrics for given price series."""
    try:
        # Create price series
        dates = pd.to_datetime(request.dates)
        prices = pd.Series(request.prices, index=dates)
        
        # Calculate risk metrics
        risk_manager = RiskManager()
        report = risk_manager.generate_risk_report(prices)
        
        return {
            "risk_metrics": report,
            "calculation_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Risk calculation failed: {str(e)}")


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Run strategy backtest."""
    try:
        # Get historical data for backtesting
        data = await get_latest_data()
        
        # Create backtest configuration
        config = BacktestConfig(
            initial_capital=request.initial_capital,
            commission=request.commission
        )
        
        # Create strategy based on request
        if request.strategy_type == "ma_crossover":
            short_window = request.parameters.get("short_window", 10)
            long_window = request.parameters.get("long_window", 20)
            strategy = MovingAverageCrossoverStrategy(short_window, long_window)
        else:
            # Default strategy
            strategy = MovingAverageCrossoverStrategy()
        
        # Run backtest
        backtester = Backtester(config)
        results = backtester.run_backtest(data, strategy)
        
        # Convert results to JSON-serializable format
        json_results = {
            "strategy_name": results["strategy_name"],
            "total_return": results["total_return"],
            "annualized_return": results["annualized_return"],
            "volatility": results["volatility"],
            "sharpe_ratio": results["sharpe_ratio"],
            "max_drawdown": results["max_drawdown"]["max_drawdown"],
            "total_trades": results["total_trades"],
            "win_rate": results["win_rate"],
            "final_portfolio_value": results["final_portfolio_value"]
        }
        
        return json_results
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "available_models": [
            "linear",
            "ridge", 
            "lasso",
            "random_forest",
            "xgboost",
            "lightgbm",
            "lstm",
            "ensemble"
        ],
        "default_model": "ensemble"
    }


@app.get("/features")
async def get_feature_info():
    """Get information about available features."""
    return {
        "technical_indicators": [
            "sma_5", "sma_10", "sma_20", "sma_50",
            "ema_12", "ema_26",
            "macd", "macd_signal", "macd_histogram",
            "rsi",
            "bb_upper", "bb_lower", "bb_width", "bb_position"
        ],
        "time_features": [
            "year", "month", "day", "dayofweek", "dayofyear", "quarter",
            "month_sin", "month_cos", "day_sin", "day_cos", "dayofweek_sin", "dayofweek_cos"
        ],
        "lag_features": [
            "Close_lag_1", "Close_lag_2", "Close_lag_3", "Close_lag_5", "Close_lag_10"
        ],
        "rolling_features": [
            "Close_rolling_mean_5", "Close_rolling_std_5",
            "Close_rolling_mean_20", "Close_rolling_std_20"
        ]
    }


@app.post("/train-model")
async def train_model(background_tasks: BackgroundTasks, model_type: str = "random_forest"):
    """Train a new model in the background."""
    
    async def train_task():
        try:
            logger.info(f"Starting training for {model_type} model")
            
            # Get data
            data = await get_latest_data()
            
            # Engineer features
            fe = FeatureEngineer()
            featured_data = fe.create_all_features(data)
            
            # Prepare training data
            X = featured_data.select_dtypes(include=[np.number]).fillna(method='ffill').dropna()
            y = X['Close'].shift(-1).dropna()  # Next day's price
            X = X.iloc[:-1]  # Remove last row to match y
            
            # Create and train model
            model = create_model(model_type)
            model.fit(X, y)
            
            # Save model
            model_dir = Path("data/models")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{model_type}_model.pkl"
            joblib.dump(model, model_path)
            
            logger.info(f"Model {model_type} trained and saved")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    background_tasks.add_task(train_task)
    
    return {
        "message": f"Training {model_type} model started in background",
        "status": "started"
    }


@app.get("/ml-predictions/{days}")
async def ml_predictions(days: int):
    """Get ML predictions for specified days"""
    try:
        # Use auto-trainer for predictions
        predictions = auto_trainer.predict_with_trained_model(horizon_days=days)
        return {
            "predictions": predictions,
            "days": days,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error",
            "days": days
        }

@app.get("/auto-train")
async def auto_train():
    """Trigger automatic ML training on historical data"""
    try:
        # Train models on historical data
        model_paths = auto_trainer.train_prediction_models()
        return {
            "status": "success",
            "message": "Auto-training completed",
            "model_paths": model_paths,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.get("/training-status")
async def training_status():
    """Get current training status and model information"""
    try:
        status = auto_trainer.get_training_status()
        return {
            "status": "success",
            "training_info": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.get("/extended-training")
async def start_extended_training():
    """Start extended ML training with horizons from 1 day to 1 year"""
    try:
        logger.info("🚀 Starting extended ML training (1 day to 1 year)...")
        results = enhanced_trainer.train_extended_models()
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
        
        return {
            "status": "success",
            "message": "Extended ML training completed",
            "models_trained": list(results.keys()),
            "training_summary": results,
            "horizons": "1 day to 1 year",
            "features": "300+ advanced features",
            "training_data": "10 years historical data",
            "algorithms": ["RandomForest", "GradientBoosting", "ExtraTrees", "Ridge", "Lasso", "ElasticNet", "SVR"],
            "note": "Extended models trained on 10 years of historical data with advanced feature engineering"
        }
        
    except Exception as e:
        logger.error(f"Extended training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/start-extended-training-async")
async def start_extended_training_async(background_tasks: BackgroundTasks):
    """Start extended training in background with progress tracking"""
    try:
        if enhanced_trainer.is_training:
            return {
                "status": "already_running",
                "message": "Training is already in progress",
                "progress": enhanced_trainer.get_training_progress()
            }
        
        # Start training in background
        background_tasks.add_task(enhanced_trainer.train_extended_models_with_progress)
        
        return {
            "status": "started",
            "message": "Extended training started in background",
            "estimated_duration": "15-20 minutes",
            "progress_endpoint": "/training-progress"
        }
        
    except Exception as e:
        logger.error(f"Failed to start async training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/training-progress")
async def get_training_progress():
    """Get real-time training progress"""
    try:
        progress = enhanced_trainer.get_training_progress()
        
        return {
            "status": "success",
            "progress": progress
        }
        
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        return {
            "status": "error", 
            "error": str(e),
            "progress": {
                "status": "error",
                "progress_percent": 0,
                "current_step": "Error getting progress"
            }
        }

@app.get("/extended-predictions")
async def get_extended_predictions():
    """Get extended predictions from 1 day to 1 year"""
    try:
        predictions = enhanced_trainer.get_extended_predictions()
        
        if "error" in predictions:
            return {
                "status": "no_models",
                "message": predictions["error"],
                "recommendation": "Run /extended-training first to train 1-year prediction models",
                "available_horizons": []
            }
        
        return {
            "status": "success",
            "predictions": predictions["predictions"],
            "current_analysis": predictions["current_analysis"],
            "notes": predictions["notes"],
            "horizons_available": list(predictions["predictions"].keys()),
            "total_models": len(predictions["predictions"]),
            "prediction_range": "1 day to 1 year"
        }
        
    except Exception as e:
        logger.error(f"Extended predictions error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/prediction-summary")
async def get_prediction_summary():
    """Get summary of all available prediction models and their performance"""
    try:
        import json
        
        # Check for extended training summary
        summary_path = Path("data/models/extended_training_summary.json")
        
        response = {
            "basic_models": {
                "status": "available",
                "horizons": ["1d", "3d", "7d", "14d"],
                "max_horizon": "14 days",
                "features": "50+ technical indicators",
                "training_data": "2 years"
            }
        }
        
        if summary_path.exists():
            with open(summary_path) as f:
                training_summary = json.load(f)
            
            response["extended_models"] = {
                "status": "available",
                "training_summary": training_summary,
                "horizons": list(training_summary.get('model_performance', {}).keys()),
                "max_horizon": "1 year",
                "total_models": len(training_summary.get('model_performance', {})),
                "features": "300+ advanced features",
                "training_data": training_summary.get('data_period', '10 years'),
                "last_trained": training_summary.get('training_completed', 'Unknown'),
                "algorithms": training_summary.get('algorithms_tested', [])
            }
            
            response["recommendation"] = "Use extended models for better long-term accuracy"
        else:
            response["extended_models"] = {
                "status": "not_available",
                "message": "Extended models not trained yet",
                "recommendation": "Run /extended-training to train models for 1 day to 1 year predictions"
            }
            
            response["recommendation"] = "Train extended models for 1-year predictions"
        
        return response
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/extended-predictions/{horizon}")
async def get_extended_prediction_by_horizon(horizon: str):
    """Get prediction for specific extended horizon (1d, 1w, 2w, 1m, 2m, 3m, 6m, 9m, 1y)"""
    try:
        # Validate horizon
        valid_horizons = ['1d', '3d', '5d', '1w', '2w', '3w', '1m', '6w', '2m', '3m', '4m', '6m', '9m', '1y']
        if horizon not in valid_horizons:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid horizon. Valid options: {', '.join(valid_horizons)}"
            )
        
        predictions = enhanced_trainer.get_extended_predictions([horizon])
        
        if "error" in predictions:
            raise HTTPException(status_code=404, detail=predictions["error"])
        
        if horizon not in predictions["predictions"]:
            raise HTTPException(
                status_code=404, 
                detail=f"Model for {horizon} horizon not found. Run /extended-training first."
            )
        
        prediction_data = predictions["predictions"][horizon]
        
        return {
            "status": "success",
            "horizon": horizon,
            "prediction": prediction_data,
            "current_price": predictions["current_analysis"]["current_price"],
            "timestamp": predictions["current_analysis"]["timestamp"],
            "model_info": {
                "training_period": "10 years",
                "features": "300+ advanced indicators",
                "validation": "Time series cross-validation"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Extended prediction error for {horizon}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "status_code": 404}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )