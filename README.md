# Gold Price Forecasting System 🥇

**✅ SYSTEM READY - NO SETUP REQUIRED!**

A comprehensive machine learning system for predicting gold prices using multiple algorithms and real-time data sources.

## 🚀 INSTANT START

The system is **fully configured and ready to use**! Choose your preferred way to get gold price forecasts:

### Quick Options:

**🎯 Interactive Menu (Recommended)**
```bash
./start.sh
```

**⚡ Quick Test & Forecast**
```bash
/Users/hasannumanoglu/Documents/SoftDev/GoldPriceForecasting/.venv/bin/python test_basic.py
```

**🌐 Start API Server**
```bash
/Users/hasannumanoglu/Documents/SoftDev/GoldPriceForecasting/.venv/bin/python -m uvicorn api.fastapi_app:app --reload
```
Then visit: http://localhost:8000/docs

**📊 Full Forecasting System**
```bash
/Users/hasannumanoglu/Documents/SoftDev/GoldPriceForecasting/.venv/bin/python main.py
```

### ✅ What's Already Working:

- **Data Collection**: Yahoo Finance integration (no API key needed)
- **Machine Learning**: Linear, Tree-based, Neural Networks  
- **REST API**: FastAPI with interactive documentation
- **Visualization**: Interactive charts and dashboards
- **Technical Analysis**: 20+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Risk Management**: VaR, portfolio analysis
- **No Database Required**: File-based storage system

## 🎯 Project Overview

This project provides a complete pipeline for gold price forecasting including:

- **Data Collection**: Real-time and historical gold price data from multiple sources
- **Feature Engineering**: Technical indicators, time-based features, and statistical measures
- **Machine Learning Models**: Linear models, tree-based models, neural networks, and ensemble methods
- **Risk Management**: VaR, Expected Shortfall, drawdown analysis, and position sizing
- **Backtesting**: Strategy performance evaluation with transaction costs
- **API Service**: FastAPI-based REST API for predictions and analysis
- **Visualization**: Comprehensive charts and interactive dashboards

## 📁 Project Structure

```
GoldPriceForecasting/
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── data_collection.py        # Data collection from various sources
│   ├── feature_engineering.py    # Technical indicators and feature creation
│   ├── models.py                 # ML models (Linear, Tree-based, LSTM, Ensemble)
│   ├── visualization.py          # Plotting and visualization tools
│   ├── risk_management.py        # Risk metrics and portfolio management
│   └── backtesting.py           # Strategy backtesting framework
│
├── api/                          # FastAPI application
│   ├── __init__.py
│   └── fastapi_app.py           # REST API endpoints
│
├── config/                       # Configuration settings
│   ├── __init__.py
│   └── settings.py              # Application configuration
│
├── notebooks/                    # Jupyter notebooks
│   └── exploration.ipynb        # Data exploration and analysis
│
├── data/                        # Data storage
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed data files
│   └── models/                  # Saved model files
│
├── tests/                       # Unit tests
│   └── test_models.py          # Model testing
│
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── main.py                     # Main application entry point
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd GoldPriceForecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```python
# Import the main modules
from src.data_collection import GoldDataCollector
from src.feature_engineering import FeatureEngineer
from src.models import create_model
from src.visualization import GoldPriceVisualizer

# Collect data
collector = GoldDataCollector()
data = collector.get_gold_prices("2020-01-01", "2023-01-01")

# Engineer features
fe = FeatureEngineer()
featured_data = fe.create_all_features(data)

# Train model
model = create_model("random_forest")
X = featured_data.select_dtypes(include=['number']).drop('Close', axis=1)
y = featured_data['Close']
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Visualize results
viz = GoldPriceVisualizer()
viz.plot_predictions_vs_actual(y, {"RF": predictions})
```

### 3. Running the API

```bash
# Start the FastAPI server
python -m uvicorn api.fastapi_app:app --reload

# Access the API documentation
# Open http://localhost:8000/docs in your browser
```

### 4. Running the Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/exploration.ipynb
```

## 📊 Features

### Data Collection
- **Yahoo Finance**: Real-time and historical gold price data
- **Alpha Vantage**: Premium financial data (API key required)
- **FRED**: Economic indicators and macroeconomic data
- **Custom data sources**: Extensible data collection framework

### Feature Engineering
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, Volume indicators
- **Time Features**: Cyclical encoding of date/time components
- **Lag Features**: Historical price lags for time series patterns
- **Rolling Statistics**: Moving averages, volatility, skewness, kurtosis
- **Price Features**: Returns, momentum, volatility measures

### Machine Learning Models
- **Linear Models**: Linear Regression, Ridge, Lasso
- **Tree-based Models**: Random Forest, XGBoost, LightGBM
- **Neural Networks**: LSTM, GRU with TensorFlow/Keras
- **Ensemble Methods**: Weighted averaging of multiple models
- **Custom Models**: Extensible framework for new model types

### Risk Management
- **Value at Risk (VaR)**: Historical, Parametric, Monte Carlo methods
- **Expected Shortfall**: Conditional VaR for tail risk
- **Drawdown Analysis**: Maximum drawdown and recovery periods
- **Position Sizing**: Kelly Criterion, fixed fraction, risk-based sizing
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios

### Backtesting
- **Strategy Framework**: Base classes for custom trading strategies
- **Built-in Strategies**: Moving average crossover, mean reversion, ML-based
- **Transaction Costs**: Commission and slippage modeling
- **Walk-Forward Analysis**: Out-of-sample validation
- **Performance Analytics**: Comprehensive performance metrics

## 🔧 Configuration

The application uses a centralized configuration system in `config/settings.py`. Key configuration options:

```python
# Data sources
DATA_SOURCE = "yahoo"  # "yahoo", "alpha_vantage"
ALPHA_VANTAGE_API_KEY = "your_api_key"

# Model settings
DEFAULT_MODEL = "ensemble"
TRAINING_PERIOD = 252  # days
TEST_SIZE = 0.2

# Risk management
CONFIDENCE_LEVEL = 0.05  # 5% VaR
MAX_POSITION_SIZE = 0.5  # 50% max position
COMMISSION = 0.001  # 0.1% commission

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
```

Environment variables can be used to override settings:
```bash
export ALPHA_VANTAGE_API_KEY="your_api_key"
export API_PORT=8080
```

## 📈 API Endpoints

The FastAPI application provides the following endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Make price predictions
- `GET /current-price` - Get current gold price
- `POST /historical-data` - Fetch historical data
- `POST /risk-metrics` - Calculate risk metrics
- `POST /backtest` - Run strategy backtests
- `GET /models` - List available models
- `POST /train-model` - Train new models

### Example API Usage

```python
import requests

# Get current price
response = requests.get("http://localhost:8000/current-price")
current_price = response.json()

# Make prediction
features = {"sma_20": 1850, "rsi": 45, "volume": 5000}
response = requests.post("http://localhost:8000/predict", 
                        json={"features": features})
prediction = response.json()

# Run backtest
backtest_config = {
    "strategy_type": "ma_crossover",
    "parameters": {"short_window": 10, "long_window": 20},
    "initial_capital": 100000
}
response = requests.post("http://localhost:8000/backtest", 
                        json=backtest_config)
results = response.json()
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_models.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📚 Dependencies

Key dependencies include:
- **Data & ML**: pandas, numpy, scikit-learn, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn, plotly
- **API**: FastAPI, uvicorn, pydantic
- **Deep Learning**: tensorflow (optional)
- **Technical Analysis**: TA-Lib
- **Development**: pytest, black, flake8, mypy

See `requirements.txt` for complete list.

## 🚧 Development

### Code Style
```bash
# Format code
black src/ tests/ api/

# Lint code
flake8 src/ tests/ api/

# Type checking
mypy src/
```

### Adding New Models
```python
from src.models import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__("my_custom_model")
        # Initialize your model
        
    def fit(self, X, y):
        # Training logic
        self.is_trained = True
        
    def predict(self, X):
        # Prediction logic
        return predictions
```

### Adding New Strategies
```python
from src.backtesting import Strategy

class MyStrategy(Strategy):
    def __init__(self, param1, param2):
        super().__init__("my_strategy")
        self.param1 = param1
        self.param2 = param2
        
    def generate_signals(self, data):
        # Signal generation logic
        signals = pd.Series(0, index=data.index)
        # Your logic here
        return signals
```

## 📊 Performance

Expected performance characteristics:
- **Data Collection**: ~1-2 seconds for 1 year of daily data
- **Feature Engineering**: ~100ms for 1000 data points
- **Model Training**: 
  - Linear models: ~10ms
  - Tree models: ~1-5 seconds
  - LSTM: ~30-60 seconds
- **Prediction**: ~1-10ms per prediction
- **API Response**: ~50-200ms typical

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Trading and investing in financial markets involves substantial risk of loss. Past performance does not guarantee future results.

## 🙏 Acknowledgments

- Yahoo Finance for providing free financial data
- The open-source community for excellent ML libraries
- Contributors and researchers in quantitative finance

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `notebooks/` directory
- Review the API documentation at `/docs` endpoint

---

**Happy Forecasting! 📈✨**