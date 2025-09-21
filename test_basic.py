"""
Simple Gold Price Data Test
Testing basic functionality without complex dependencies
"""

import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that basic imports work"""
    print("Testing basic imports...")
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✓ yfinance imported successfully")
    except ImportError as e:
        print(f"✗ yfinance import failed: {e}")
        return False
    
    return True

def test_data_collection():
    """Test basic data collection"""
    print("\nTesting Yahoo Finance data collection...")
    
    try:
        import yfinance as yf
        
        # Test with a major stock first to verify yfinance works
        test_ticker = yf.Ticker("AAPL")
        
        print("Testing yfinance with AAPL...")
        test_data = test_ticker.history(period="5d")
        
        if test_data.empty:
            print("✗ yfinance not working - cannot fetch AAPL data")
            return False
        
        print(f"✓ yfinance working - fetched {len(test_data)} days of AAPL data")
        
        # Now try gold-related ticker
        print("Testing gold-related ticker...")
        gold_ticker = yf.Ticker("GLD")  # SPDR Gold Trust ETF
        
        gold_data = gold_ticker.history(period="5d")
        
        if not gold_data.empty:
            print(f"✓ Successfully fetched {len(gold_data)} days of gold ETF data")
            print(f"Latest GLD price: ${gold_data['Close'].iloc[-1]:.2f}")
            print(f"Date range: {gold_data.index[0].date()} to {gold_data.index[-1].date()}")
            return True
        else:
            # Try another gold ticker
            print("Trying alternative gold ticker (IAU)...")
            gold_ticker = yf.Ticker("IAU")  # iShares Gold Trust
            gold_data = gold_ticker.history(period="5d")
            
            if not gold_data.empty:
                print(f"✓ Successfully fetched {len(gold_data)} days of gold ETF data")
                print(f"Latest IAU price: ${gold_data['Close'].iloc[-1]:.2f}")
                return True
            else:
                print("✗ No gold-related data received")
                return False
            
    except Exception as e:
        print(f"✗ Data collection failed: {e}")
        return False

def test_basic_predictions():
    """Test a simple prediction model"""
    print("\nTesting basic prediction model...")
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        import yfinance as yf
        
        # Get some real data using GLD ETF
        gold_ticker = yf.Ticker("GLD")
        
        print("Fetching GLD ETF data for prediction model...")
        gold_data = gold_ticker.history(period="3mo")  # 3 months of data
        
        if gold_data.empty:
            # Try IAU if GLD doesn't work
            gold_ticker = yf.Ticker("IAU")
            gold_data = gold_ticker.history(period="3mo")
            
            if gold_data.empty:
                print("✗ No data available for prediction test")
                return False
        
        # Create simple features
        gold_data['Price_Lag1'] = gold_data['Close'].shift(1)
        gold_data['Price_Lag2'] = gold_data['Close'].shift(2)
        gold_data['MA_5'] = gold_data['Close'].rolling(window=5).mean()
        gold_data['Returns'] = gold_data['Close'].pct_change()
        
        # Clean data
        gold_data = gold_data.dropna()
        
        if len(gold_data) < 10:
            print("✗ Not enough data for prediction test")
            return False
        
        # Prepare features and target
        features = ['Price_Lag1', 'Price_Lag2', 'MA_5', 'Returns']
        X = gold_data[features]
        y = gold_data['Close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train simple model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate simple accuracy metric
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        print(f"✓ Simple linear model trained successfully")
        print(f"Model MAPE: {mape:.2f}%")
        print(f"Latest prediction: ${predictions[-1]:.2f}")
        print(f"Actual price: ${y_test.iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("GOLD PRICE FORECASTING - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test imports
    if not test_basic_imports():
        print("\n❌ Basic imports test failed")
        return
    
    # Test data collection
    if not test_data_collection():
        print("\n❌ Data collection test failed")
        return
    
    # Test basic predictions
    if not test_basic_predictions():
        print("\n❌ Prediction test failed")
        return
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("The gold price forecasting system is ready to run!")
    print("=" * 60)
    
    # Show next steps
    print("\nNext steps:")
    print("1. Run 'python main.py' for full forecasting functionality")
    print("2. Run 'python -m uvicorn api.fastapi_app:app --reload' to start the API server")
    print("3. Open notebooks/exploration.ipynb for interactive analysis")

if __name__ == "__main__":
    main()