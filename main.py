"""
Gold Price Forecasting Model
Main entry point for the application
"""

import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_collection import GoldDataCollector
from src.feature_engineering import FeatureEngineer
from src.models import GoldPriceForecaster
from src.visualization import GoldPriceVisualizer
from src.backtesting import Backtester
from config.settings import Config

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("GOLD PRICE FORECASTING SYSTEM")
    print("=" * 60)
    print()
    
    # Load configuration
    config = Config()
    
    # Initialize components
    print("Initializing components...")
    collector = GoldDataCollector(api_key=config.METALS_API_KEY)
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data
    
    # Collect data
    print(f"Collecting gold price data from {start_date.date()} to {end_date.date()}...")
    gold_prices = collector.get_gold_prices(
        start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')
    )
    
    print("Collecting economic indicators...")
    economic_data = collector.get_economic_indicators(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Merge data
    import pandas as pd
    data = pd.merge(gold_prices, economic_data, 
                    left_index=True, right_index=True, how='inner')
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Feature engineering
    print("\nEngineering features...")
    engineer = FeatureEngineer(data)
    data = engineer.create_technical_indicators()
    data = engineer.create_lag_features(['Close', 'Volume'])
    data = engineer.create_interaction_features()
    
    print(f"Total features created: {len(data.columns)}")
    
    # Prepare for modeling
    print("\nPreparing data for modeling...")
    forecaster = GoldPriceForecaster()
    X_train, X_test, y_train, y_test = forecaster.prepare_data(data)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train models
    print("\n" + "=" * 40)
    print("TRAINING MODELS")
    print("=" * 40)
    
    print("\n1. Training Random Forest...")
    forecaster.train_random_forest(X_train, y_train)
    
    print("2. Training XGBoost...")
    forecaster.train_xgboost(X_train, y_train)
    
    # Evaluate models
    print("\n" + "=" * 40)
    print("MODEL EVALUATION")
    print("=" * 40)
    
    results = forecaster.evaluate_models(X_test, y_test)
    
    # Display results
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Performance:")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"  {metric:10s}: {value:.4f}")
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['RMSE'])[0]
    print(f"\nBest performing model: {best_model}")
    
    # Make predictions for next day
    print("\n" + "=" * 40)
    print("NEXT DAY PREDICTION")
    print("=" * 40)
    
    try:
        # Get current gold price from API
        current_price = collector.get_realtime_gold_price()
        if current_price:
            print(f"Current Gold Price: ${current_price:.2f}")
        
        # Make prediction
        last_features = data.drop(columns=['Close']).iloc[-1:].fillna(method='ffill')
        predicted_price = forecaster.predict(last_features, model_name=best_model)[0]
        print(f"Predicted Next Price: ${predicted_price:.2f}")
        
        if current_price:
            change = ((predicted_price - current_price) / current_price) * 100
            print(f"Expected Change: {change:+.2f}%")
            
            if change > 1:
                print("Signal: BUY")
            elif change < -1:
                print("Signal: SELL")
            else:
                print("Signal: HOLD")
    except Exception as e:
        print(f"Could not make real-time prediction: {e}")
    
    # Backtesting
    print("\n" + "=" * 40)
    print("BACKTESTING RESULTS")
    print("=" * 40)
    
    backtester = Backtester(forecaster.models[best_model])
    backtest_results = backtester.run_backtest(
        data.dropna(), 
        data.drop(columns=['Close']).columns.tolist()
    )
    
    print(f"Total Return: {backtest_results['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
    print(f"Final Portfolio Value: ${backtest_results['final_value']:.2f}")
    
    # Visualizations
    print("\n" + "=" * 40)
    print("GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    visualizer = GoldPriceVisualizer()
    
    # Save predictions
    predictions = forecaster.predict(X_test, best_model)
    
    # Create and save plots
    print("Creating price history plot...")
    fig = visualizer.plot_price_history(
        data.iloc[-len(y_test):], 
        pd.Series(predictions, index=y_test.index)
    )
    fig.write_html("output/gold_price_forecast.html")
    print("Saved to: output/gold_price_forecast.html")
    
    # Save model
    print("\nSaving best model...")
    import joblib
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(forecaster.models[best_model], f"data/models/{best_model}_model.pkl")
    joblib.dump(forecaster.scaler, "data/models/scaler.pkl")
    print(f"Model saved to: data/models/{best_model}_model.pkl")
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")