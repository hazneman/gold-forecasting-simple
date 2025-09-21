"""Test Module for Gold Price Forecasting Models

This module contains comprehensive unit tests for the machine learning models
used in gold price forecasting.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    LinearModel, TreeModel, LSTMModel, ModelEnsemble, 
    create_model, BaseModel
)
from src.feature_engineering import FeatureEngineer
from src.data_collection import GoldDataCollector
from src.risk_management import RiskManager
from src.backtesting import Backtester, BacktestConfig, MovingAverageCrossoverStrategy


class TestBaseModel(unittest.TestCase):
    """Test the BaseModel abstract class functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        self.X = pd.DataFrame(
            np.random.randn(len(dates), 5),
            columns=[f'feature_{i}' for i in range(5)],
            index=dates
        )
        self.y = pd.Series(
            np.random.randn(len(dates)).cumsum() + 1800,
            index=dates,
            name='price'
        )
    
    def test_create_model_factory(self):
        """Test the model factory function."""
        # Test linear models
        linear_model = create_model('linear')
        self.assertIsInstance(linear_model, LinearModel)
        
        ridge_model = create_model('ridge', alpha=1.0)
        self.assertIsInstance(ridge_model, LinearModel)
        
        # Test tree models
        rf_model = create_model('random_forest', n_estimators=10)
        self.assertIsInstance(rf_model, TreeModel)
        
        # Test invalid model type
        with self.assertRaises(ValueError):
            create_model('invalid_model')


class TestLinearModel(unittest.TestCase):
    """Test LinearModel implementations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        self.y = pd.Series(np.random.randn(100) + self.X.sum(axis=1))
    
    def test_linear_regression(self):
        """Test linear regression model."""
        model = LinearModel('linear')
        
        # Test fitting
        model.fit(self.X, self.y)
        self.assertTrue(model.is_trained)
        
        # Test prediction
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        
        # Test evaluation
        metrics = model.evaluate(self.X, self.y)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertGreater(metrics['r2'], 0.5)  # Should have reasonable fit
    
    def test_ridge_regression(self):
        """Test ridge regression model."""
        model = LinearModel('ridge', alpha=1.0)
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X)
        metrics = model.evaluate(self.X, self.y)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertIn('r2', metrics)
    
    def test_lasso_regression(self):
        """Test lasso regression model."""
        model = LinearModel('lasso', alpha=0.1)
        model.fit(self.X, self.y)
        
        predictions = model.predict(self.X)
        metrics = model.evaluate(self.X, self.y)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertIn('r2', metrics)
    
    def test_invalid_model_type(self):
        """Test invalid linear model type."""
        with self.assertRaises(ValueError):
            LinearModel('invalid_type')


class TestTreeModel(unittest.TestCase):
    """Test TreeModel implementations."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 5), 
                             columns=[f'feature_{i}' for i in range(5)])
        self.y = pd.Series(np.random.randn(100).cumsum() + 1800)
    
    def test_random_forest(self):
        """Test random forest model."""
        try:
            model = TreeModel('random_forest', n_estimators=10, random_state=42)
            model.fit(self.X, self.y)
            
            self.assertTrue(model.is_trained)
            
            # Test prediction
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.y))
            
            # Test feature importance
            importance = model.get_feature_importance()
            self.assertEqual(len(importance), self.X.shape[1])
            
            # Test evaluation
            metrics = model.evaluate(self.X, self.y)
            self.assertIn('mse', metrics)
            
        except ImportError:
            self.skipTest("Scikit-learn not available")
    
    def test_xgboost_model(self):
        """Test XGBoost model."""
        try:
            model = TreeModel('xgboost', n_estimators=10)
            model.fit(self.X, self.y)
            
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.y))
            
        except ImportError:
            self.skipTest("XGBoost not available")
    
    def test_lightgbm_model(self):
        """Test LightGBM model."""
        try:
            model = TreeModel('lightgbm', n_estimators=10)
            model.fit(self.X, self.y)
            
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.y))
            
        except ImportError:
            self.skipTest("LightGBM not available")


class TestLSTMModel(unittest.TestCase):
    """Test LSTM model implementation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        self.X = pd.DataFrame(
            np.random.randn(200, 3),
            columns=['feature_1', 'feature_2', 'feature_3'],
            index=dates
        )
        self.y = pd.Series(
            np.random.randn(200).cumsum() + 1800,
            index=dates
        )
    
    def test_lstm_model_creation(self):
        """Test LSTM model creation."""
        try:
            model = LSTMModel(sequence_length=30, units=10, layers=1)
            self.assertEqual(model.sequence_length, 30)
            self.assertEqual(model.units, 10)
            self.assertEqual(model.layers, 1)
            
        except ImportError:
            self.skipTest("TensorFlow not available")
    
    def test_lstm_sequence_preparation(self):
        """Test LSTM sequence preparation."""
        try:
            model = LSTMModel(sequence_length=30)
            X_seq, y_seq = model.prepare_sequences(self.X, self.y)
            
            expected_samples = len(self.X) - model.sequence_length
            self.assertEqual(X_seq.shape[0], expected_samples)
            self.assertEqual(len(y_seq), expected_samples)
            self.assertEqual(X_seq.shape[1], model.sequence_length)
            self.assertEqual(X_seq.shape[2], self.X.shape[1])
            
        except ImportError:
            self.skipTest("TensorFlow not available")


class TestModelEnsemble(unittest.TestCase):
    """Test ModelEnsemble functionality."""
    
    def setUp(self):
        """Set up test data and models."""
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        self.y = pd.Series(np.random.randn(100) + self.X.sum(axis=1))
        
        # Create simple models for ensemble
        self.models = [
            LinearModel('linear'),
            LinearModel('ridge', alpha=1.0)
        ]
    
    def test_ensemble_creation(self):
        """Test ensemble creation."""
        ensemble = ModelEnsemble(self.models)
        self.assertEqual(len(ensemble.models), 2)
        self.assertEqual(len(ensemble.weights), 2)
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)
    
    def test_ensemble_with_custom_weights(self):
        """Test ensemble with custom weights."""
        weights = [0.7, 0.3]
        ensemble = ModelEnsemble(self.models, weights)
        self.assertEqual(ensemble.weights, weights)
    
    def test_ensemble_training_and_prediction(self):
        """Test ensemble training and prediction."""
        ensemble = ModelEnsemble(self.models)
        ensemble.fit(self.X, self.y)
        
        self.assertTrue(ensemble.is_trained)
        
        predictions = ensemble.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
    
    def test_ensemble_evaluation(self):
        """Test ensemble evaluation."""
        ensemble = ModelEnsemble(self.models)
        ensemble.fit(self.X, self.y)
        
        results = ensemble.evaluate(self.X, self.y)
        
        # Should have results for ensemble and individual models
        self.assertIn('ensemble', results)
        self.assertIn('linear_regression', results)
        self.assertIn('ridge_regression', results)


class TestFeatureEngineer(unittest.TestCase):
    """Test FeatureEngineer functionality."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', '2021-01-01', freq='D')
        np.random.seed(42)
        self.data = pd.DataFrame({
            'Open': np.random.randn(len(dates)).cumsum() + 1800,
            'High': np.random.randn(len(dates)).cumsum() + 1810,
            'Low': np.random.randn(len(dates)).cumsum() + 1790,
            'Close': np.random.randn(len(dates)).cumsum() + 1800,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
    
    def test_technical_indicators(self):
        """Test technical indicators creation."""
        fe = FeatureEngineer()
        result = fe.create_technical_indicators(self.data)
        
        # Check that new columns were added
        self.assertGreater(result.shape[1], self.data.shape[1])
        
        # Check for specific indicators
        expected_indicators = ['sma_5', 'sma_10', 'rsi', 'macd']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
    
    def test_time_features(self):
        """Test time feature creation."""
        fe = FeatureEngineer()
        result = fe.create_time_features(self.data)
        
        # Check for time features
        time_features = ['year', 'month', 'day', 'dayofweek']
        for feature in time_features:
            self.assertIn(feature, result.columns)
        
        # Check cyclical features
        cyclical_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos']
        for feature in cyclical_features:
            self.assertIn(feature, result.columns)
    
    def test_lag_features(self):
        """Test lag feature creation."""
        fe = FeatureEngineer()
        lags = [1, 2, 5]
        result = fe.create_lag_features(self.data, lags=lags)
        
        for lag in lags:
            self.assertIn(f'Close_lag_{lag}', result.columns)
    
    def test_rolling_features(self):
        """Test rolling feature creation."""
        fe = FeatureEngineer()
        windows = [5, 10]
        result = fe.create_rolling_features(self.data, windows=windows)
        
        for window in windows:
            self.assertIn(f'Close_rolling_mean_{window}', result.columns)
            self.assertIn(f'Close_rolling_std_{window}', result.columns)
    
    def test_complete_feature_engineering(self):
        """Test complete feature engineering pipeline."""
        fe = FeatureEngineer()
        result = fe.create_all_features(self.data)
        
        # Should have significantly more features
        self.assertGreater(result.shape[1], self.data.shape[1] * 3)
        
        # Should have feature names stored
        self.assertTrue(len(fe.feature_names) > 0)


class TestRiskManager(unittest.TestCase):
    """Test RiskManager functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        self.prices = pd.Series(1800 * np.exp(np.cumsum(returns)), index=dates)
        self.returns = self.prices.pct_change().dropna()
    
    def test_risk_manager_creation(self):
        """Test risk manager creation."""
        rm = RiskManager(confidence_level=0.05)
        self.assertEqual(rm.confidence_level, 0.05)
    
    def test_var_calculation(self):
        """Test VaR calculation methods."""
        rm = RiskManager()
        
        # Historical VaR
        var_hist = rm.calculate_var(self.returns, method='historical')
        self.assertIsInstance(var_hist, float)
        self.assertLess(var_hist, 0)  # VaR should be negative
        
        # Parametric VaR
        var_param = rm.calculate_var(self.returns, method='parametric')
        self.assertIsInstance(var_param, float)
        
        # Monte Carlo VaR
        var_mc = rm.calculate_var(self.returns, method='monte_carlo')
        self.assertIsInstance(var_mc, float)
    
    def test_expected_shortfall(self):
        """Test Expected Shortfall calculation."""
        rm = RiskManager()
        es = rm.calculate_expected_shortfall(self.returns)
        
        self.assertIsInstance(es, float)
        self.assertLess(es, 0)  # ES should be negative
    
    def test_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        rm = RiskManager()
        dd_info = rm.calculate_maximum_drawdown(self.prices)
        
        self.assertIn('max_drawdown', dd_info)
        self.assertIn('max_drawdown_date', dd_info)
        self.assertIn('drawdown_series', dd_info)
        
        self.assertLess(dd_info['max_drawdown'], 0)  # Max DD should be negative
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        rm = RiskManager()
        sharpe = rm.calculate_sharpe_ratio(self.returns)
        
        self.assertIsInstance(sharpe, float)
    
    def test_position_sizing(self):
        """Test position sizing calculation."""
        rm = RiskManager()
        position_info = rm.calculate_position_size(
            portfolio_value=100000,
            entry_price=1850,
            stop_loss_price=1800,
            risk_per_trade=0.02
        )
        
        self.assertIn('position_size', position_info)
        self.assertIn('position_value', position_info)
        self.assertIn('risk_amount', position_info)
        
        self.assertGreater(position_info['position_size'], 0)
        self.assertEqual(position_info['risk_amount'], 2000)  # 2% of 100k
    
    def test_risk_report(self):
        """Test comprehensive risk report generation."""
        rm = RiskManager()
        report = rm.generate_risk_report(self.prices)
        
        self.assertIn('return_metrics', report)
        self.assertIn('risk_metrics', report)
        self.assertIn('position_sizing', report)
        
        # Check specific metrics
        self.assertIn('total_return_pct', report['return_metrics'])
        self.assertIn('var_historical_pct', report['risk_metrics'])
        self.assertIn('kelly_criterion', report['position_sizing'])


class TestBacktester(unittest.TestCase):
    """Test Backtester functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2021-01-01', freq='D')
        
        # Create realistic price data
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 1800 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        self.config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
            position_fraction=0.1
        )
    
    def test_backtester_creation(self):
        """Test backtester creation."""
        backtester = Backtester(self.config)
        self.assertEqual(backtester.cash, self.config.initial_capital)
        self.assertEqual(backtester.current_position, 0)
    
    def test_moving_average_strategy(self):
        """Test moving average crossover strategy."""
        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=10)
        signals = strategy.generate_signals(self.data)
        
        self.assertEqual(len(signals), len(self.data))
        self.assertTrue(all(sig in [-1, 0, 1] for sig in signals.values))
    
    def test_backtest_execution(self):
        """Test complete backtest execution."""
        strategy = MovingAverageCrossoverStrategy(short_window=5, long_window=10)
        backtester = Backtester(self.config)
        
        results = backtester.run_backtest(self.data, strategy)
        
        # Check that results contain expected keys
        expected_keys = [
            'strategy_name', 'total_return', 'annualized_return',
            'volatility', 'sharpe_ratio', 'max_drawdown',
            'total_trades', 'win_rate', 'final_portfolio_value'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check that final portfolio value is reasonable
        self.assertGreater(results['final_portfolio_value'], 0)
        
        # Check that we have some trades
        self.assertGreaterEqual(results['total_trades'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete forecasting pipeline."""
        try:
            # 1. Generate sample data
            np.random.seed(42)
            dates = pd.date_range('2020-01-01', '2021-01-01', freq='D')
            
            data = pd.DataFrame({
                'Open': np.random.randn(len(dates)).cumsum() + 1800,
                'High': np.random.randn(len(dates)).cumsum() + 1810,
                'Low': np.random.randn(len(dates)).cumsum() + 1790,
                'Close': np.random.randn(len(dates)).cumsum() + 1800,
                'Volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            
            # 2. Feature engineering
            fe = FeatureEngineer()
            featured_data = fe.create_all_features(data)
            clean_data = featured_data.dropna()
            
            # 3. Prepare modeling data
            X = clean_data.select_dtypes(include=[np.number]).drop('Close', axis=1)
            y = clean_data['Close']
            
            # 4. Train model
            model = create_model('linear')
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # 5. Risk analysis
            rm = RiskManager()
            risk_report = rm.generate_risk_report(clean_data['Close'])
            
            # 6. Backtesting
            strategy = MovingAverageCrossoverStrategy()
            config = BacktestConfig()
            backtester = Backtester(config)
            backtest_results = backtester.run_backtest(clean_data, strategy)
            
            # Verify pipeline completed successfully
            self.assertEqual(len(predictions), len(y_test))
            self.assertIn('return_metrics', risk_report)
            self.assertIn('total_return', backtest_results)
            
            print("✅ Complete pipeline test passed!")
            
        except Exception as e:
            self.fail(f"Pipeline test failed: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_classes = [
        TestBaseModel,
        TestLinearModel,
        TestTreeModel,
        TestLSTMModel,
        TestModelEnsemble,
        TestFeatureEngineer,
        TestRiskManager,
        TestBacktester,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    return result


if __name__ == "__main__":
    print("Running Gold Price Forecasting Model Tests...")
    print("=" * 60)
    
    result = run_tests()
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)