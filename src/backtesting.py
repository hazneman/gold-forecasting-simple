"""Backtesting Module

This module provides comprehensive backtesting framework for gold price forecasting strategies including:
- Historical simulation
- Walk-forward analysis
- Strategy performance evaluation
- Transaction cost modeling
- Portfolio rebalancing
- Out-of-sample testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100000
    commission: float = 0.001  # 0.1% commission
    slippage: float = 0.0005  # 0.05% slippage
    position_size_method: str = 'fixed_fraction'  # 'fixed_fraction', 'kelly', 'equal_weight'
    position_fraction: float = 0.1  # 10% of capital per position
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    max_position_size: float = 0.5  # Maximum 50% in single position
    
    
@dataclass
class Trade:
    """Individual trade record."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    commission: float
    slippage: float
    duration_days: int
    

class PerformanceMetrics:
    """Class for calculating portfolio performance metrics."""
    
    @staticmethod
    def calculate_returns(portfolio_values: pd.Series) -> pd.Series:
        """Calculate portfolio returns."""
        return portfolio_values.pct_change().dropna()
    
    @staticmethod
    def calculate_total_return(portfolio_values: pd.Series) -> float:
        """Calculate total return."""
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
    
    @staticmethod
    def calculate_annualized_return(portfolio_values: pd.Series) -> float:
        """Calculate annualized return."""
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0]
        annualized = (total_return ** (365.25 / days) - 1) * 100
        return annualized
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
        """Calculate volatility."""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Assuming daily returns
        return vol * 100
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - (risk_free_rate / 252)
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: pd.Series) -> Dict[str, Any]:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        return {
            'max_drawdown': max_dd * 100,
            'max_drawdown_date': max_dd_date,
            'drawdown_series': drawdown * 100
        }
    
    @staticmethod
    def calculate_calmar_ratio(portfolio_values: pd.Series) -> float:
        """Calculate Calmar ratio."""
        annual_return = PerformanceMetrics.calculate_annualized_return(portfolio_values) / 100
        max_dd = abs(PerformanceMetrics.calculate_max_drawdown(portfolio_values)['max_drawdown'] / 100)
        
        if max_dd == 0:
            return np.inf
        
        return annual_return / max_dd
    
    @staticmethod
    def calculate_win_rate(trades: List[Trade]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0
        
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        return (winning_trades / len(trades)) * 100
    
    @staticmethod
    def calculate_profit_factor(trades: List[Trade]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0
        
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0
        
        return gross_profit / gross_loss


class Strategy:
    """Base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals.
        
        Args:
            data: DataFrame with price and feature data
            
        Returns:
            Series with signals (1: buy, -1: sell, 0: hold)
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")


class MovingAverageCrossoverStrategy(Strategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, short_window: int = 10, long_window: int = 20):
        super().__init__(f"MA_Crossover_{short_window}_{long_window}")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on moving average crossover."""
        signals = pd.Series(0, index=data.index)
        
        # Calculate moving averages
        short_ma = data['Close'].rolling(window=self.short_window).mean()
        long_ma = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals[short_ma > long_ma] = 1  # Buy signal
        signals[short_ma < long_ma] = -1  # Sell signal
        
        # Only generate signal on crossover
        signals = signals.diff().fillna(0)
        
        return signals


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using Bollinger Bands."""
    
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f"MeanReversion_{window}_{num_std}")
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on mean reversion."""
        signals = pd.Series(0, index=data.index)
        
        # Calculate Bollinger Bands
        ma = data['Close'].rolling(window=self.window).mean()
        std = data['Close'].rolling(window=self.window).std()
        upper_band = ma + (std * self.num_std)
        lower_band = ma - (std * self.num_std)
        
        # Generate signals
        signals[data['Close'] < lower_band] = 1  # Buy when price below lower band
        signals[data['Close'] > upper_band] = -1  # Sell when price above upper band
        
        return signals


class MLPredictionStrategy(Strategy):
    """Strategy based on ML model predictions."""
    
    def __init__(self, model, threshold: float = 0.01):
        super().__init__(f"ML_Prediction_{model.__class__.__name__}")
        self.model = model
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on ML predictions."""
        signals = pd.Series(0, index=data.index)
        
        try:
            # Get predictions
            features = data.select_dtypes(include=[np.number]).fillna(method='ffill')
            predictions = self.model.predict(features)
            
            # Calculate expected returns
            current_prices = data['Close'].values
            expected_returns = (predictions - current_prices) / current_prices
            
            # Generate signals based on threshold
            signals[expected_returns > self.threshold] = 1  # Buy
            signals[expected_returns < -self.threshold] = -1  # Sell
            
        except Exception as e:
            logger.warning(f"Error generating ML signals: {e}")
        
        return signals


class Backtester:
    """Main backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_values = []
        self.positions = []
        self.trades = []
        self.cash = config.initial_capital
        self.current_position = 0
        
    def calculate_position_size(self, price: float, signal: int) -> float:
        """Calculate position size based on configuration."""
        if self.config.position_size_method == 'fixed_fraction':
            max_position_value = self.cash * self.config.position_fraction
            position_size = max_position_value / price
            
        elif self.config.position_size_method == 'equal_weight':
            position_size = self.cash / price
            
        elif self.config.position_size_method == 'kelly':
            # Simplified Kelly criterion (would need historical performance)
            kelly_fraction = 0.1  # Placeholder
            position_size = (self.cash * kelly_fraction) / price
            
        else:
            position_size = self.cash / price
        
        # Apply maximum position size limit
        max_size = (self.cash * self.config.max_position_size) / price
        position_size = min(position_size, max_size)
        
        return position_size
    
    def execute_trade(self, date: datetime, price: float, signal: int) -> None:
        """Execute a trade based on signal."""
        if signal == 0:
            return
        
        # Calculate transaction costs
        commission = self.config.commission
        slippage = self.config.slippage
        
        if signal == 1 and self.current_position <= 0:  # Buy signal
            # Close short position if any
            if self.current_position < 0:
                cost = abs(self.current_position) * price * (1 + slippage + commission)
                self.cash += cost
                pnl = abs(self.current_position) * price - cost
                
                # Record trade
                trade = Trade(
                    entry_date=self.positions[-1]['date'],
                    exit_date=date,
                    entry_price=self.positions[-1]['price'],
                    exit_price=price,
                    quantity=abs(self.current_position),
                    side='short',
                    pnl=pnl,
                    commission=cost * commission,
                    slippage=cost * slippage,
                    duration_days=(date - self.positions[-1]['date']).days
                )
                self.trades.append(trade)
                self.current_position = 0
            
            # Open long position
            position_size = self.calculate_position_size(price, signal)
            cost = position_size * price * (1 + slippage + commission)
            
            if cost <= self.cash:
                self.cash -= cost
                self.current_position = position_size
                self.positions.append({
                    'date': date,
                    'price': price,
                    'quantity': position_size,
                    'side': 'long'
                })
        
        elif signal == -1 and self.current_position >= 0:  # Sell signal
            # Close long position if any
            if self.current_position > 0:
                proceeds = self.current_position * price * (1 - slippage - commission)
                self.cash += proceeds
                cost = self.current_position * self.positions[-1]['price']
                pnl = proceeds - cost
                
                # Record trade
                trade = Trade(
                    entry_date=self.positions[-1]['date'],
                    exit_date=date,
                    entry_price=self.positions[-1]['price'],
                    exit_price=price,
                    quantity=self.current_position,
                    side='long',
                    pnl=pnl,
                    commission=proceeds * commission,
                    slippage=proceeds * slippage,
                    duration_days=(date - self.positions[-1]['date']).days
                )
                self.trades.append(trade)
                self.current_position = 0
            
            # Open short position
            position_size = self.calculate_position_size(price, signal)
            proceeds = position_size * price * (1 - slippage - commission)
            
            self.cash += proceeds
            self.current_position = -position_size
            self.positions.append({
                'date': date,
                'price': price,
                'quantity': -position_size,
                'side': 'short'
            })
    
    def calculate_portfolio_value(self, date: datetime, price: float) -> float:
        """Calculate current portfolio value."""
        position_value = 0
        if self.current_position != 0:
            if self.current_position > 0:  # Long position
                position_value = self.current_position * price
            else:  # Short position
                # For short positions, the value is the cash received minus current cost to cover
                position_value = -abs(self.current_position) * price
        
        return self.cash + position_value
    
    def run_backtest(self, data: pd.DataFrame, strategy: Strategy) -> Dict[str, Any]:
        """Run the backtest."""
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Initialize portfolio tracking
        portfolio_values = []
        dates = []
        
        # Run through each day
        for date, row in data.iterrows():
            price = row['Close']
            signal = signals.get(date, 0)
            
            # Execute trade if signal exists
            if signal != 0:
                self.execute_trade(date, price, signal)
            
            # Calculate portfolio value
            portfolio_value = self.calculate_portfolio_value(date, price)
            portfolio_values.append(portfolio_value)
            dates.append(date)
        
        # Create portfolio value series
        portfolio_series = pd.Series(portfolio_values, index=dates)
        
        # Calculate performance metrics
        returns = PerformanceMetrics.calculate_returns(portfolio_series)
        
        results = {
            'strategy_name': strategy.name,
            'portfolio_values': portfolio_series,
            'trades': self.trades,
            'total_trades': len(self.trades),
            'final_portfolio_value': portfolio_values[-1],
            'total_return': PerformanceMetrics.calculate_total_return(portfolio_series),
            'annualized_return': PerformanceMetrics.calculate_annualized_return(portfolio_series),
            'volatility': PerformanceMetrics.calculate_volatility(returns),
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns, self.config.risk_free_rate),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(portfolio_series),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(portfolio_series),
            'win_rate': PerformanceMetrics.calculate_win_rate(self.trades),
            'profit_factor': PerformanceMetrics.calculate_profit_factor(self.trades)
        }
        
        logger.info(f"Backtest completed for {strategy.name}")
        return results
    
    def reset(self):
        """Reset backtester for new test."""
        self.cash = self.config.initial_capital
        self.current_position = 0
        self.portfolio_values = []
        self.positions = []
        self.trades = []


class WalkForwardAnalysis:
    """Walk-forward analysis for strategy validation."""
    
    def __init__(self, training_period: int = 252, testing_period: int = 63):
        """Initialize walk-forward analysis.
        
        Args:
            training_period: Number of days for training
            testing_period: Number of days for testing
        """
        self.training_period = training_period
        self.testing_period = testing_period
    
    def run_analysis(self, data: pd.DataFrame, 
                    strategy_factory: Callable,
                    config: BacktestConfig) -> Dict[str, Any]:
        """Run walk-forward analysis."""
        results = []
        total_periods = len(data) // self.testing_period
        
        for i in range(total_periods):
            # Define training and testing windows
            start_train = i * self.testing_period
            end_train = start_train + self.training_period
            start_test = end_train
            end_test = start_test + self.testing_period
            
            if end_test > len(data):
                break
            
            # Get data slices
            train_data = data.iloc[start_train:end_train]
            test_data = data.iloc[start_test:end_test]
            
            # Create and train strategy
            strategy = strategy_factory(train_data)
            
            # Run backtest on test data
            backtester = Backtester(config)
            period_results = backtester.run_backtest(test_data, strategy)
            period_results['period'] = i
            period_results['train_start'] = train_data.index[0]
            period_results['train_end'] = train_data.index[-1]
            period_results['test_start'] = test_data.index[0]
            period_results['test_end'] = test_data.index[-1]
            
            results.append(period_results)
        
        # Aggregate results
        summary = self._aggregate_results(results)
        
        return {
            'period_results': results,
            'summary': summary
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate results across all periods."""
        if not results:
            return {}
        
        metrics = ['total_return', 'volatility', 'sharpe_ratio', 'win_rate']
        aggregated = {}
        
        for metric in metrics:
            values = [r[metric] for r in results if metric in r]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        return aggregated


def main():
    """Example usage of the backtesting framework."""
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    
    # Simulate price data with some trend and noise
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = 1800 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Create backtesting configuration
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        position_fraction=0.1
    )
    
    # Create strategies
    ma_strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=20)
    mr_strategy = MeanReversionStrategy(window=20, num_std=2.0)
    
    # Run backtests
    strategies = [ma_strategy, mr_strategy]
    
    for strategy in strategies:
        backtester = Backtester(config)
        results = backtester.run_backtest(data, strategy)
        
        print(f"\n{strategy.name} Results:")
        print(f"  Total Return: {results['total_return']:.2f}%")
        print(f"  Annualized Return: {results['annualized_return']:.2f}%")
        print(f"  Volatility: {results['volatility']:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']['max_drawdown']:.2f}%")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.2f}%")


if __name__ == "__main__":
    main()