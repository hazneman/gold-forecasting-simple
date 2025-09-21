"""Risk Management Module

This module provides risk management tools for gold price forecasting including:
- Portfolio risk metrics calculation
- Value at Risk (VaR) and Expected Shortfall (ES)
- Position sizing algorithms
- Stop-loss and take-profit strategies
- Drawdown analysis
- Risk-adjusted performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class RiskManager:
    """Class for managing trading and investment risks."""
    
    def __init__(self, confidence_level: float = 0.05):
        """Initialize the risk manager.
        
        Args:
            confidence_level: Confidence level for VaR calculations (default 5%)
        """
        self.confidence_level = confidence_level
        self.alpha = confidence_level
        
    def calculate_returns(self, prices: pd.Series, 
                         return_type: str = 'simple') -> pd.Series:
        """Calculate returns from price series.
        
        Args:
            prices: Price series
            return_type: Type of returns ('simple' or 'log')
            
        Returns:
            Series of returns
        """
        if return_type == 'simple':
            returns = prices.pct_change().dropna()
        elif return_type == 'log':
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError("return_type must be 'simple' or 'log'")
        
        return returns
    
    def calculate_volatility(self, returns: pd.Series, 
                           window: Optional[int] = None,
                           annualize: bool = True) -> float:
        """Calculate volatility of returns.
        
        Args:
            returns: Return series
            window: Rolling window for calculation (if None, use all data)
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility measure
        """
        if window is not None:
            vol = returns.rolling(window=window).std()
        else:
            vol = returns.std()
        
        if annualize:
            # Assuming daily data, annualize by sqrt(252)
            vol = vol * np.sqrt(252)
        
        return vol
    
    def calculate_var(self, returns: pd.Series, 
                     method: str = 'historical',
                     window: Optional[int] = None) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns: Return series
            method: VaR calculation method ('historical', 'parametric', 'monte_carlo')
            window: Rolling window for calculation
            
        Returns:
            VaR value
        """
        if window is not None:
            returns = returns.tail(window)
        
        if method == 'historical':
            var = np.percentile(returns, self.alpha * 100)
        
        elif method == 'parametric':
            # Assume normal distribution
            mean = returns.mean()
            std = returns.std()
            var = stats.norm.ppf(self.alpha, mean, std)
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            mean = returns.mean()
            std = returns.std()
            simulated_returns = np.random.normal(mean, std, n_simulations)
            var = np.percentile(simulated_returns, self.alpha * 100)
        
        else:
            raise ValueError("method must be 'historical', 'parametric', or 'monte_carlo'")
        
        return var
    
    def calculate_expected_shortfall(self, returns: pd.Series,
                                   window: Optional[int] = None) -> float:
        """Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Return series
            window: Rolling window for calculation
            
        Returns:
            Expected Shortfall value
        """
        if window is not None:
            returns = returns.tail(window)
        
        var = self.calculate_var(returns, method='historical')
        
        # Calculate average of returns below VaR threshold
        shortfall_returns = returns[returns <= var]
        
        if len(shortfall_returns) > 0:
            es = shortfall_returns.mean()
        else:
            es = var  # Fallback to VaR if no shortfall returns
        
        return es
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate maximum drawdown and related metrics.
        
        Args:
            prices: Price series
            
        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate cumulative returns
        cumulative = (1 + self.calculate_returns(prices)).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find peak and trough dates
        peak_date = running_max.loc[:max_dd_date].idxmax()
        
        # Recovery date (first date after trough where drawdown returns to 0)
        recovery_mask = drawdown.loc[max_dd_date:] >= 0
        if recovery_mask.any():
            recovery_date = recovery_mask.idxmax()
            recovery_period = (recovery_date - max_dd_date).days
        else:
            recovery_date = None
            recovery_period = None
        
        # Drawdown duration
        drawdown_duration = (max_dd_date - peak_date).days
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_date': max_dd_date,
            'peak_date': peak_date,
            'recovery_date': recovery_date,
            'drawdown_duration_days': drawdown_duration,
            'recovery_period_days': recovery_period,
            'drawdown_series': drawdown
        }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        # Convert risk-free rate to same frequency as returns
        rf_rate_period = risk_free_rate / 252  # Assuming daily returns
        
        excess_returns = returns - rf_rate_period
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        
        return sharpe
    
    def calculate_calmar_ratio(self, returns: pd.Series, 
                             prices: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Return series
            prices: Price series
            
        Returns:
            Calmar ratio
        """
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        max_dd = abs(self.calculate_maximum_drawdown(prices)['max_drawdown'])
        
        if max_dd == 0:
            return np.inf
        
        calmar = annual_return / max_dd
        return calmar
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.02,
                              target_return: Optional[float] = None) -> float:
        """Calculate Sortino ratio.
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate
            target_return: Target return (if None, use risk-free rate)
            
        Returns:
            Sortino ratio
        """
        if target_return is None:
            target_return = risk_free_rate / 252
        
        excess_returns = returns - target_return
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        annual_excess_return = excess_returns.mean() * 252
        
        sortino = annual_excess_return / downside_deviation
        return sortino
    
    def calculate_position_size(self, 
                              portfolio_value: float,
                              entry_price: float,
                              stop_loss_price: float,
                              risk_per_trade: float = 0.02) -> Dict[str, float]:
        """Calculate position size based on risk management rules.
        
        Args:
            portfolio_value: Total portfolio value
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            risk_per_trade: Risk per trade as fraction of portfolio (default 2%)
            
        Returns:
            Dictionary with position sizing information
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        # Calculate maximum risk amount
        max_risk_amount = portfolio_value * risk_per_trade
        
        # Calculate position size
        if risk_per_share > 0:
            position_size = max_risk_amount / risk_per_share
            position_value = position_size * entry_price
            position_percentage = (position_value / portfolio_value) * 100
        else:
            position_size = 0
            position_value = 0
            position_percentage = 0
        
        return {
            'position_size': position_size,
            'position_value': position_value,
            'position_percentage': position_percentage,
            'risk_amount': max_risk_amount,
            'risk_per_share': risk_per_share
        }
    
    def calculate_kelly_criterion(self, returns: pd.Series) -> float:
        """Calculate Kelly criterion for optimal bet sizing.
        
        Args:
            returns: Return series
            
        Returns:
            Kelly fraction
        """
        # Separate wins and losses
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0
        
        # Calculate win rate and average win/loss
        win_rate = len(wins) / len(returns)
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_fraction = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly_fraction = 0
        
        # Cap Kelly fraction to reasonable limits
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        return kelly_fraction
    
    def generate_risk_report(self, prices: pd.Series, 
                           benchmark_prices: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate comprehensive risk report.
        
        Args:
            prices: Price series
            benchmark_prices: Benchmark price series for comparison
            
        Returns:
            Dictionary with risk metrics
        """
        returns = self.calculate_returns(prices)
        
        # Basic risk metrics
        volatility = self.calculate_volatility(returns)
        var_hist = self.calculate_var(returns, method='historical')
        var_param = self.calculate_var(returns, method='parametric')
        es = self.calculate_expected_shortfall(returns)
        
        # Performance metrics
        sharpe = self.calculate_sharpe_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns, prices)
        sortino = self.calculate_sortino_ratio(returns)
        
        # Drawdown analysis
        drawdown_metrics = self.calculate_maximum_drawdown(prices)
        
        # Kelly criterion
        kelly = self.calculate_kelly_criterion(returns)
        
        # Return statistics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        annual_return = ((1 + returns.mean()) ** 252 - 1) * 100
        
        report = {
            'return_metrics': {
                'total_return_pct': total_return,
                'annualized_return_pct': annual_return,
                'volatility_pct': volatility * 100,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar
            },
            'risk_metrics': {
                'var_historical_pct': var_hist * 100,
                'var_parametric_pct': var_param * 100,
                'expected_shortfall_pct': es * 100,
                'max_drawdown_pct': drawdown_metrics['max_drawdown'] * 100,
                'drawdown_duration_days': drawdown_metrics['drawdown_duration_days']
            },
            'position_sizing': {
                'kelly_criterion': kelly
            }
        }
        
        # Add benchmark comparison if provided
        if benchmark_prices is not None:
            benchmark_returns = self.calculate_returns(benchmark_prices)
            
            # Calculate beta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
            
            # Calculate alpha
            benchmark_annual_return = ((1 + benchmark_returns.mean()) ** 252 - 1)
            alpha = annual_return / 100 - beta * benchmark_annual_return
            
            # Tracking error
            tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
            
            # Information ratio
            excess_return = annual_return / 100 - benchmark_annual_return
            information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
            
            report['benchmark_comparison'] = {
                'beta': beta,
                'alpha': alpha,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            }
        
        return report
    
    def set_stop_loss_take_profit(self, 
                                 entry_price: float,
                                 stop_loss_pct: float = 0.05,
                                 take_profit_pct: float = 0.10,
                                 position_type: str = 'long') -> Dict[str, float]:
        """Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage  
            position_type: 'long' or 'short'
            
        Returns:
            Dictionary with stop loss and take profit levels
        """
        if position_type == 'long':
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
        elif position_type == 'short':
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        else:
            raise ValueError("position_type must be 'long' or 'short'")
        
        risk_reward_ratio = take_profit_pct / stop_loss_pct
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_loss_pct': stop_loss_pct * 100,
            'take_profit_pct': take_profit_pct * 100,
            'risk_reward_ratio': risk_reward_ratio
        }


def main():
    """Example usage of the risk manager."""
    # Generate sample price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    
    # Simulate price path
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = pd.Series(1800 * np.exp(np.cumsum(returns)), index=dates)
    
    # Create risk manager
    risk_manager = RiskManager()
    
    # Generate risk report
    report = risk_manager.generate_risk_report(prices)
    
    print("Risk Management Report")
    print("=" * 50)
    
    print("\nReturn Metrics:")
    for key, value in report['return_metrics'].items():
        print(f"  {key}: {value:.2f}")
    
    print("\nRisk Metrics:")
    for key, value in report['risk_metrics'].items():
        print(f"  {key}: {value:.2f}")
    
    print("\nPosition Sizing:")
    for key, value in report['position_sizing'].items():
        print(f"  {key}: {value:.4f}")
    
    # Example position sizing
    position_info = risk_manager.calculate_position_size(
        portfolio_value=100000,
        entry_price=1850,
        stop_loss_price=1800,
        risk_per_trade=0.02
    )
    
    print(f"\nPosition Sizing Example:")
    print(f"  Position size: {position_info['position_size']:.2f} units")
    print(f"  Position value: ${position_info['position_value']:.2f}")
    print(f"  Risk amount: ${position_info['risk_amount']:.2f}")


if __name__ == "__main__":
    main()