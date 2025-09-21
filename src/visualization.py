"""Visualization Module

This module provides comprehensive visualization tools for gold price forecasting including:
- Price trend analysis
- Technical indicator visualization
- Model performance visualization
- Feature importance plots
- Prediction vs actual comparisons
- Interactive dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Optional, Any, Tuple
import warnings
import logging

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldPriceVisualizer:
    """Class for creating visualizations for gold price analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_price_trend(self, df: pd.DataFrame, 
                        price_col: str = 'Close',
                        title: str = "Gold Price Trend") -> None:
        """Plot gold price trend over time.
        
        Args:
            df: DataFrame with price data
            price_col: Name of the price column
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(df.index, df[price_col], linewidth=2, color=self.colors[0])
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        x_numeric = range(len(df))
        z = np.polyfit(x_numeric, df[price_col], 1)
        p = np.poly1d(z)
        ax.plot(df.index, p(x_numeric), "--", alpha=0.7, color=self.colors[1],
                label=f'Trend (slope: {z[0]:.2f})')
        
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        logger.info("Price trend plot created")
    
    def plot_technical_indicators(self, df: pd.DataFrame) -> None:
        """Plot technical indicators.
        
        Args:
            df: DataFrame with technical indicators
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price with moving averages
        axes[0].plot(df.index, df['Close'], label='Close Price', linewidth=2)
        if 'sma_20' in df.columns:
            axes[0].plot(df.index, df['sma_20'], label='SMA 20', alpha=0.7)
        if 'sma_50' in df.columns:
            axes[0].plot(df.index, df['sma_50'], label='SMA 50', alpha=0.7)
        
        # Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
            axes[0].fill_between(df.index, df['bb_upper'], df['bb_lower'], 
                               alpha=0.2, label='Bollinger Bands')
        
        axes[0].set_title('Price and Moving Averages', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # RSI
        if 'rsi' in df.columns:
            axes[1].plot(df.index, df['rsi'], color=self.colors[2], linewidth=2)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
            axes[1].set_title('RSI (Relative Strength Index)', fontweight='bold')
            axes[1].set_ylabel('RSI')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            axes[2].plot(df.index, df['macd'], label='MACD', linewidth=2)
            axes[2].plot(df.index, df['macd_signal'], label='Signal', linewidth=2)
            if 'macd_histogram' in df.columns:
                axes[2].bar(df.index, df['macd_histogram'], alpha=0.3, label='Histogram')
            axes[2].set_title('MACD', fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Technical indicators plot created")
    
    def plot_feature_importance(self, importance: pd.Series, 
                              top_n: int = 20,
                              title: str = "Feature Importance") -> None:
        """Plot feature importance.
        
        Args:
            importance: Series with feature importance scores
            top_n: Number of top features to show
            title: Plot title
        """
        top_features = importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        bars = ax.barh(range(len(top_features)), top_features.values, 
                      color=self.colors[0], alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Importance Score')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Feature importance plot created")
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]]) -> None:
        """Plot model performance comparison.
        
        Args:
            results: Dictionary with model names as keys and metrics as values
        """
        metrics = ['rmse', 'mae', 'r2', 'mape']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in model_names]
            
            bars = axes[i].bar(model_names, values, color=self.colors[:len(model_names)], alpha=0.7)
            axes[i].set_title(f'{metric.upper()}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Model comparison plot created")
    
    def plot_predictions_vs_actual(self, actual: pd.Series, 
                                 predictions: Dict[str, np.ndarray],
                                 title: str = "Predictions vs Actual") -> None:
        """Plot predictions vs actual values.
        
        Args:
            actual: Actual values
            predictions: Dictionary with model names and predictions
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot actual values
        ax.plot(actual.index, actual.values, label='Actual', 
               linewidth=2, color='black', alpha=0.8)
        
        # Plot predictions
        for i, (model_name, pred) in enumerate(predictions.items()):
            if len(pred) == len(actual):
                ax.plot(actual.index, pred, label=model_name, 
                       linewidth=2, alpha=0.7, color=self.colors[i % len(self.colors)])
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Predictions vs actual plot created")
    
    def plot_residuals(self, actual: pd.Series, 
                      predicted: np.ndarray,
                      model_name: str = "Model") -> None:
        """Plot residual analysis.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            model_name: Name of the model
        """
        residuals = actual.values - predicted
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals over time
        axes[0, 0].plot(actual.index, residuals, alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title(f'{model_name} - Residuals Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[0, 1].scatter(predicted, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title(f'{model_name} - Residuals vs Predicted')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title(f'{model_name} - Residuals Distribution')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_name} - Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Residual analysis plot created")
    
    def create_interactive_dashboard(self, df: pd.DataFrame) -> None:
        """Create an interactive dashboard with Plotly.
        
        Args:
            df: DataFrame with price and indicator data
        """
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Price Chart', 'Volume', 'RSI', 'MACD'),
            vertical_spacing=0.08,
            row_width=[0.2, 0.1, 0.1, 0.1]
        )
        
        # Price chart with candlesticks
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Add moving averages
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Volume
        if 'Volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color='rgba(0,100,80,0.6)'
                ),
                row=2, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Overbought", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Oversold", row=3, col=1)
        
        # MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='red')
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Gold Price Analysis Dashboard',
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        fig.show()
        
        logger.info("Interactive dashboard created")
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              features: Optional[List[str]] = None) -> None:
        """Plot correlation matrix of features.
        
        Args:
            df: DataFrame with features
            features: List of features to include (if None, use all numeric features)
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        correlation_matrix = df[features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        logger.info("Correlation matrix plot created")


def main():
    """Example usage of the visualizer."""
    # Generate sample data
    dates = pd.date_range('2020-01-01', '2023-01-01', freq='D')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'Open': np.random.randn(len(dates)).cumsum() + 1800,
        'High': np.random.randn(len(dates)).cumsum() + 1810,
        'Low': np.random.randn(len(dates)).cumsum() + 1790,
        'Close': np.random.randn(len(dates)).cumsum() + 1800,
        'Volume': np.random.randint(1000, 10000, len(dates)),
        'sma_20': np.random.randn(len(dates)).cumsum() + 1800,
        'rsi': np.random.uniform(20, 80, len(dates)),
        'macd': np.random.randn(len(dates)),
        'macd_signal': np.random.randn(len(dates))
    }, index=dates)
    
    # Create visualizer
    viz = GoldPriceVisualizer()
    
    # Create various plots
    viz.plot_price_trend(sample_data)
    viz.plot_technical_indicators(sample_data)
    
    # Example feature importance
    importance = pd.Series({
        'feature_1': 0.15,
        'feature_2': 0.12,
        'feature_3': 0.10,
        'feature_4': 0.08,
        'feature_5': 0.07
    })
    viz.plot_feature_importance(importance)
    
    print("Visualization examples completed")


if __name__ == "__main__":
    main()