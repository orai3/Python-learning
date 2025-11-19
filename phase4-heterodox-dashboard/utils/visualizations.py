"""
Visualization Utilities for Economic Data

Provides publication-quality charts and graphs for heterodox economic analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import seaborn as sns


class ChartGenerator:
    """
    Generate publication-quality charts for economic analysis.

    All charts follow best practices for academic publications:
    - Clear labels and titles
    - Proper units and scales
    - Source citations where applicable
    - Colorblind-friendly palettes
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize chart generator with matplotlib style.

        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            pass

        # Set colorblind-friendly palette
        self.colors = sns.color_palette('colorblind')
        sns.set_palette(self.colors)

    def create_figure(self, figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, any]:
        """Create a new figure and axis."""
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    def plot_time_series(self, data: pd.DataFrame, variables: List[str],
                        title: str, ylabel: str,
                        xlabel: str = 'Date',
                        figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        Plot time series data.

        Args:
            data: DataFrame with time index
            variables: List of column names to plot
            title: Chart title
            ylabel: Y-axis label
            xlabel: X-axis label
            figsize: Figure size

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, var in enumerate(variables):
            if var in data.columns:
                ax.plot(data.index, data[var], label=var,
                       linewidth=2, color=self.colors[i % len(self.colors)])

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)

        # Format x-axis for dates if applicable
        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def plot_sectoral_balances(self, data: pd.DataFrame,
                              title: str = "Sectoral Financial Balances (Godley)") -> Figure:
        """
        Plot sectoral balances following Godley's SFC approach.

        Args:
            data: DataFrame with balance columns
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 7))

        balance_cols = [col for col in data.columns if 'balance' in col.lower()]

        for i, col in enumerate(balance_cols):
            ax.plot(data.index, data[col], label=col.replace('_', ' ').title(),
                   linewidth=2.5, color=self.colors[i % len(self.colors)])

        # Add zero line
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Balance (% of GDP)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)

        # Add annotation about sectoral balances identity
        fig.text(0.5, 0.02,
                'Note: Sectoral balances must sum to zero (accounting identity)',
                ha='center', fontsize=9, style='italic')

        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def plot_distribution(self, data: pd.Series, title: str,
                         xlabel: str, bins: int = 30) -> Figure:
        """
        Plot distribution histogram.

        Args:
            data: Series with distribution data
            title: Chart title
            xlabel: X-axis label
            bins: Number of bins

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data.dropna(), bins=bins, edgecolor='black',
               alpha=0.7, color=self.colors[0])

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean and median lines
        mean_val = data.mean()
        median_val = data.median()

        ax.axvline(mean_val, color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--',
                  linewidth=2, label=f'Median: {median_val:.2f}')

        ax.legend()
        plt.tight_layout()
        return fig

    def plot_lorenz_curve(self, data: pd.Series,
                         title: str = "Lorenz Curve") -> Figure:
        """
        Plot Lorenz curve for inequality analysis.

        Args:
            data: Series with income/wealth data
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Sort data and calculate cumulative shares
        sorted_data = np.sort(data.dropna())
        n = len(sorted_data)

        cumulative_pop = np.arange(1, n + 1) / n
        cumulative_income = np.cumsum(sorted_data) / np.sum(sorted_data)

        # Plot Lorenz curve
        ax.plot(cumulative_pop, cumulative_income,
               linewidth=2.5, color=self.colors[0], label='Lorenz Curve')

        # Plot line of equality
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5,
               label='Perfect Equality', alpha=0.5)

        # Calculate and display Gini coefficient
        gini = 1 - 2 * np.trapz(cumulative_income, cumulative_pop)
        ax.text(0.6, 0.2, f'Gini Coefficient: {gini:.3f}',
               fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Cumulative Share of Population', fontsize=12)
        ax.set_ylabel('Cumulative Share of Income/Wealth', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        return fig

    def plot_wage_profit_share(self, wage_share: pd.Series,
                              profit_share: pd.Series,
                              title: str = "Functional Income Distribution") -> Figure:
        """
        Plot wage and profit shares (key for PK and Marxian analysis).

        Args:
            wage_share: Time series of wage share
            profit_share: Time series of profit share
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(wage_share.index, wage_share, label='Wage Share',
               linewidth=2.5, color=self.colors[0])
        ax.plot(profit_share.index, profit_share, label='Profit Share',
               linewidth=2.5, color=self.colors[1])

        # Add historical averages
        wage_avg = wage_share.mean()
        profit_avg = profit_share.mean()

        ax.axhline(y=wage_avg, color=self.colors[0], linestyle=':',
                  alpha=0.5, label=f'Avg Wage Share: {wage_avg:.1f}%')
        ax.axhline(y=profit_avg, color=self.colors[1], linestyle=':',
                  alpha=0.5, label=f'Avg Profit Share: {profit_avg:.1f}%')

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Share of GDP (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)

        if isinstance(wage_share.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def plot_correlation_matrix(self, corr_matrix: pd.DataFrame,
                                title: str = "Correlation Matrix") -> Figure:
        """
        Plot correlation matrix heatmap.

        Args:
            corr_matrix: Correlation matrix DataFrame
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, linewidths=0.5,
                   cbar_kws={'shrink': 0.8})

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig

    def plot_rate_of_profit(self, rate_of_profit: pd.Series,
                           title: str = "Rate of Profit (Marxian Analysis)") -> Figure:
        """
        Plot rate of profit with trend line (key Marxian indicator).

        Args:
            rate_of_profit: Time series of profit rate
            title: Chart title

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot actual rate of profit
        ax.plot(rate_of_profit.index, rate_of_profit,
               linewidth=2.5, color=self.colors[2], label='Rate of Profit')

        # Add trend line
        x = np.arange(len(rate_of_profit))
        y = rate_of_profit.values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        ax.plot(rate_of_profit.index, p(x),
               linestyle='--', linewidth=2, color='red',
               label=f'Trend (slope: {z[0]:.3f})', alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Rate of Profit (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.3)

        # Add annotation about tendency
        if z[0] < 0:
            trend_text = "Falling rate of profit (consistent with Marx's TRPF)"
        else:
            trend_text = "Rising rate of profit (counter-tendencies dominant)"

        fig.text(0.5, 0.02, trend_text, ha='center',
                fontsize=9, style='italic')

        if isinstance(rate_of_profit.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def plot_comparative_frameworks(self, results: Dict,
                                   indicator: str) -> Figure:
        """
        Create comparative visualization across theoretical frameworks.

        Args:
            results: Dictionary with framework analysis results
            indicator: Indicator to compare

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        frameworks = []
        values = []

        for framework_name, framework_results in results.items():
            # Try to find indicator in results
            if 'indicators' in framework_results and indicator in framework_results['indicators']:
                frameworks.append(framework_name.replace('_', ' ').title())
                values.append(framework_results['indicators'][indicator])

        if frameworks:
            bars = ax.bar(frameworks, values, color=self.colors[:len(frameworks)])

            ax.set_ylabel(indicator.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{indicator.replace("_", " ").title()} Across Frameworks',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        return fig

    def create_canvas(self, figure: Figure) -> FigureCanvas:
        """
        Create Qt canvas from matplotlib figure for embedding in PyQt.

        Args:
            figure: Matplotlib Figure object

        Returns:
            FigureCanvas for PyQt integration
        """
        return FigureCanvas(figure)
