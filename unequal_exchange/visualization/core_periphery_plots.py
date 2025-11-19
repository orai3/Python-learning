"""
Visualization Suite for Core-Periphery Analysis

Comprehensive plotting tools for dependency theory and unequal exchange:
1. Network diagrams of value flows
2. Sankey diagrams for value chains
3. Geographical maps of transfers
4. Time series of exploitation metrics
5. Comparative charts (North vs South)
6. Distribution plots (inequality)
7. Smile curve visualizations
8. Historical trend analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.sankey import Sankey
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CorePeripheryVisualizer:
    """
    Comprehensive visualization suite for core-periphery analysis.
    """

    def __init__(self, style: str = "academic"):
        """
        Initialize visualizer.

        Args:
            style: Plot style ("academic", "presentation", "publication")
        """
        self.style = style
        if style == "publication":
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.size'] = 9
        elif style == "presentation":
            plt.rcParams['font.size'] = 14

    def plot_value_transfer_network(self, transfers_df: pd.DataFrame,
                                   country_categories: Dict[str, str],
                                   min_transfer: float = 10,
                                   figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot network diagram of value transfers between countries.

        Args:
            transfers_df: DataFrame with columns: exporter, importer, value_transfer
            country_categories: Dict mapping country to category (core/periphery/semi-periphery)
            min_transfer: Minimum transfer value to display
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Filter significant transfers
        significant = transfers_df[transfers_df['value_transfer'].abs() > min_transfer].copy()

        # Position countries in circles by category
        countries = list(set(significant['exporter'].tolist() + significant['importer'].tolist()))

        # Separate by category
        core = [c for c in countries if country_categories.get(c) == 'core']
        periphery = [c for c in countries if country_categories.get(c) == 'periphery']
        semi_periphery = [c for c in countries if country_categories.get(c) == 'semi_periphery']

        # Position countries
        positions = {}

        # Core in center
        n_core = len(core)
        for i, country in enumerate(core):
            angle = 2 * np.pi * i / max(1, n_core)
            positions[country] = (0.3 * np.cos(angle), 0.3 * np.sin(angle))

        # Periphery on outer ring
        n_periphery = len(periphery)
        for i, country in enumerate(periphery):
            angle = 2 * np.pi * i / max(1, n_periphery)
            positions[country] = (0.8 * np.cos(angle), 0.8 * np.sin(angle))

        # Semi-periphery in middle ring
        n_semi = len(semi_periphery)
        for i, country in enumerate(semi_periphery):
            angle = 2 * np.pi * i / max(1, n_semi)
            positions[country] = (0.55 * np.cos(angle), 0.55 * np.sin(angle))

        # Plot transfers as arrows
        for _, row in significant.iterrows():
            exp, imp = row['exporter'], row['importer']
            if exp not in positions or imp not in positions:
                continue

            transfer = row['value_transfer']
            x1, y1 = positions[exp]
            x2, y2 = positions[imp]

            # Arrow width based on transfer size
            width = min(5, max(0.5, abs(transfer) / 50))

            # Color based on direction (periphery->core = red, core->periphery = blue)
            if country_categories.get(exp) == 'periphery' and country_categories.get(imp) == 'core':
                color = 'red'
                alpha = 0.6
            elif country_categories.get(exp) == 'core' and country_categories.get(imp) == 'periphery':
                color = 'blue'
                alpha = 0.4
            else:
                color = 'gray'
                alpha = 0.3

            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle='->', mutation_scale=20,
                linewidth=width, color=color, alpha=alpha,
                connectionstyle="arc3,rad=.1"
            )
            ax.add_patch(arrow)

        # Plot country nodes
        for country, (x, y) in positions.items():
            category = country_categories.get(country, 'unknown')
            if category == 'core':
                color = '#2ecc71'
                size = 800
            elif category == 'periphery':
                color = '#e74c3c'
                size = 600
            else:
                color = '#f39c12'
                size = 700

            ax.scatter([x], [y], s=size, c=color, alpha=0.8, edgecolors='black', linewidths=2, zorder=10)
            ax.text(x, y, country, ha='center', va='center', fontsize=9, fontweight='bold', zorder=11)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(color='#2ecc71', label='Core'),
            mpatches.Patch(color='#f39c12', label='Semi-Periphery'),
            mpatches.Patch(color='#e74c3c', label='Periphery'),
            mpatches.Patch(color='red', alpha=0.6, label='Periphery → Core (Value Loss)'),
            mpatches.Patch(color='blue', alpha=0.4, label='Core → Periphery')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.title('Value Transfer Network: Core-Periphery Structure', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()

        return fig

    def plot_historical_transfers(self, time_series: pd.DataFrame,
                                  countries: Optional[List[str]] = None,
                                  figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot historical value transfers over time.

        Args:
            time_series: DataFrame with columns: year, country, value_transfer
            countries: List of countries to plot (None = all)
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        if countries is None:
            countries = time_series['country'].unique()

        # Plot 1: Individual country transfers
        for country in countries:
            country_data = time_series[time_series['country'] == country]
            ax1.plot(country_data['year'], country_data['value_transfer'],
                    marker='o', label=country, linewidth=2, markersize=4)

        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Value Transfer (Billions USD)', fontsize=12)
        ax1.set_title('Value Transfers by Country (1960-2020)', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Aggregate North-South
        if 'category' in time_series.columns:
            core_transfers = time_series[time_series['category'] == 'core'].groupby('year')['value_transfer'].sum()
            periphery_transfers = time_series[time_series['category'] == 'periphery'].groupby('year')['value_transfer'].sum()

            ax2.fill_between(core_transfers.index, 0, core_transfers.values,
                            alpha=0.4, color='#2ecc71', label='Core Net Gain')
            ax2.fill_between(periphery_transfers.index, 0, periphery_transfers.values,
                            alpha=0.4, color='#e74c3c', label='Periphery Net Loss')

            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
            ax2.set_xlabel('Year', fontsize=12)
            ax2.set_ylabel('Net Transfer (Billions USD)', fontsize=12)
            ax2.set_title('Aggregate North-South Value Transfers', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_smile_curve(self, value_chain_data: pd.DataFrame,
                        figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot "smile curve" showing value distribution in global value chains.

        Args:
            value_chain_data: DataFrame with columns: segment, value_added_pct, position
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by position
        data = value_chain_data.sort_values('position')

        # Plot curve
        ax.plot(data['position'], data['value_added_pct'],
               marker='o', linewidth=3, markersize=10, color='#3498db')

        # Fill area under curve
        ax.fill_between(data['position'], 0, data['value_added_pct'],
                       alpha=0.3, color='#3498db')

        # Annotate segments
        for _, row in data.iterrows():
            ax.annotate(row['segment'],
                       xy=(row['position'], row['value_added_pct']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, fontweight='bold')

        # Highlight high-value segments
        upstream = data[data['position'] < 3]
        downstream = data[data['position'] > data['position'].max() - 3]

        ax.axvspan(upstream['position'].min() - 0.5, upstream['position'].max() + 0.5,
                  alpha=0.2, color='green', label='High Value (R&D, Design)')
        ax.axvspan(downstream['position'].min() - 0.5, downstream['position'].max() + 0.5,
                  alpha=0.2, color='green', label='High Value (Marketing, Brand)')

        ax.set_xlabel('Value Chain Position (Upstream → Downstream)', fontsize=12)
        ax.set_ylabel('Value Added Share (%)', fontsize=12)
        ax.set_title('Smile Curve: Value Distribution in Global Value Chain', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_exploitation_metrics(self, metrics_df: pd.DataFrame,
                                  figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot super-exploitation metrics comparison.

        Args:
            metrics_df: DataFrame with columns: country, wage_productivity_gap, labor_share, etc.
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Wage-Productivity Gap
        metrics_df = metrics_df.sort_values('wage_productivity_gap', ascending=False)
        colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in metrics_df['wage_productivity_gap']]
        ax1.barh(metrics_df['country'], metrics_df['wage_productivity_gap'], color=colors, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Wage-Productivity Gap (%)', fontsize=11)
        ax1.set_title('Super-Exploitation: Wage-Productivity Decoupling', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. Labor Share
        metrics_df = metrics_df.sort_values('labor_share')
        ax2.barh(metrics_df['country'], metrics_df['labor_share'], color='#3498db', alpha=0.7)
        ax2.axvline(x=50, color='red', linestyle='--', linewidth=1, label='50% benchmark')
        ax2.set_xlabel('Labor Share of Income (%)', fontsize=11)
        ax2.set_title('Labor Share of National Income', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.legend()

        # 3. Rate of Exploitation
        if 'rate_of_exploitation' in metrics_df.columns:
            metrics_df = metrics_df.sort_values('rate_of_exploitation', ascending=False)
            ax3.barh(metrics_df['country'], metrics_df['rate_of_exploitation'], color='#e67e22', alpha=0.7)
            ax3.set_xlabel('Rate of Exploitation (s/v %)', fontsize=11)
            ax3.set_title('Marxian Rate of Exploitation', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')

        # 4. Value Transfer as % of GDP
        if 'value_transfer_pct_gdp' in metrics_df.columns:
            metrics_df = metrics_df.sort_values('value_transfer_pct_gdp', ascending=False)
            colors = ['#e74c3c' if cat == 'periphery' else '#2ecc71'
                     for cat in metrics_df.get('category', ['unknown'] * len(metrics_df))]
            ax4.barh(metrics_df['country'], metrics_df['value_transfer_pct_gdp'], color=colors, alpha=0.7)
            ax4.set_xlabel('Value Transfer (% of GDP)', fontsize=11)
            ax4.set_title('Drain Rate: Transfers as % of GDP', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_policy_comparison(self, scenarios: pd.DataFrame,
                              metric: str = 'gdp',
                              figsize: Tuple[int, int] = (12, 7)) -> plt.Figure:
        """
        Compare policy scenarios over time.

        Args:
            scenarios: DataFrame with columns: year, scenario, [metric]
            metric: Metric to plot
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        scenario_list = scenarios['scenario'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_list)))

        for i, scenario in enumerate(scenario_list):
            scenario_data = scenarios[scenarios['scenario'] == scenario]
            ax.plot(scenario_data['year'], scenario_data[metric],
                   marker='o', linewidth=2.5, markersize=6,
                   label=scenario.replace('_', ' ').title(),
                   color=colors[i])

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Policy Scenario Comparison: {metric.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_terms_of_trade(self, tot_data: pd.DataFrame,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot terms of trade trends (Prebisch-Singer).

        Args:
            tot_data: DataFrame with columns: year, tot_index, primary_prices, manufactures_prices
            figsize: Figure size

        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot 1: Price indices
        ax1.plot(tot_data['year'], tot_data['primary_prices'],
                label='Primary Commodity Prices', linewidth=2.5, color='#8B4513')
        ax1.plot(tot_data['year'], tot_data['manufactures_prices'],
                label='Manufactured Goods Prices', linewidth=2.5, color='#4169E1')
        ax1.set_ylabel('Price Index (Base Year = 100)', fontsize=11)
        ax1.set_title('Prebisch-Singer: Commodity vs Manufactures Prices', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Terms of Trade
        ax2.plot(tot_data['year'], tot_data['tot_index'],
                linewidth=3, color='#e74c3c')
        ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.7, label='Base Year')
        ax2.fill_between(tot_data['year'], 100, tot_data['tot_index'],
                        where=(tot_data['tot_index'] < 100),
                        alpha=0.3, color='red', label='ToT Deterioration')
        ax2.set_xlabel('Year', fontsize=11)
        ax2.set_ylabel('Terms of Trade Index', fontsize=11)
        ax2.set_title('Terms of Trade Deterioration (South Perspective)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_dashboard(self, data: Dict[str, pd.DataFrame],
                        figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple visualizations.

        Args:
            data: Dictionary with various dataframes
            figsize: Figure size

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # This would create a comprehensive dashboard layout
        # Implementation details depend on available data

        fig.suptitle('Unequal Exchange Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)

        return fig


def save_all_plots(visualizer: CorePeripheryVisualizer,
                   data: Dict[str, pd.DataFrame],
                   output_dir: str = './plots/'):
    """
    Generate and save all plots.

    Args:
        visualizer: Visualizer instance
        data: Dictionary of dataframes
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots and save
    # (Implementation details...)
    pass
