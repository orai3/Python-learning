"""
Visualization Module for Historical Economics
=============================================

Comprehensive visualization tools for long-run economic analysis.

Visualizations:
- Long-run trend plots with regime shading
- Periodization diagrams
- Crisis timelines
- Kondratiev wave decomposition
- Hegemonic cycle diagrams
- Distribution dynamics
- Comparative cross-country analysis
- Interactive dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Optional
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class HistoricalPlotter:
    """
    Create publication-quality plots for historical economic analysis.
    """

    def __init__(self, data: pd.DataFrame, figsize: Tuple = (14, 8)):
        """
        Initialize plotter.

        Parameters
        ----------
        data : pd.DataFrame
            Historical economic data
        figsize : Tuple
            Default figure size
        """
        self.data = data
        self.figsize = figsize

    def plot_long_run_trends(self,
                            variables: List[str],
                            regime_periods: List[Dict] = None,
                            log_scale: bool = False,
                            title: str = None) -> plt.Figure:
        """
        Plot long-run trends with optional regime shading.

        Parameters
        ----------
        variables : List[str]
            Variables to plot
        regime_periods : List[Dict], optional
            Regime periods for shading
        log_scale : bool
            Use log scale for y-axis
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            Figure object
        """
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars), sharex=True)

        if n_vars == 1:
            axes = [axes]

        df = self.data.copy()

        for idx, var in enumerate(variables):
            ax = axes[idx]

            # Plot variable
            plot_data = df[df[var].notna()]
            ax.plot(plot_data['year'], plot_data[var], linewidth=2, color='darkblue')

            # Add regime shading
            if regime_periods:
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(regime_periods)))

                for i, period in enumerate(regime_periods):
                    ax.axvspan(period['start_year'], period['end_year'],
                              alpha=0.3, color=colors[i],
                              label=period.get('regime_label', f"Regime {i+1}"))

            ax.set_ylabel(var.replace('_', ' ').title(), fontsize=11)
            ax.grid(True, alpha=0.3)

            if log_scale:
                ax.set_yscale('log')

            # Add legend for first subplot
            if idx == 0 and regime_periods:
                ax.legend(loc='best', fontsize=9)

        axes[-1].set_xlabel('Year', fontsize=12)

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle('Long-Run Economic Trends', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_crisis_timeline(self,
                           crises: pd.DataFrame,
                           variable: str = 'gdp_growth',
                           title: str = None) -> plt.Figure:
        """
        Plot crisis timeline with severity indicators.

        Parameters
        ----------
        crises : pd.DataFrame
            Detected crises
        variable : str
            Variable to plot (typically GDP growth)
        title : str, optional
            Plot title

        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        df = self.data[self.data[variable].notna()].copy()

        # Plot growth rate
        ax.plot(df['year'], df[variable], linewidth=1.5, color='darkblue', label=variable)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        # Shade crisis periods
        for _, crisis in crises.iterrows():
            # Color intensity based on severity
            alpha = min(0.3 + crisis['severity'] * 0.5, 0.7)

            ax.axvspan(crisis['start_year'], crisis['end_year'],
                      alpha=alpha, color='red', label='Crisis' if _ == 0 else '')

            # Add crisis label
            mid_year = (crisis['start_year'] + crisis['end_year']) / 2
            ax.text(mid_year, df[variable].min() * 0.9,
                   f"{crisis['start_year']}\n({crisis['duration']}y)",
                   ha='center', fontsize=8, rotation=0)

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(variable.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('Economic Crisis Timeline', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_kondratiev_decomposition(self,
                                     variable: str,
                                     long_wave_series: pd.Series,
                                     waves: List[Dict] = None) -> plt.Figure:
        """
        Plot Kondratiev wave decomposition.

        Parameters
        ----------
        variable : str
            Original variable
        long_wave_series : pd.Series
            Extracted long wave component
        waves : List[Dict], optional
            Identified waves with peaks/troughs

        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        df = self.data.copy()

        # Top panel: Original series
        ax1 = axes[0]
        plot_data = df[df[variable].notna()]
        ax1.plot(plot_data['year'], plot_data[variable],
                linewidth=1.5, color='darkblue', label='Original')

        ax1.set_ylabel(variable.replace('_', ' ').title(), fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_title('Original Series', fontsize=11, fontweight='bold')

        # Bottom panel: Long wave component
        ax2 = axes[1]
        lw_data = df.iloc[long_wave_series.index]
        ax2.plot(lw_data['year'], long_wave_series.values,
                linewidth=2, color='red', label='Long Wave (~50yr)')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        # Mark peaks and troughs
        if waves:
            for wave in waves:
                # Mark trough 1
                t1_idx = wave.get('trough1_year', 0) - self.data['year'].min()
                if 0 <= t1_idx < len(long_wave_series):
                    ax2.plot(wave['trough1_year'],
                            long_wave_series.iloc[int(t1_idx)],
                            'go', markersize=8, label='Trough' if wave == waves[0] else '')

                # Mark peak
                p_idx = wave.get('peak_year', 0) - self.data['year'].min()
                if 0 <= p_idx < len(long_wave_series):
                    ax2.plot(wave['peak_year'],
                            long_wave_series.iloc[int(p_idx)],
                            'r^', markersize=8, label='Peak' if wave == waves[0] else '')

                # Mark trough 2
                t2_idx = wave.get('trough2_year', 0) - self.data['year'].min()
                if 0 <= t2_idx < len(long_wave_series):
                    ax2.plot(wave['trough2_year'],
                            long_wave_series.iloc[int(t2_idx)],
                            'go', markersize=8)

        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Long Wave Component', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_title('Kondratiev Long Wave Component', fontsize=11, fontweight='bold')

        fig.suptitle('Kondratiev Wave Decomposition', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_hegemonic_cycle(self,
                           hegemony_var: str = 'hegemony',
                           transitions: List[Dict] = None) -> plt.Figure:
        """
        Plot hegemonic strength over time with transition periods.

        Parameters
        ----------
        hegemony_var : str
            Hegemony indicator variable
        transitions : List[Dict], optional
            Detected hegemonic transitions

        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        df = self.data[self.data[hegemony_var].notna()].copy()

        # Plot hegemony index
        ax.plot(df['year'], df[hegemony_var], linewidth=2.5, color='darkgreen',
               label='Hegemonic Strength')

        # Shade transition periods
        if transitions:
            for trans in transitions:
                ax.axvspan(trans['start_year'], trans['end_year'],
                          alpha=0.3, color='orange', label='Transition' if trans == transitions[0] else '')

        # Add historical regime labels
        arrighi_cycles = [
            {'name': 'British\nHegemony', 'start': 1815, 'end': 1914, 'y': 0.85},
            {'name': 'Interregnum', 'start': 1914, 'end': 1945, 'y': 0.4},
            {'name': 'US\nHegemony', 'start': 1945, 'end': 1973, 'y': 0.9},
            {'name': 'US Decline?', 'start': 1973, 'end': 2020, 'y': 0.65}
        ]

        for cycle in arrighi_cycles:
            if cycle['start'] >= df['year'].min() and cycle['end'] <= df['year'].max():
                mid_year = (cycle['start'] + cycle['end']) / 2
                ax.text(mid_year, cycle['y'], cycle['name'],
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Hegemonic Strength Index', fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_title('Hegemonic Cycles in World Capitalism', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_distribution_dynamics(self,
                                  wage_share: bool = True,
                                  inequality: bool = True,
                                  regimes: List[Dict] = None) -> plt.Figure:
        """
        Plot distributional dynamics over time.

        Parameters
        ----------
        wage_share : bool
            Include wage share plot
        inequality : bool
            Include inequality plot
        regimes : List[Dict], optional
            Regime periods for shading

        Returns
        -------
        plt.Figure
            Figure object
        """
        n_plots = wage_share + inequality
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=True)

        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        if wage_share and 'wage_share' in self.data.columns:
            ax = axes[plot_idx]
            df = self.data[self.data['wage_share'].notna()]

            ax.plot(df['year'], df['wage_share'], linewidth=2, color='darkblue',
                   label='Wage Share')
            ax.plot(df['year'], df['profit_share'], linewidth=2, color='darkred',
                   label='Profit Share')

            # Add regime shading
            if regimes:
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(regimes)))
                for i, regime in enumerate(regimes):
                    ax.axvspan(regime['start_year'], regime['end_year'],
                              alpha=0.2, color=colors[i])

            ax.set_ylabel('Income Share', fontsize=11)
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            ax.set_title('Functional Income Distribution', fontsize=12, fontweight='bold')

            plot_idx += 1

        if inequality and 'gini' in self.data.columns:
            ax = axes[plot_idx]
            df = self.data[self.data['gini'].notna()]

            ax.plot(df['year'], df['gini'], linewidth=2, color='purple',
                   label='Gini Coefficient')

            # Add regime shading
            if regimes:
                colors = plt.cm.Pastel1(np.linspace(0, 1, len(regimes)))
                for i, regime in enumerate(regimes):
                    ax.axvspan(regime['start_year'], regime['end_year'],
                              alpha=0.2, color=colors[i])

            ax.set_ylabel('Gini Coefficient', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            ax.set_title('Personal Income Inequality', fontsize=12, fontweight='bold')

        axes[-1].set_xlabel('Year', fontsize=12)
        fig.suptitle('Distributional Dynamics', fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_comparative_countries(self,
                                  variable: str,
                                  countries: List[str] = None,
                                  normalize: bool = False) -> plt.Figure:
        """
        Plot comparative trends across countries.

        Parameters
        ----------
        variable : str
            Variable to compare
        countries : List[str], optional
            Countries to include
        normalize : bool
            Normalize to base year = 100

        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        if countries is None:
            countries = self.data['country'].unique()[:5]

        colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))

        for i, country in enumerate(countries):
            country_data = self.data[self.data['country'] == country]
            country_data = country_data[country_data[variable].notna()]

            if len(country_data) == 0:
                continue

            y_vals = country_data[variable].values

            if normalize:
                y_vals = (y_vals / y_vals[0]) * 100

            ax.plot(country_data['year'], y_vals,
                   linewidth=2, color=colors[i], label=country)

        ax.set_xlabel('Year', fontsize=12)

        if normalize:
            ax.set_ylabel(f'{variable.replace("_", " ").title()} (Base Year = 100)', fontsize=12)
        else:
            ax.set_ylabel(variable.replace('_', ' ').title(), fontsize=12)

        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        title = f'Comparative {variable.replace("_", " ").title()}'
        if normalize:
            title += ' (Normalized)'

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig

    def plot_profit_squeeze(self,
                          profit_rate_var: str = 'profit_rate',
                          wage_share_var: str = 'wage_share') -> plt.Figure:
        """
        Plot profit squeeze dynamics (Marxian/Goodwin framework).

        Shows relationship between profit rate and wage share.

        Parameters
        ----------
        profit_rate_var : str
            Profit rate variable
        wage_share_var : str
            Wage share variable

        Returns
        -------
        plt.Figure
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        df = self.data[[profit_rate_var, wage_share_var, 'year']].dropna()

        # Time series
        ax1 = axes[0]
        ax1_twin = ax1.twinx()

        ax1.plot(df['year'], df[profit_rate_var], linewidth=2, color='blue',
                label='Profit Rate')
        ax1_twin.plot(df['year'], df[wage_share_var], linewidth=2, color='red',
                     label='Wage Share')

        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Profit Rate', fontsize=11, color='blue')
        ax1_twin.set_ylabel('Wage Share', fontsize=11, color='red')

        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')

        ax1.grid(True, alpha=0.3)
        ax1.set_title('Time Series', fontsize=12, fontweight='bold')

        # Phase diagram (Goodwin cycle)
        ax2 = axes[1]

        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

        for i in range(len(df) - 1):
            ax2.plot(df[wage_share_var].iloc[i:i+2],
                    df[profit_rate_var].iloc[i:i+2],
                    color=colors[i], linewidth=1.5, alpha=0.7)

        # Add arrows to show direction
        every_nth = max(len(df) // 10, 1)
        for i in range(0, len(df) - 1, every_nth):
            ax2.annotate('',
                        xy=(df[wage_share_var].iloc[i+1], df[profit_rate_var].iloc[i+1]),
                        xytext=(df[wage_share_var].iloc[i], df[profit_rate_var].iloc[i]),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1))

        ax2.set_xlabel('Wage Share', fontsize=12)
        ax2.set_ylabel('Profit Rate', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Phase Diagram (Goodwin Cycle)', fontsize=12, fontweight='bold')

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=df['year'].min(),
                                                     vmax=df['year'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Year', fontsize=10)

        fig.suptitle('Profit Squeeze Dynamics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def create_summary_dashboard(data: pd.DataFrame,
                            crises: pd.DataFrame,
                            regime_periods: List[Dict]) -> plt.Figure:
    """
    Create comprehensive summary dashboard.

    Parameters
    ----------
    data : pd.DataFrame
        Historical data
    crises : pd.DataFrame
        Detected crises
    regime_periods : List[Dict]
        Regime periods

    Returns
    -------
    plt.Figure
        Dashboard figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # GDP growth with crises
    ax1 = fig.add_subplot(gs[0, :])
    df = data[data['gdp_growth'].notna()]
    ax1.plot(df['year'], df['gdp_growth'], linewidth=1.5, color='darkblue')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    for _, crisis in crises.iterrows():
        ax1.axvspan(crisis['start_year'], crisis['end_year'],
                   alpha=0.3, color='red')

    ax1.set_ylabel('GDP Growth Rate', fontsize=11)
    ax1.set_title('Economic Growth and Crises', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Distributional dynamics
    ax2 = fig.add_subplot(gs[1, 0])
    df = data[data['wage_share'].notna()]
    ax2.plot(df['year'], df['wage_share'], linewidth=2, label='Wage Share')
    ax2.plot(df['year'], df['profit_share'], linewidth=2, label='Profit Share')
    ax2.set_ylabel('Income Share', fontsize=10)
    ax2.set_title('Functional Distribution', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Inequality
    ax3 = fig.add_subplot(gs[1, 1])
    df = data[data['gini'].notna()]
    ax3.plot(df['year'], df['gini'], linewidth=2, color='purple')
    ax3.set_ylabel('Gini Coefficient', fontsize=10)
    ax3.set_title('Inequality', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Financialization
    ax4 = fig.add_subplot(gs[2, 0])
    df = data[data['financialization'].notna()]
    ax4.plot(df['year'], df['financialization'], linewidth=2, color='darkred')
    ax4.set_xlabel('Year', fontsize=11)
    ax4.set_ylabel('Financialization Index', fontsize=10)
    ax4.set_title('Financialization', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Profit rate
    ax5 = fig.add_subplot(gs[2, 1])
    df = data[data['profit_rate'].notna()]
    ax5.plot(df['year'], df['profit_rate'], linewidth=2, color='green')
    ax5.set_xlabel('Year', fontsize=11)
    ax5.set_ylabel('Profit Rate', fontsize=10)
    ax5.set_title('Rate of Profit', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    fig.suptitle('Historical Capitalism: Summary Dashboard', fontsize=16, fontweight='bold')

    return fig


if __name__ == '__main__':
    print("Visualization module loaded successfully.")
    print("\nAvailable classes:")
    print("- HistoricalPlotter: Publication-quality historical plots")
    print("\nAvailable functions:")
    print("- create_summary_dashboard: Comprehensive overview dashboard")
