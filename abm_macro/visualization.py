"""
Visualization Tools for ABM Macroeconomic Model

Provides comprehensive visualization:
- Time series plots of aggregate dynamics
- Distributional analysis (Lorenz curves, histograms)
- Network visualizations
- Animated evolution of the economy
- Policy comparison plots

References:
- Emphasis on heterodox concerns: distribution, instability, cycles
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any

from .economy import MacroeconomyABM, SimulationResults
from .base import calculate_lorenz_curve


# Set style for heterodox economics aesthetics
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ABMVisualizer:
    """
    Comprehensive visualization suite for ABM results.
    """

    def __init__(self, economy: MacroeconomyABM):
        self.economy = economy
        self.results = economy.get_results()

    def plot_macro_dashboard(self, save_path: Optional[str] = None):
        """
        Create comprehensive macro dashboard.

        Shows key aggregates and their evolution.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        time = range(len(self.results.time_series['gdp']))

        # 1. GDP and components
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(time, self.results.time_series['gdp'], label='GDP', linewidth=2, color='black')
        ax1.plot(time, self.results.time_series['consumption'], label='Consumption', alpha=0.7)
        ax1.plot(time, self.results.time_series['investment'], label='Investment', alpha=0.7)
        ax1.set_title('National Accounts: Emergent Business Cycles', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Output')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Unemployment rate
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(time, self.results.time_series['unemployment_rate'], color='red', linewidth=2)
        ax2.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='Target')
        ax2.set_title('Unemployment Rate\n(Involuntary unemployment from rationing)', fontsize=11)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Inflation
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(time, self.results.time_series['inflation'], color='orange', linewidth=2)
        ax3.axhline(y=0.02, color='gray', linestyle='--', alpha=0.5, label='Target')
        ax3.set_title('Inflation Rate\n(Markup pricing dynamics)', fontsize=11)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Interest rate
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(time, self.results.time_series['interest_rate'], color='blue', linewidth=2)
        ax4.set_title('Policy Interest Rate\n(Taylor rule or discretionary)', fontsize=11)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Rate')
        ax4.grid(True, alpha=0.3)

        # 5. Functional income distribution
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time, self.results.time_series['wage_share'], label='Wage Share', linewidth=2)
        ax5.plot(time, self.results.time_series['profit_share'], label='Profit Share', linewidth=2)
        ax5.axhline(y=0.65, color='gray', linestyle='--', alpha=0.3)
        ax5.set_title('Functional Distribution\n(Kaleckian: wage share affects demand)', fontsize=11)
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Share of GDP')
        ax5.legend()
        ax5.set_ylim([0, 1])
        ax5.grid(True, alpha=0.3)

        # 6. Wealth inequality (Gini)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time, self.results.time_series['gini_wealth'], color='purple', linewidth=2)
        ax6.set_title('Wealth Inequality (Gini)\n(Heterogeneity matters for dynamics)', fontsize=11)
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Gini Coefficient')
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3)

        # 7. Income inequality (Gini)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.plot(time, self.results.time_series['gini_income'], color='green', linewidth=2)
        ax7.set_title('Income Inequality (Gini)\n(Different MPCs by class)', fontsize=11)
        ax7.set_xlabel('Time')
        ax7.set_ylabel('Gini Coefficient')
        ax7.set_ylim([0, 1])
        ax7.grid(True, alpha=0.3)

        # 8. Credit dynamics
        ax8 = fig.add_subplot(gs[3, 0])
        ax8.plot(time, self.results.time_series['total_credit'], color='brown', linewidth=2)
        ax8.set_title('Total Credit\n(Endogenous money creation)', fontsize=11)
        ax8.set_xlabel('Time')
        ax8.set_ylabel('Credit')
        ax8.grid(True, alpha=0.3)

        # 9. Credit rationing
        ax9 = fig.add_subplot(gs[3, 1])
        ax9.plot(time, self.results.time_series['credit_rationing_rate'], color='red', linewidth=2)
        ax9.set_title('Credit Rationing Rate\n(Stiglitz-Weiss: rationing is equilibrium)', fontsize=11)
        ax9.set_xlabel('Time')
        ax9.set_ylabel('Share of Requests Denied')
        ax9.grid(True, alpha=0.3)

        # 10. Bankruptcies
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.plot(time, self.results.time_series['bankruptcies'], color='darkred', linewidth=2)
        ax10.set_title('Firm Bankruptcies\n(Minskyan fragility & crises)', fontsize=11)
        ax10.set_xlabel('Time')
        ax10.set_ylabel('Count')
        ax10.grid(True, alpha=0.3)

        fig.suptitle('Agent-Based Macroeconomic Model: Emergent Dynamics from Micro Interactions',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Dashboard saved to {save_path}")

        return fig

    def plot_distributional_analysis(self, save_path: Optional[str] = None):
        """
        Detailed distributional analysis.

        Key heterodox insight: distribution is not just normative, it affects macro dynamics!
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        dist_data = self.economy.get_distributional_data()
        firm_data = self.economy.get_firm_distribution()

        # 1. Wealth Lorenz curve
        pop_share, wealth_share = calculate_lorenz_curve(dist_data['wealth'])
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Equality')
        axes[0, 0].plot(pop_share, wealth_share, linewidth=2, label='Actual Distribution')
        axes[0, 0].fill_between(pop_share, pop_share, wealth_share, alpha=0.3)
        axes[0, 0].set_title(f"Wealth Lorenz Curve\nGini = {self.economy.state.gini_wealth:.3f}", fontweight='bold')
        axes[0, 0].set_xlabel('Cumulative Population Share')
        axes[0, 0].set_ylabel('Cumulative Wealth Share')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Income Lorenz curve
        pop_share, income_share = calculate_lorenz_curve(dist_data['income'])
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect Equality')
        axes[0, 1].plot(pop_share, income_share, linewidth=2, label='Actual Distribution', color='green')
        axes[0, 1].fill_between(pop_share, pop_share, income_share, alpha=0.3, color='green')
        axes[0, 1].set_title(f"Income Lorenz Curve\nGini = {self.economy.state.gini_income:.3f}", fontweight='bold')
        axes[0, 1].set_xlabel('Cumulative Population Share')
        axes[0, 1].set_ylabel('Cumulative Income Share')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Wealth distribution histogram
        axes[0, 2].hist(dist_data['wealth'], bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(np.median(dist_data['wealth']), color='red',
                          linestyle='--', linewidth=2, label='Median')
        axes[0, 2].axvline(np.mean(dist_data['wealth']), color='blue',
                          linestyle='--', linewidth=2, label='Mean')
        axes[0, 2].set_title('Wealth Distribution\n(Log-normal with fat tail)', fontweight='bold')
        axes[0, 2].set_xlabel('Wealth')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        axes[0, 2].set_yscale('log')

        # 4. Consumption by wealth class
        class_consumption = {}
        for i, wc in enumerate(dist_data['wealth_class']):
            if wc not in class_consumption:
                class_consumption[wc] = []
            class_consumption[wc].append(dist_data['consumption'][i])

        class_avg = {k: np.mean(v) for k, v in class_consumption.items()}
        axes[1, 0].bar(class_avg.keys(), class_avg.values(), alpha=0.7)
        axes[1, 0].set_title('Average Consumption by Class\n(Different MPCs → inequality matters)', fontweight='bold')
        axes[1, 0].set_xlabel('Wealth Class')
        axes[1, 0].set_ylabel('Average Consumption')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 5. Firm size distribution
        axes[1, 1].hist(firm_data['capital'], bins=50, alpha=0.7, edgecolor='black', color='brown')
        axes[1, 1].set_title('Firm Size Distribution\n(Power law: few large, many small)', fontweight='bold')
        axes[1, 1].set_xlabel('Capital Stock')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_xscale('log')

        # 6. Profit rate distribution
        profit_rates = firm_data['profits'] / (firm_data['capital'] + 1)
        axes[1, 2].hist(profit_rates, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Profits')
        axes[1, 2].set_title('Firm Profit Rate Distribution\n(Heterogeneous performance)', fontweight='bold')
        axes[1, 2].set_xlabel('Profit Rate')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()

        fig.suptitle('Distributional Analysis: Heterogeneity and Inequality Dynamics',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Distributional analysis saved to {save_path}")

        return fig

    def plot_business_cycle_analysis(self, save_path: Optional[str] = None):
        """
        Analyze emergent business cycles.

        Shows cyclical dynamics absent in representative agent models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        time = np.array(range(len(self.results.time_series['gdp'])))
        gdp = np.array(self.results.time_series['gdp'])
        investment = np.array(self.results.time_series['investment'])
        credit = np.array(self.results.time_series['total_credit'])
        unemployment = np.array(self.results.time_series['unemployment_rate'])

        # 1. Goodwin-style cycle (unemployment vs wage share)
        wage_share = np.array(self.results.time_series['wage_share'])
        axes[0, 0].plot(unemployment * 100, wage_share * 100, linewidth=2, alpha=0.7)
        axes[0, 0].scatter(unemployment[0] * 100, wage_share[0] * 100,
                          s=100, c='green', marker='o', label='Start', zorder=5)
        axes[0, 0].scatter(unemployment[-1] * 100, wage_share[-1] * 100,
                          s=100, c='red', marker='X', label='End', zorder=5)
        axes[0, 0].set_title('Goodwin Cycle\n(Class struggle dynamics)', fontweight='bold')
        axes[0, 0].set_xlabel('Unemployment Rate (%)')
        axes[0, 0].set_ylabel('Wage Share (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Investment-profits dynamics (Kaleckian)
        profits = np.array([self.results.time_series['profit_share'][i] *
                           self.results.time_series['gdp'][i]
                           for i in range(len(time))])
        axes[0, 1].scatter(profits[:-1], investment[1:], alpha=0.5, s=20)
        axes[0, 1].set_title('Investment-Profit Relationship\n(Kalecki: profits drive investment)', fontweight='bold')
        axes[0, 1].set_xlabel('Profits (t)')
        axes[0, 1].set_ylabel('Investment (t+1)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Credit-GDP dynamics (Minskyan)
        debt_gdp = np.array(self.results.time_series['total_credit']) / (np.array(self.results.time_series['gdp']) + 1)
        axes[1, 0].plot(time, debt_gdp * 100, linewidth=2, color='brown')
        axes[1, 0].set_title('Credit/GDP Ratio\n(Minsky: rising leverage → fragility)', fontweight='bold')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Credit/GDP (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Financial fragility indicator
        bankruptcies = np.array(self.results.time_series['bankruptcies'])
        rationing = np.array(self.results.time_series['credit_rationing_rate'])

        ax4a = axes[1, 1]
        ax4b = ax4a.twinx()

        ax4a.plot(time, bankruptcies, color='red', linewidth=2, label='Bankruptcies')
        ax4b.plot(time, rationing * 100, color='blue', linewidth=2, label='Credit Rationing Rate', alpha=0.7)

        ax4a.set_xlabel('Time')
        ax4a.set_ylabel('Bankruptcies', color='red')
        ax4b.set_ylabel('Credit Rationing Rate (%)', color='blue')
        ax4a.tick_params(axis='y', labelcolor='red')
        ax4b.tick_params(axis='y', labelcolor='blue')
        ax4a.set_title('Financial Fragility\n(Endogenous boom-bust cycles)', fontweight='bold')
        ax4a.grid(True, alpha=0.3)

        lines1, labels1 = ax4a.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        fig.suptitle('Business Cycle Analysis: Emergent Fluctuations from Micro Interactions',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Business cycle analysis saved to {save_path}")

        return fig

    def create_animation(self, save_path: str = 'abm_animation.gif', fps: int = 10):
        """
        Create animated visualization of economy evolution.

        Shows how distribution and aggregates evolve over time.
        """
        print("🎬 Creating animation...")

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.4)

        # Prepare data
        time = range(len(self.results.time_series['gdp']))
        n_frames = len(time)

        def update(frame):
            fig.clear()
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.4)

            # 1. GDP evolution
            ax1 = fig.add_subplot(gs[0, :])
            ax1.plot(time[:frame+1], self.results.time_series['gdp'][:frame+1],
                    linewidth=2, color='black')
            ax1.set_xlim(0, n_frames)
            ax1.set_ylim(0, max(self.results.time_series['gdp']) * 1.1)
            ax1.set_title(f'GDP Evolution (t={frame})', fontweight='bold')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('GDP')
            ax1.grid(True, alpha=0.3)

            # 2. Unemployment
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(time[:frame+1], self.results.time_series['unemployment_rate'][:frame+1],
                    linewidth=2, color='red')
            ax2.set_xlim(0, n_frames)
            ax2.set_ylim(0, max(self.results.time_series['unemployment_rate']) * 1.1)
            ax2.set_title('Unemployment Rate', fontweight='bold')
            ax2.grid(True, alpha=0.3)

            # 3. Inequality
            ax3 = fig.add_subplot(gs[1, 1])
            ax3.plot(time[:frame+1], self.results.time_series['gini_wealth'][:frame+1],
                    linewidth=2, color='purple')
            ax3.set_xlim(0, n_frames)
            ax3.set_ylim(0, 1)
            ax3.set_title('Wealth Gini', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # 4. Credit
            ax4 = fig.add_subplot(gs[1, 2])
            ax4.plot(time[:frame+1], self.results.time_series['total_credit'][:frame+1],
                    linewidth=2, color='brown')
            ax4.set_xlim(0, n_frames)
            ax4.set_ylim(0, max(self.results.time_series['total_credit']) * 1.1)
            ax4.set_title('Total Credit', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            # 5. Wage share
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(time[:frame+1], self.results.time_series['wage_share'][:frame+1],
                    linewidth=2, color='green')
            ax5.set_xlim(0, n_frames)
            ax5.set_ylim(0, 1)
            ax5.set_title('Wage Share', fontweight='bold')
            ax5.grid(True, alpha=0.3)

            # 6. Bankruptcies
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(time[:frame+1], self.results.time_series['bankruptcies'][:frame+1],
                    linewidth=2, color='darkred')
            ax6.set_xlim(0, n_frames)
            ax6.set_ylim(0, max(self.results.time_series['bankruptcies']) * 1.1 + 1)
            ax6.set_title('Bankruptcies', fontweight='bold')
            ax6.grid(True, alpha=0.3)

            # 7. Interest rate
            ax7 = fig.add_subplot(gs[2, 2])
            ax7.plot(time[:frame+1], self.results.time_series['interest_rate'][:frame+1],
                    linewidth=2, color='blue')
            ax7.set_xlim(0, n_frames)
            ax7.set_ylim(0, max(self.results.time_series['interest_rate']) * 1.1)
            ax7.set_title('Interest Rate', fontweight='bold')
            ax7.grid(True, alpha=0.3)

            fig.suptitle('Agent-Based Macroeconomy: Real-Time Evolution',
                        fontsize=14, fontweight='bold')

        # Create animation
        anim = FuncAnimation(fig, update, frames=range(0, n_frames, 3),  # Every 3rd frame
                            interval=100, repeat=True)

        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)

        print(f"✓ Animation saved to {save_path}")
        plt.close()

        return anim


def compare_policy_experiments(results_dict: Dict[str, SimulationResults],
                               save_path: Optional[str] = None):
    """
    Compare multiple policy experiments side-by-side.

    Args:
        results_dict: Dictionary mapping experiment names to results
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for name, results in results_dict.items():
        time = range(len(results.time_series['gdp']))

        # GDP
        axes[0, 0].plot(time, results.time_series['gdp'], label=name, linewidth=2)

        # Unemployment
        axes[0, 1].plot(time, results.time_series['unemployment_rate'], label=name, linewidth=2)

        # Inequality
        axes[0, 2].plot(time, results.time_series['gini_wealth'], label=name, linewidth=2)

        # Wage share
        axes[1, 0].plot(time, results.time_series['wage_share'], label=name, linewidth=2)

        # Public debt
        axes[1, 1].plot(time, results.time_series['public_debt'], label=name, linewidth=2)

        # Bankruptcies
        axes[1, 2].plot(time, results.time_series['bankruptcies'], label=name, linewidth=2)

    # Formatting
    axes[0, 0].set_title('GDP', fontweight='bold')
    axes[0, 0].set_ylabel('Output')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Unemployment Rate', fontweight='bold')
    axes[0, 1].set_ylabel('Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_title('Wealth Inequality (Gini)', fontweight='bold')
    axes[0, 2].set_ylabel('Gini')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_title('Wage Share', fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Share')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Public Debt', fontweight='bold')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Debt')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].set_title('Firm Bankruptcies', fontweight='bold')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle('Policy Experiment Comparison', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Policy comparison saved to {save_path}")

    return fig
