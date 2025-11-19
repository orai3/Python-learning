"""
Representative Agent Model for Comparison

Implements a standard DSGE-style representative agent model to contrast with ABM.

Shows what ABMs can capture that representative agent models cannot:
- Heterogeneity and distributional dynamics
- Emergent aggregate behavior
- Coordination failures and involuntary unemployment
- Financial instability and endogenous cycles
- Network effects and contagion

References:
- Kirman (1992): "Whom or What Does the Representative Individual Represent?"
- Stiglitz (2018): "Where modern macroeconomics went wrong"
"""

import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class RepresentativeAgentModel:
    """
    Standard Representative Agent Macro Model.

    Simplified DSGE-style model with:
    - Single representative household
    - Single representative firm
    - Market clearing in all markets (no involuntary unemployment!)
    - Rational expectations
    - No financial sector (real economy only)
    """

    def __init__(self):
        # Parameters
        self.beta = 0.97  # Discount factor
        self.alpha = 0.7  # Labor share
        self.delta = 0.05  # Depreciation
        self.sigma = 2.0  # Risk aversion
        self.phi = 1.5  # Labor disutility

        # State variables
        self.K = 100.0  # Capital stock
        self.L = 1.0  # Labor (normalized)
        self.A = 1.0  # TFP

        # Prices
        self.w = 1.0  # Wage
        self.r = 0.05  # Interest rate

        # Aggregates
        self.Y = 0.0
        self.C = 0.0
        self.I = 0.0

        # Time series
        self.time_series = {
            'gdp': [], 'consumption': [], 'investment': [],
            'capital': [], 'wage': [], 'interest_rate': []
        }

    def production(self) -> float:
        """Cobb-Douglas production: Y = A * K^(1-α) * L^α"""
        return self.A * (self.K ** (1 - self.alpha)) * (self.L ** self.alpha)

    def marginal_product_labor(self) -> float:
        """MPL = α * Y / L"""
        return self.alpha * self.production() / self.L

    def marginal_product_capital(self) -> float:
        """MPK = (1-α) * Y / K"""
        return (1 - self.alpha) * self.production() / self.K

    def solve_equilibrium(self):
        """
        Solve for market-clearing equilibrium.

        In representative agent models:
        - Labor market clears: w = MPL (no unemployment!)
        - Capital market clears: r = MPK - δ
        - Goods market clears: Y = C + I
        - All agents identical → no distributional concerns
        """
        # Production
        self.Y = self.production()

        # Marginal products determine prices (perfect competition)
        self.w = self.marginal_product_labor()
        self.r = self.marginal_product_capital() - self.delta

        # Consumption (from Euler equation and budget constraint)
        # Simplified: consume fixed share of income
        income = self.w * self.L + self.r * self.K
        self.C = 0.7 * income  # Ad hoc consumption rule

        # Investment (residual)
        self.I = self.Y - self.C

        # Capital accumulation
        self.K += self.I - self.delta * self.K

        # Prevent capital from going negative
        self.K = max(1.0, self.K)

        # Labor supply = labor demand (always!)
        # No involuntary unemployment in representative agent models
        self.L = 1.0

    def simulate(self, n_periods: int = 300, shock_time: Optional[int] = None,
                 shock_size: float = -0.1):
        """
        Simulate the model.

        Can introduce productivity shock to compare with ABM crisis dynamics.
        """
        print(f"\n🔹 Simulating Representative Agent Model for {n_periods} periods...")

        for t in range(n_periods):
            # Productivity shock (if specified)
            if shock_time is not None and t == shock_time:
                self.A *= (1 + shock_size)
                print(f"   ⚡ Productivity shock: A = {self.A:.3f}")

            # Solve equilibrium
            self.solve_equilibrium()

            # Record
            self.time_series['gdp'].append(self.Y)
            self.time_series['consumption'].append(self.C)
            self.time_series['investment'].append(self.I)
            self.time_series['capital'].append(self.K)
            self.time_series['wage'].append(self.w)
            self.time_series['interest_rate'].append(self.r)

            # Progress
            if (t + 1) % 50 == 0:
                print(f"  t={t+1}/{n_periods} | GDP={self.Y:.1f} | K={self.K:.1f}")

        print(f"\n✓ Representative Agent simulation complete!")
        print(f"  Final GDP: {self.Y:.1f}")
        print(f"  Final Capital: {self.K:.1f}")

        return self.time_series


def compare_abm_vs_representative(abm_results: Dict[str, List[float]],
                                   ra_results: Dict[str, List[float]],
                                   save_path: Optional[str] = None):
    """
    Create comparison plots showing what ABM captures that RA models miss.

    Key differences:
    1. ABM has involuntary unemployment; RA assumes full employment
    2. ABM has inequality dynamics; RA has single agent
    3. ABM has endogenous cycles; RA only exogenous shocks
    4. ABM has financial fragility; RA often no finance
    5. ABM has bankruptcies/exits; RA immortal agent
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    time_abm = range(len(abm_results['gdp']))
    time_ra = range(len(ra_results['gdp']))

    # 1. GDP comparison
    axes[0, 0].plot(time_abm, abm_results['gdp'], label='ABM', linewidth=2, color='blue')
    axes[0, 0].plot(time_ra, ra_results['gdp'], label='Representative Agent',
                   linewidth=2, color='red', linestyle='--')
    axes[0, 0].set_title('GDP: Emergent Cycles vs Smooth Adjustment', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('GDP')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, 'ABM: Endogenous cycles\nRA: Deterministic convergence',
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Consumption
    axes[0, 1].plot(time_abm, abm_results['consumption'], label='ABM', linewidth=2, color='blue')
    axes[0, 1].plot(time_ra, ra_results['consumption'], label='Representative Agent',
                   linewidth=2, color='red', linestyle='--')
    axes[0, 1].set_title('Consumption: Volatile vs Smooth', fontweight='bold')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Consumption')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Investment
    axes[0, 2].plot(time_abm, abm_results['investment'], label='ABM', linewidth=2, color='blue')
    axes[0, 2].plot(time_ra, ra_results['investment'], label='Representative Agent',
                   linewidth=2, color='red', linestyle='--')
    axes[0, 2].set_title('Investment: Financial Constraints vs Frictionless', fontweight='bold')
    axes[0, 2].set_xlabel('Time')
    axes[0, 2].set_ylabel('Investment')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Unemployment - ABM ONLY!
    axes[1, 0].plot(time_abm, abm_results['unemployment_rate'], linewidth=2, color='blue')
    axes[1, 0].axhline(y=0.0, color='red', linestyle='--', linewidth=2,
                      label='RA Model (Always 0%)')
    axes[1, 0].set_title('Unemployment: Involuntary vs Non-existent', fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Unemployment Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.05, 0.95, 'RA models assume\nfull employment!',
                   transform=axes[1, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7),
                   fontweight='bold')

    # 5. Inequality - ABM ONLY!
    axes[1, 1].plot(time_abm, abm_results['gini_wealth'], linewidth=2, color='blue')
    axes[1, 1].axhline(y=0.0, color='red', linestyle='--', linewidth=2,
                      label='RA Model (N/A - single agent)')
    axes[1, 1].set_title('Wealth Inequality: Evolving vs Undefined', fontweight='bold')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Gini Coefficient')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].text(0.05, 0.95, 'RA: Single agent\n→ No inequality!',
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7),
                   fontweight='bold')

    # 6. Wage share - ABM ONLY!
    axes[1, 2].plot(time_abm, abm_results['wage_share'], linewidth=2, color='blue')
    axes[1, 2].axhline(y=0.7, color='red', linestyle='--', linewidth=2,
                      label='RA Model (Fixed at α)')
    axes[1, 2].set_title('Wage Share: Evolving vs Fixed Parameter', fontweight='bold')
    axes[1, 2].set_xlabel('Time')
    axes[1, 2].set_ylabel('Wage Share')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].text(0.05, 0.95, 'RA: Wage share = α\n(production parameter)',
                   transform=axes[1, 2].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # 7. Credit - ABM ONLY!
    axes[2, 0].plot(time_abm, abm_results['total_credit'], linewidth=2, color='blue')
    axes[2, 0].axhline(y=0.0, color='red', linestyle='--', linewidth=2,
                      label='RA Model (Often no finance)')
    axes[2, 0].set_title('Credit Dynamics: Endogenous vs Absent', fontweight='bold')
    axes[2, 0].set_xlabel('Time')
    axes[2, 0].set_ylabel('Total Credit')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].text(0.05, 0.95, 'Many RA models\nhave no banking!',
                   transform=axes[2, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7),
                   fontweight='bold')

    # 8. Bankruptcies - ABM ONLY!
    axes[2, 1].plot(time_abm, abm_results['bankruptcies'], linewidth=2, color='blue')
    axes[2, 1].axhline(y=0.0, color='red', linestyle='--', linewidth=2,
                      label='RA Model (Immortal agent)')
    axes[2, 1].set_title('Firm Exits: Creative Destruction vs Immortality', fontweight='bold')
    axes[2, 1].set_xlabel('Time')
    axes[2, 1].set_ylabel('Bankruptcies')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].text(0.05, 0.95, 'RA: Representative firm\nnever fails!',
                   transform=axes[2, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7),
                   fontweight='bold')

    # 9. Summary comparison table
    axes[2, 2].axis('off')
    comparison_text = """
    ABM vs Representative Agent Models

    Feature               | ABM      | RA Model
    ─────────────────────────────────────────
    Heterogeneity         | ✓        | ✗
    Involuntary Unemp.    | ✓        | ✗
    Inequality Dynamics   | ✓        | ✗
    Endogenous Cycles     | ✓        | ✗
    Financial Fragility   | ✓        | ✗
    Bankruptcies          | ✓        | ✗
    Distributional Effects| ✓        | ✗
    Coordination Failure  | ✓        | ✗
    Network Effects       | ✓        | ✗

    RA models assume:
    • Single representative agent
    • Perfect markets
    • Rational expectations
    • No coordination failures

    ABM captures:
    • Heterogeneous agents
    • Market frictions
    • Bounded rationality
    • Emergent phenomena
    """

    axes[2, 2].text(0.1, 0.9, comparison_text, transform=axes[2, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    fig.suptitle('ABM vs Representative Agent Model: What Heterogeneity Reveals',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ABM vs RA comparison saved to {save_path}")

    plt.show()

    return fig
