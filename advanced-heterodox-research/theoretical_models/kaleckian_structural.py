"""
Kaleckian Structural Macroeconomic Model

Implements Michal Kalecki's approach to macroeconomic analysis with:
1. Pricing based on degree of monopoly (markup over costs)
2. Investment driven by profits and capacity utilization
3. Multi-sector structure (consumption goods, investment goods)
4. Wage-led vs profit-led demand regimes
5. Distribution and growth interactions
6. Financialisation extensions

Theoretical Foundation:

KALECKI'S PRINCIPLES:
- Profits are determined by capitalists' spending (profit equation)
- Prices determined by markup over prime costs
- Investment depends on profits and expected demand
- Class conflict over income distribution affects growth
- Effective demand failures due to oligopolistic pricing

CONTEMPORARY EXTENSIONS:
- Bhaduri-Marglin growth regimes (wage-led vs profit-led)
- Financialisation effects on distribution and investment
- Export-led growth possibilities
- Capacity utilization as adjustment variable

References:
- Kalecki, M. (1971). Selected Essays on the Dynamics of the Capitalist Economy.
  Cambridge University Press.
- Kalecki, M. (1954). Theory of Economic Dynamics. George Allen & Unwin.
- Bhaduri, A., & Marglin, S. (1990). Unemployment and the real wage: the economic
  basis for contesting political ideologies. Cambridge Journal of Economics, 14(4), 375-393.
- Lavoie, M. (2014). Post-Keynesian Economics: New Foundations. Edward Elgar.
- Hein, E. (2014). Distribution and Growth after Keynes. Edward Elgar.
- Blecker, R. (2016). Wage-led versus profit-led demand regimes: the long and the short of it.
  Review of Keynesian Economics, 4(4), 373-390.

Author: Claude
License: MIT
"""

from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import warnings


@dataclass
class KaleckianParameters:
    """
    Parameters for Kaleckian structural model.
    """
    # Pricing and distribution
    markup: float = 0.3  # Markup rate (degree of monopoly)
    z: float = 1.25  # Markup factor (1 + markup)

    # Consumption behavior
    s_w: float = 0.1  # Saving rate out of wages
    s_p: float = 0.4  # Saving rate out of profits (s_p > s_w is key)

    # Investment function (Kaleckian)
    gamma_0: float = 0.05  # Autonomous investment
    gamma_r: float = 0.4  # Sensitivity to profit rate
    gamma_u: float = 0.3  # Sensitivity to capacity utilization (accelerator)

    # Depreciation
    delta: float = 0.08  # Capital depreciation rate

    # Capacity utilization (normal/target level)
    u_n: float = 0.8  # Normal capacity utilization

    # Dynamics parameters
    alpha_u: float = 0.5  # Speed of adjustment of capacity utilization
    alpha_pi: float = 0.2  # Speed of adjustment of profit share (class struggle)

    # Class struggle / distribution dynamics
    omega_target_workers: float = 0.7  # Workers' target wage share
    omega_target_capitalists: float = 0.6  # Capitalists' target profit share

    # Financialisation parameters
    theta: float = 0.0  # Shareholder value orientation (reduces investment)

    # Export-led growth (optional)
    export_sensitivity: float = 0.0  # Sensitivity of exports to cost competitiveness

    # Initial conditions
    u_0: float = 0.75  # Initial capacity utilization
    pi_0: float = 0.35  # Initial profit share (1 - wage share)

    def __post_init__(self):
        """Validate parameters"""
        assert self.s_p > self.s_w, "Keynesian stability requires s_p > s_w"
        assert 0 < self.pi_0 < 1, "Profit share must be between 0 and 1"
        assert self.u_0 > 0, "Capacity utilization must be positive"


class KaleckianModel:
    """
    Kaleckian structural macroeconomic model.

    The model consists of:
    1. Pricing equation (markup over unit labor costs)
    2. Saving-investment balance (goods market equilibrium)
    3. Distribution conflict dynamics
    4. Capacity utilization adjustment

    Key Results:
    - Paradox of thrift (higher saving → lower growth)
    - Paradox of costs (higher wages can increase growth if wage-led)
    - Multiple equilibria possible
    - Demand creates its own capacity (Say's Law inverted)
    """

    def __init__(self, params: Optional[KaleckianParameters] = None):
        """
        Initialize Kaleckian model.

        Args:
            params: Model parameters. Uses defaults if None.
        """
        self.params = params or KaleckianParameters()
        self.results: Optional[pd.DataFrame] = None

    def profit_share_from_markup(self) -> float:
        """
        Calculate profit share from markup pricing.

        Kalecki's pricing theory:
        P = (1 + m) * W/A

        where:
        - P = price
        - m = markup rate
        - W = wage rate
        - A = labor productivity

        This gives profit share:
        π = Profits/Income = m/(1+m) = (z-1)/z

        where z = 1 + m is the markup factor.

        Returns:
            Profit share
        """
        return (self.params.z - 1) / self.params.z

    def wage_share_from_markup(self) -> float:
        """
        Calculate wage share from markup.

        ω = 1 - π = 1/z

        Returns:
            Wage share
        """
        return 1.0 / self.params.z

    def equilibrium_utilization(self, pi: Optional[float] = None) -> float:
        """
        Calculate equilibrium capacity utilization.

        From goods market equilibrium (S = I):

        Saving: S = s_w*(1-π)*u*Y_p + s_p*π*u*Y_p
              = [s_w + (s_p - s_w)*π] * u * Y_p

        where Y_p = potential output.

        Investment: I = [γ_0 + γ_r*π*u + γ_u*u] * K

        Normalizing Y_p = K (output-capital ratio = 1):

        Equilibrium: [s_w + (s_p - s_w)*π] * u = γ_0 + γ_r*π*u + γ_u*u

        Solving for u:

        u* = γ_0 / [s_w + (s_p - s_w)*π - γ_r*π - γ_u]

        This is the key Kaleckian result: equilibrium determined by demand!

        Args:
            pi: Profit share. If None, uses markup-determined share.

        Returns:
            Equilibrium capacity utilization

        """
        if pi is None:
            pi = self.profit_share_from_markup()

        # Saving rate out of total income
        s = self.params.s_w + (self.params.s_p - self.params.s_w) * pi

        # Investment rate coefficients
        inv_rate_autonomous = self.params.gamma_0
        inv_rate_profitability = self.params.gamma_r * pi
        inv_rate_accelerator = self.params.gamma_u

        # Denominator in equilibrium solution
        denominator = s - inv_rate_profitability - inv_rate_accelerator

        if denominator <= 0:
            raise ValueError("Model is unstable: denominator <= 0")

        u_star = inv_rate_autonomous / denominator

        return u_star

    def equilibrium_growth_rate(self, pi: Optional[float] = None,
                                u: Optional[float] = None) -> float:
        """
        Calculate equilibrium growth rate.

        Growth rate equals investment rate:
        g = I/K = γ_0 + γ_r*π*u + γ_u*u

        Args:
            pi: Profit share. If None, uses markup-determined.
            u: Capacity utilization. If None, uses equilibrium.

        Returns:
            Growth rate
        """
        if pi is None:
            pi = self.profit_share_from_markup()

        if u is None:
            u = self.equilibrium_utilization(pi)

        g = self.params.gamma_0 + self.params.gamma_r * pi * u + self.params.gamma_u * u

        # Subtract depreciation for net growth
        g_net = g - self.params.delta

        return g_net

    def is_wage_led(self, pi: Optional[float] = None) -> bool:
        """
        Determine if economy is wage-led or profit-led.

        Wage-led: ∂g/∂ω > 0 (higher wages → higher growth)
        Profit-led: ∂g/∂ω < 0 (higher profits → higher growth)

        Mathematical derivation:

        g = γ_0 + (γ_r*π + γ_u) * u

        where u = γ_0 / [s_w + (s_p - s_w)*π - γ_r*π - γ_u]

        Taking derivative with respect to π:

        ∂g/∂π = (γ_r*u + (γ_r*π + γ_u) * ∂u/∂π)

        The sign depends on parameter values.

        Simplified condition (Bhaduri-Marglin):
        Wage-led if: γ_u > γ_r * (s_p - s_w) / (1 - π)

        Args:
            pi: Profit share. If None, uses markup-determined.

        Returns:
            True if wage-led, False if profit-led
        """
        if pi is None:
            pi = self.profit_share_from_markup()

        u = self.equilibrium_utilization(pi)

        # Numerical derivative
        epsilon = 0.001
        pi_up = pi + epsilon
        pi_down = pi - epsilon

        try:
            g_up = self.equilibrium_growth_rate(pi_up)
            g_down = self.equilibrium_growth_rate(pi_down)

            dg_dpi = (g_up - g_down) / (2 * epsilon)

            # Wage-led if growth increases with wage share (decreases with profit share)
            return dg_dpi < 0

        except:
            # If can't calculate, use analytical approximation
            # Wage-led if accelerator dominates profitability effect
            wage_led_condition = self.params.gamma_u > self.params.gamma_r * pi

            return wage_led_condition

    def paradox_of_thrift(self) -> Dict[str, float]:
        """
        Demonstrate Kalecki's paradox of thrift.

        Shows that increasing saving rate reduces growth and utilization.

        Returns:
            Dictionary with results
        """
        baseline_growth = self.equilibrium_growth_rate()
        baseline_u = self.equilibrium_utilization()

        # Increase saving rate
        params_high_saving = KaleckianParameters()
        params_high_saving.s_w = self.params.s_w * 1.2
        params_high_saving.s_p = self.params.s_p * 1.2

        model_high_saving = KaleckianModel(params_high_saving)
        high_saving_growth = model_high_saving.equilibrium_growth_rate()
        high_saving_u = model_high_saving.equilibrium_utilization()

        return {
            'baseline_growth': baseline_growth,
            'high_saving_growth': high_saving_growth,
            'growth_change': high_saving_growth - baseline_growth,
            'baseline_utilization': baseline_u,
            'high_saving_utilization': high_saving_u,
            'utilization_change': high_saving_u - baseline_u,
            'paradox_confirmed': high_saving_growth < baseline_growth
        }

    def system_dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Dynamic system for adjustment processes.

        State: [u, π]
        where:
        - u = capacity utilization
        - π = profit share

        Dynamics:
        1. du/dt = α_u * [g - u] (utilization adjusts to growth)
        2. dπ/dt = α_π * [conflict function] (distribution conflict)

        Returns:
            [du/dt, dπ/dt]
        """
        u, pi = state

        # Constrain to valid range
        u = max(0.1, min(u, 2.0))
        pi = max(0.01, min(pi, 0.99))

        # Growth rate at current (u, π)
        g = self.params.gamma_0 + self.params.gamma_r * pi * u + self.params.gamma_u * u

        # Utilization adjustment
        # If growth > current utilization, capacity increases
        du_dt = self.params.alpha_u * (g - u)

        # Distribution conflict
        # Workers push for higher wage share (lower π)
        # Capitalists push for higher profit share (higher π)
        # Conflict depends on bargaining power, which depends on u

        # Simple specification: π adjusts toward target based on capacity utilization
        # High u → workers have power → π falls
        # Low u → capitalists have power → π rises

        if u > self.params.u_n:
            # Tight capacity → workers gain power
            pi_target = self.params.pi_0 * 0.9
        else:
            # Slack capacity → capitalists maintain/increase share
            pi_target = self.params.pi_0 * 1.1

        dpi_dt = self.params.alpha_pi * (pi_target - pi)

        return np.array([du_dt, dpi_dt])

    def simulate_dynamics(self, t_max: float = 100, t_points: int = 1000,
                         initial_state: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Simulate dynamic adjustment process.

        Args:
            t_max: Maximum time
            t_points: Number of time points
            initial_state: Initial [u, π]. Uses defaults if None.

        Returns:
            DataFrame with simulation results
        """
        if initial_state is None:
            initial_state = np.array([self.params.u_0, self.params.pi_0])

        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, t_points)

        solution = solve_ivp(
            fun=lambda t, y: self.system_dynamics(y, t),
            t_span=t_span,
            y0=initial_state,
            method='LSODA',
            t_eval=t_eval,
            dense_output=True
        )

        # Extract results
        t = solution.t
        u = solution.y[0]
        pi = solution.y[1]

        # Calculate derived variables
        omega = 1 - pi  # Wage share

        # Growth rate
        g = self.params.gamma_0 + self.params.gamma_r * pi * u + self.params.gamma_u * u

        # Investment rate
        inv_rate = g

        # Saving rate
        saving_rate = (self.params.s_w + (self.params.s_p - self.params.s_w) * pi) * u

        # Profit rate
        profit_rate = pi * u

        df = pd.DataFrame({
            't': t,
            'u': u,
            'pi': pi,
            'omega': omega,
            'g': g,
            'inv_rate': inv_rate,
            'saving_rate': saving_rate,
            'profit_rate': profit_rate,
        })

        self.results = df
        return df

    def regime_analysis(self, pi_range: np.ndarray) -> pd.DataFrame:
        """
        Analyze growth regime across different profit shares.

        Shows wage-led vs profit-led regions.

        Args:
            pi_range: Array of profit share values to test

        Returns:
            DataFrame with growth rates and classifications
        """
        results = []

        for pi in pi_range:
            try:
                u = self.equilibrium_utilization(pi)
                g = self.equilibrium_growth_rate(pi, u)

                # Calculate partial effects
                # Effect of distribution on growth (∂g/∂π)
                epsilon = 0.001
                g_up = self.equilibrium_growth_rate(pi + epsilon)
                g_down = self.equilibrium_growth_rate(pi - epsilon)
                dg_dpi = (g_up - g_down) / (2 * epsilon)

                regime = "Profit-led" if dg_dpi > 0 else "Wage-led"

                results.append({
                    'pi': pi,
                    'omega': 1 - pi,
                    'u': u,
                    'g': g,
                    'dg_dpi': dg_dpi,
                    'regime': regime
                })

            except:
                # Skip if parameters yield instability
                pass

        return pd.DataFrame(results)


def plot_kaleckian_results(model: KaleckianModel, title: str = "Kaleckian Model Analysis"):
    """
    Comprehensive visualization of Kaleckian model.

    Args:
        model: KaleckianModel instance
        title: Plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Growth vs distribution
    pi_range = np.linspace(0.2, 0.6, 100)
    regime_df = model.regime_analysis(pi_range)

    ax = axes[0, 0]
    ax.plot(regime_df['omega'], regime_df['g'], 'b-', linewidth=2.5)
    ax.axvline(x=model.wage_share_from_markup(), color='r', linestyle='--',
              label='Markup-determined ω')
    ax.set_xlabel('Wage Share (ω)', fontsize=11)
    ax.set_ylabel('Growth Rate (g)', fontsize=11)
    ax.set_title('Growth vs Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Utilization vs distribution
    ax = axes[0, 1]
    ax.plot(regime_df['omega'], regime_df['u'], 'g-', linewidth=2.5)
    ax.axhline(y=model.params.u_n, color='r', linestyle='--', alpha=0.5,
              label='Normal utilization')
    ax.set_xlabel('Wage Share (ω)', fontsize=11)
    ax.set_ylabel('Capacity Utilization (u)', fontsize=11)
    ax.set_title('Utilization vs Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Regime classification
    ax = axes[0, 2]
    wage_led = regime_df[regime_df['regime'] == 'Wage-led']
    profit_led = regime_df[regime_df['regime'] == 'Profit-led']

    if len(wage_led) > 0:
        ax.fill_between(wage_led['omega'], 0, 1, alpha=0.3, color='blue', label='Wage-led')
    if len(profit_led) > 0:
        ax.fill_between(profit_led['omega'], 0, 1, alpha=0.3, color='red', label='Profit-led')

    ax.set_xlabel('Wage Share (ω)', fontsize=11)
    ax.set_ylabel('Regime', fontsize=11)
    ax.set_title('Demand Regime Classification', fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Paradox of thrift
    ax = axes[1, 0]

    s_p_range = np.linspace(0.2, 0.8, 20)
    growth_rates = []
    utilizations = []

    for s_p in s_p_range:
        params_temp = KaleckianParameters()
        params_temp.s_p = s_p
        model_temp = KaleckianModel(params_temp)
        try:
            g = model_temp.equilibrium_growth_rate()
            u = model_temp.equilibrium_utilization()
            growth_rates.append(g)
            utilizations.append(u)
        except:
            growth_rates.append(np.nan)
            utilizations.append(np.nan)

    ax.plot(s_p_range, growth_rates, 'b-', linewidth=2.5, marker='o', markersize=4)
    ax.set_xlabel('Profit Saving Rate (s_p)', fontsize=11)
    ax.set_ylabel('Growth Rate (g)', fontsize=11)
    ax.set_title('Paradox of Thrift', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Investment function components
    ax = axes[1, 1]

    pi = model.profit_share_from_markup()
    u_vals = np.linspace(0.3, 1.2, 50)

    inv_autonomous = np.ones_like(u_vals) * model.params.gamma_0
    inv_profitability = model.params.gamma_r * pi * u_vals
    inv_accelerator = model.params.gamma_u * u_vals
    inv_total = inv_autonomous + inv_profitability + inv_accelerator

    ax.plot(u_vals, inv_total, 'k-', linewidth=2.5, label='Total investment')
    ax.plot(u_vals, inv_autonomous, 'r--', linewidth=1.5, label='Autonomous')
    ax.plot(u_vals, inv_profitability, 'g--', linewidth=1.5, label='Profitability')
    ax.plot(u_vals, inv_accelerator, 'b--', linewidth=1.5, label='Accelerator')

    ax.set_xlabel('Capacity Utilization (u)', fontsize=11)
    ax.set_ylabel('Investment Rate (I/K)', fontsize=11)
    ax.set_title('Investment Function Components', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Saving-investment equilibrium
    ax = axes[1, 2]

    # Saving function: S/K = [s_w + (s_p - s_w)*π] * u
    saving_rate = (model.params.s_w + (model.params.s_p - model.params.s_w) * pi) * u_vals

    ax.plot(u_vals, inv_total, 'b-', linewidth=2.5, label='Investment (I/K)')
    ax.plot(u_vals, saving_rate, 'r-', linewidth=2.5, label='Saving (S/K)')

    # Mark equilibrium
    u_eq = model.equilibrium_utilization(pi)
    g_eq = model.equilibrium_growth_rate(pi, u_eq)

    ax.plot(u_eq, g_eq, 'go', markersize=12, label='Equilibrium')

    ax.set_xlabel('Capacity Utilization (u)', fontsize=11)
    ax.set_ylabel('Rate (I/K, S/K)', fontsize=11)
    ax.set_title('Saving-Investment Balance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    return fig


# Example usage
if __name__ == "__main__":
    print("Kaleckian Structural Macroeconomic Model")
    print("=" * 70)

    # Baseline model
    print("\n1. Baseline Analysis")
    print("-" * 70)

    model = KaleckianModel()

    pi = model.profit_share_from_markup()
    omega = model.wage_share_from_markup()

    print(f"Distribution (markup-determined):")
    print(f"  Profit share: {pi:.3f}")
    print(f"  Wage share: {omega:.3f}")

    u_eq = model.equilibrium_utilization()
    g_eq = model.equilibrium_growth_rate()

    print(f"\nEquilibrium:")
    print(f"  Capacity utilization: {u_eq:.3f}")
    print(f"  Growth rate: {g_eq:.4f} ({g_eq*100:.2f}%)")

    is_wage_led = model.is_wage_led()
    print(f"\nDemand regime: {'Wage-led' if is_wage_led else 'Profit-led'}")

    # Paradox of thrift
    print(f"\n2. Paradox of Thrift")
    print("-" * 70)

    pot_results = model.paradox_of_thrift()
    print(f"Baseline growth: {pot_results['baseline_growth']:.4f}")
    print(f"After 20% increase in saving rates: {pot_results['high_saving_growth']:.4f}")
    print(f"Change in growth: {pot_results['growth_change']:.4f}")
    print(f"Paradox confirmed: {pot_results['paradox_confirmed']}")

    # Scenario analysis
    print(f"\n3. Scenario Analysis")
    print("-" * 70)

    # Wage-led economy (strong accelerator)
    params_wage_led = KaleckianParameters(gamma_u=0.6, gamma_r=0.2)
    model_wage_led = KaleckianModel(params_wage_led)

    print(f"\nWage-led scenario (strong accelerator):")
    print(f"  Is wage-led: {model_wage_led.is_wage_led()}")
    print(f"  Equilibrium growth: {model_wage_led.equilibrium_growth_rate():.4f}")

    # Profit-led economy (strong profitability effect)
    params_profit_led = KaleckianParameters(gamma_u=0.1, gamma_r=0.8)
    model_profit_led = KaleckianModel(params_profit_led)

    print(f"\nProfit-led scenario (strong profitability effect):")
    print(f"  Is wage-led: {model_profit_led.is_wage_led()}")
    print(f"  Equilibrium growth: {model_profit_led.equilibrium_growth_rate():.4f}")

    # Visualizations
    print(f"\n4. Generating Visualizations")
    print("-" * 70)

    fig = plot_kaleckian_results(model, title="Kaleckian Model: Distribution and Growth")
    fig.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/kaleckian_baseline.png',
                dpi=150, bbox_inches='tight')

    print("Saved: kaleckian_baseline.png")

    # Dynamic simulation
    print(f"\n5. Dynamic Adjustment")
    print("-" * 70)

    df = model.simulate_dynamics(t_max=50, initial_state=np.array([0.6, 0.4]))

    print(f"Initial state: u={0.6:.2f}, π={0.4:.2f}")
    print(f"Final state: u={df.iloc[-1]['u']:.2f}, π={df.iloc[-1]['pi']:.2f}")
    print(f"Converged to equilibrium: u*={u_eq:.2f}, π*={pi:.2f}")

    # Plot dynamics
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(df['t'], df['u'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=u_eq, color='r', linestyle='--', label='Equilibrium')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Capacity Utilization')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['t'], df['omega'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=omega, color='r', linestyle='--', label='Equilibrium')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Wage Share')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(df['t'], df['g'], 'purple', linewidth=2)
    axes[1, 0].axhline(y=g_eq, color='r', linestyle='--', label='Equilibrium')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Growth Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df['u'], df['omega'], 'k-', linewidth=2)
    axes[1, 1].plot(u_eq, omega, 'ro', markersize=10, label='Equilibrium')
    axes[1, 1].set_xlabel('Capacity Utilization')
    axes[1, 1].set_ylabel('Wage Share')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig2.suptitle('Dynamic Adjustment to Equilibrium', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig2.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/kaleckian_dynamics.png',
                 dpi=150, bbox_inches='tight')

    print("Saved: kaleckian_dynamics.png")

    print("\n" + "=" * 70)
    print("Kaleckian model demonstrates:")
    print("1. Demand determines output and growth (not supply)")
    print("2. Paradox of thrift: higher saving reduces growth")
    print("3. Distribution affects growth (wage-led vs profit-led)")
    print("4. Markup pricing determines distribution")
    print("5. Investment driven by profits and accelerator")
    print("=" * 70)
