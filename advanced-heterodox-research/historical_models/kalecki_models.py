"""
Kalecki's Collected Models

Computational implementations of Michal Kalecki's key models from his essays
on economic dynamics, business cycles, and distribution theory.

Models Implemented:
1. Profit Equation (1935) - "Workers spend what they earn, capitalists earn what they spend"
2. Investment Function (1943) - Profits, capital stock, and rate of change
3. Business Cycle Theory (1937) - Investment-income dynamics
4. Distribution Theory (1938) - Degree of monopoly pricing
5. Political Business Cycle (1943) - Government spending and full employment

Each model includes:
- Mathematical formulation
- Historical context from Kalecki's original work
- Computational implementation
- Numerical examples
- Modern extensions

References:
- Kalecki, M. (1971). Selected Essays on the Dynamics of the Capitalist Economy 1933-1970.
  Cambridge University Press.
- Kalecki, M. (1954). Theory of Economic Dynamics: An Essay on Cyclical and Long-run Changes in
  Capitalist Economy. George Allen & Unwin.
- Kalecki, M. (1943). Political Aspects of Full Employment. The Political Quarterly, 14(4), 322-330.
- Sawyer, M. (1985). The Economics of Michal Kalecki. Macmillan.

Author: Claude
License: MIT
"""

from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve


# ============================================================================
# MODEL 1: PROFIT EQUATION
# ============================================================================

@dataclass
class ProfitEquationParams:
    """Parameters for Kalecki's profit equation model"""
    s_w: float = 1.0  # Workers' saving rate (typically ≈ 0)
    s_p: float = 0.3  # Capitalists' saving rate


def kalecki_profit_equation(c_w: float, c_p: float, i: float, params: ProfitEquationParams) -> Dict[str, float]:
    """
    Kalecki's Profit Equation (1935)

    One of Kalecki's most famous insights: "Workers spend what they earn,
    capitalists earn what they spend"

    Mathematical Derivation:
    National income identity: Y = W + P (wages + profits)
    Expenditure identity: Y = C + I (consumption + investment)

    Therefore: W + P = C + I

    Consumption splits into workers and capitalists:
    C = C_w + C_p

    Workers spend all wages (c_w ≈ 1):
    C_w = c_w * W

    Capitalists save part of profits (s_p < 1):
    C_p = (1 - s_p) * P

    Substituting:
    W + P = c_w * W + (1 - s_p) * P + I

    Solving for P:
    P - (1 - s_p) * P = I + c_w * W - W
    s_p * P = I - (1 - c_w) * W

    If workers spend all income (c_w = 1):
    P = I / s_p

    "Capitalists earn what they spend" on investment!

    This shows:
    1. Profits determined by capitalists' spending decisions
    2. Causation runs from investment to profits (not vice versa)
    3. Saving doesn't determine investment - investment determines saving

    Args:
        c_w: Workers' consumption propensity
        c_p: Capitalists' consumption propensity (= 1 - s_p)
        i: Investment
        params: Model parameters

    Returns:
        Dictionary with profits, wages, income, consumption
    """
    # From profit equation with workers saving
    s_w = params.s_w
    s_p = params.s_p

    # Total saving = total investment (equilibrium condition)
    # S = s_w * W + s_p * P = I

    # Also: Y = W + P and Y = C + I
    # C = (1-s_w)*W + (1-s_p)*P

    # Solve system:
    # W + P = (1-s_w)*W + (1-s_p)*P + I
    # W + P - (1-s_w)*W - (1-s_p)*P = I
    # s_w*W + s_p*P = I

    # Need another equation. Assume income distribution determined by markup.
    # Simplified: assume Y is given externally, or solve simultaneously.

    # Simplified solution assuming W adjusts:
    # If c_w = 1 (s_w = 0): P = I / s_p

    if abs(s_w) < 1e-6:
        # Workers don't save
        profits = i / s_p

        # Determine wages from national income identity
        # Need to assume something about Y. Use iterative approach.
        # For simplicity, use formula when workers spend all

        wages = i * (1 - s_p) / s_p  # Derived from equilibrium
        income = wages + profits
        consumption = income - i

    else:
        # General case
        # From s_w*W + s_p*P = I and some distribution assumption
        # This requires additional structure. Use simple markup model.

        # Assume P/Y = π (exogenous profit share)
        # Then: s_w*(1-π)*Y + s_p*π*Y = I
        # Y * [s_w + (s_p - s_w)*π] = I

        # Set profit share at 0.3
        pi = 0.3
        saving_rate = s_w + (s_p - s_w) * pi

        income = i / saving_rate
        profits = pi * income
        wages = (1 - pi) * income
        consumption = income - i

    return {
        'profits': profits,
        'wages': wages,
        'income': income,
        'consumption': consumption,
        'profit_share': profits / income,
        'wage_share': wages / income,
    }


# ============================================================================
# MODEL 2: INVESTMENT FUNCTION
# ============================================================================

@dataclass
class InvestmentFunctionParams:
    """Parameters for Kalecki's investment function"""
    a: float = 0.1  # Constant term
    b: float = 0.3  # Sensitivity to profits
    c: float = 0.2  # Sensitivity to capital stock (accelerator)
    d: float = 0.5  # Sensitivity to profit rate of change


def kalecki_investment_function(profits: np.ndarray, capital: np.ndarray,
                                delta_profits: np.ndarray,
                                params: InvestmentFunctionParams) -> np.ndarray:
    """
    Kalecki's Investment Function (1943)

    Investment decisions determined by three factors:

    I = a + b*P + c*ΔK + d*ΔP

    where:
    - P = profits
    - ΔK = rate of change of capital stock
    - ΔP = rate of change of profits

    Interpretations:
    - b*P: Profitability effect (higher profits → more internal funds)
    - c*ΔK: Accelerator effect (growing capital stock signals expansion)
    - d*ΔP: Expectations effect (rising profits signal good prospects)

    This combines:
    1. Keynesian animal spirits (a)
    2. Profits as source of finance (b)
    3. Accelerator principle (c)
    4. Expectations (d)

    Args:
        profits: Current profits
        capital: Capital stock (for ΔK calculation)
        delta_profits: Change in profits
        params: Model parameters

    Returns:
        Investment
    """
    # Calculate capital stock changes
    delta_capital = np.gradient(capital)

    # Investment function
    investment = (params.a +
                 params.b * profits +
                 params.c * delta_capital +
                 params.d * delta_profits)

    return investment


# ============================================================================
# MODEL 3: BUSINESS CYCLE MODEL
# ============================================================================

class KaleckiBusinessCycle:
    """
    Kalecki's Business Cycle Model (1937)

    Generates endogenous cycles through:
    1. Investment creates profits
    2. Profits stimulate investment
    3. Lags in capital accumulation create oscillations

    Differential equation system:
    dK/dt = I - δK (capital accumulation)
    dI/dt = f(P, K, dP/dt) (investment adjustment)
    P = I/s (profits from profit equation)
    """

    def __init__(self, params: Optional[InvestmentFunctionParams] = None):
        self.params = params or InvestmentFunctionParams()
        self.delta = 0.05  # Depreciation rate
        self.s_p = 0.3  # Capitalist saving rate
        self.tau = 0.5  # Investment adjustment speed

    def system_equations(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        System of differential equations.

        State: [K, I, P]
        where K=capital, I=investment, P=profits

        Returns:
            [dK/dt, dI/dt, dP/dt]
        """
        K, I, P = state

        # Profits determined by investment (profit equation)
        # P = I / s_p
        # But this is algebraic, not differential. Use adjustment process:

        # Capital accumulation
        dK_dt = I - self.delta * K

        # Target profits from profit equation
        P_target = I / self.s_p

        # Profit adjustment (gradual)
        dP_dt = self.tau * (P_target - P)

        # Target investment from investment function
        # Simplified to avoid needing dP/dt in calculation
        I_target = self.params.a + self.params.b * P + self.params.c * dK_dt

        # Investment adjustment
        dI_dt = self.tau * (I_target - I)

        return np.array([dK_dt, dI_dt, dP_dt])

    def simulate(self, t_max: float = 100, initial_state: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Simulate the business cycle.

        Args:
            t_max: Simulation time
            initial_state: Initial [K, I, P]

        Returns:
            DataFrame with time series
        """
        if initial_state is None:
            # Start near equilibrium
            K_0 = 100.0
            I_0 = self.delta * K_0
            P_0 = I_0 / self.s_p
            initial_state = np.array([K_0, I_0, P_0])

        t_eval = np.linspace(0, t_max, 1000)

        solution = solve_ivp(
            fun=lambda t, y: self.system_equations(y, t),
            t_span=(0, t_max),
            y0=initial_state,
            method='LSODA',
            t_eval=t_eval,
            dense_output=True
        )

        df = pd.DataFrame({
            't': solution.t,
            'K': solution.y[0],
            'I': solution.y[1],
            'P': solution.y[2],
        })

        # Derived variables
        df['Y'] = df['K']  # Simplified: assume Y/K = 1
        df['profit_share'] = df['P'] / df['Y']
        df['investment_rate'] = df['I'] / df['K']

        return df


# ============================================================================
# MODEL 4: DEGREE OF MONOPOLY PRICING
# ============================================================================

@dataclass
class DegreeOfMonopolyParams:
    """Parameters for Kalecki's pricing model"""
    m: float = 0.25  # Average markup rate
    w: float = 1.0  # Wage rate
    a: float = 1.0  # Labor productivity


def degree_of_monopoly_pricing(unit_costs: np.ndarray, markup: float) -> Tuple[np.ndarray, float]:
    """
    Kalecki's Degree of Monopoly Pricing Theory (1938)

    Firms set prices as markup over unit costs:
    P = (1 + m) * UC

    where:
    - P = price
    - m = markup rate (degree of monopoly)
    - UC = unit costs

    Unit costs depend on:
    UC = w/a + materials

    where w = wage rate, a = labor productivity

    This gives profit share:
    π = m / (1 + m)

    The markup m depends on:
    1. Industrial concentration
    2. Power of trade unions
    3. Overhead costs
    4. Advertising intensity

    Higher monopoly power → higher m → lower wage share

    This is a theory of distribution based on market structure,
    not marginal productivity!

    Args:
        unit_costs: Unit costs of production
        markup: Markup rate (degree of monopoly)

    Returns:
        (prices, profit_share)
    """
    prices = (1 + markup) * unit_costs

    # Profit share in value added
    profit_share = markup / (1 + markup)

    return prices, profit_share


# ============================================================================
# MODEL 5: POLITICAL BUSINESS CYCLE
# ============================================================================

class PoliticalBusinessCycle:
    """
    Kalecki's Political Business Cycle (1943)

    "Political Aspects of Full Employment"

    Key insight: Capitalists oppose sustained full employment because:
    1. Weakens discipline of unemployment
    2. Increases workers' bargaining power
    3. Threatens social and political stability

    Model: Government can maintain full employment through spending,
    but faces political constraints from business interests.
    """

    def __init__(self):
        self.u_target_business = 0.05  # Business prefers moderate unemployment
        self.u_target_labor = 0.02  # Labor wants near-full employment
        self.g_max = 100  # Maximum government spending
        self.political_power_business = 0.6  # Relative political power (0-1)

    def equilibrium_unemployment(self, g: float, y_potential: float) -> float:
        """
        Calculate equilibrium unemployment rate given government spending.

        Args:
            g: Government spending
            y_potential: Potential output (full employment)

        Returns:
            Unemployment rate
        """
        # Simplified: U = f(Y, Y*) where Y depends on G
        # Y = C + I + G

        # Assume consumption and investment functions
        c = 0.7 * y_potential  # Simplified
        i = 0.2 * y_potential

        y_actual = c + i + g

        # Unemployment rate
        u = 1 - (y_actual / y_potential)

        return max(0, u)

    def political_equilibrium(self, y_potential: float) -> Dict[str, float]:
        """
        Find political equilibrium unemployment rate.

        Business wants higher unemployment, labor wants lower.
        Equilibrium depends on relative power.

        Args:
            y_potential: Potential output

        Returns:
            Dictionary with equilibrium values
        """
        # Political equilibrium unemployment
        u_equilibrium = (self.political_power_business * self.u_target_business +
                        (1 - self.political_power_business) * self.u_target_labor)

        # Required government spending to achieve this
        def objective(g):
            return self.equilibrium_unemployment(g, y_potential) - u_equilibrium

        from scipy.optimize import fsolve
        g_equilibrium = fsolve(objective, 50.0)[0]
        g_equilibrium = np.clip(g_equilibrium, 0, self.g_max)

        # Actual unemployment with this spending
        u_actual = self.equilibrium_unemployment(g_equilibrium, y_potential)

        return {
            'unemployment_rate': u_actual,
            'government_spending': g_equilibrium,
            'employment_rate': 1 - u_actual,
            'output': y_potential * (1 - u_actual),
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_kalecki_business_cycle(df: pd.DataFrame, title: str = "Kalecki Business Cycle") -> plt.Figure:
    """Plot Kalecki business cycle simulation"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Capital and investment
    ax = axes[0, 0]
    ax.plot(df['t'], df['K'], 'b-', linewidth=2, label='Capital (K)')
    ax.plot(df['t'], df['I'], 'r-', linewidth=2, label='Investment (I)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Capital Accumulation')

    # Profits
    ax = axes[0, 1]
    ax.plot(df['t'], df['P'], 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Profits')
    ax.grid(True, alpha=0.3)
    ax.set_title('Profits Over Time')

    # Profit share
    ax = axes[1, 0]
    ax.plot(df['t'], df['profit_share'], 'purple', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Profit Share')
    ax.grid(True, alpha=0.3)
    ax.set_title('Distribution Dynamics')

    # Phase diagram: I vs P
    ax = axes[1, 1]
    ax.plot(df['P'], df['I'], 'b-', linewidth=1, alpha=0.6)
    ax.scatter(df['P'].iloc[0], df['I'].iloc[0], c='green', s=100, label='Start', zorder=5)
    ax.scatter(df['P'].iloc[-1], df['I'].iloc[-1], c='red', s=100, marker='X', label='End', zorder=5)
    ax.set_xlabel('Profits (P)')
    ax.set_ylabel('Investment (I)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Phase Diagram')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# Example usage
if __name__ == "__main__":
    print("Kalecki's Collected Models")
    print("=" * 70)

    # Model 1: Profit Equation
    print("\n1. Profit Equation (1935)")
    print("-" * 70)

    params_profit = ProfitEquationParams(s_w=0.0, s_p=0.3)
    result = kalecki_profit_equation(c_w=1.0, c_p=0.7, i=100, params=params_profit)

    print(f"Given investment = 100:")
    print(f"  Profits: {result['profits']:.2f}")
    print(f"  Wages: {result['wages']:.2f}")
    print(f"  Income: {result['income']:.2f}")
    print(f"  Profit share: {result['profit_share']:.3f}")
    print(f"\nKalecki's insight: Profits = Investment / Saving rate")
    print(f"  Calculated: {result['profits']:.2f}")
    print(f"  Formula: {100 / params_profit.s_p:.2f}")

    # Model 2: Business Cycle
    print("\n2. Business Cycle Model (1937)")
    print("-" * 70)

    cycle_model = KaleckiBusinessCycle()
    df_cycle = cycle_model.simulate(t_max=100)

    print(f"Simulated {len(df_cycle)} periods")
    print(f"\nCycle characteristics:")
    print(f"  Mean profit share: {df_cycle['profit_share'].mean():.3f}")
    print(f"  Profit share volatility: {df_cycle['profit_share'].std():.4f}")
    print(f"  Mean investment rate: {df_cycle['investment_rate'].mean():.3f}")

    # Detect cycles
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df_cycle['P'])
    if len(peaks) >= 2:
        avg_period = np.mean(np.diff(df_cycle['t'].iloc[peaks]))
        print(f"  Average cycle period: {avg_period:.1f} time units")

    # Model 3: Degree of Monopoly
    print("\n3. Degree of Monopoly Pricing (1938)")
    print("-" * 70)

    unit_costs = np.array([10.0])
    markup_low = 0.15
    markup_high = 0.40

    prices_low, pi_low = degree_of_monopoly_pricing(unit_costs, markup_low)
    prices_high, pi_high = degree_of_monopoly_pricing(unit_costs, markup_high)

    print(f"Low monopoly power (m={markup_low}):")
    print(f"  Price: {prices_low[0]:.2f}")
    print(f"  Profit share: {pi_low:.3f}")

    print(f"\nHigh monopoly power (m={markup_high}):")
    print(f"  Price: {prices_high[0]:.2f}")
    print(f"  Profit share: {pi_high:.3f}")

    print(f"\nEffect of increased monopoly power:")
    print(f"  Price increase: {(prices_high[0]/prices_low[0] - 1)*100:.1f}%")
    print(f"  Profit share increase: {(pi_high - pi_low)*100:.1f} percentage points")

    # Model 4: Political Business Cycle
    print("\n4. Political Business Cycle (1943)")
    print("-" * 70)

    pbc_model = PoliticalBusinessCycle()

    # Scenario 1: Business dominance
    pbc_model.political_power_business = 0.8
    eq_business = pbc_model.political_equilibrium(y_potential=1000)

    print(f"Business-dominated regime:")
    print(f"  Unemployment rate: {eq_business['unemployment_rate']*100:.1f}%")
    print(f"  Government spending: {eq_business['government_spending']:.1f}")

    # Scenario 2: Labor strength
    pbc_model.political_power_business = 0.3
    eq_labor = pbc_model.political_equilibrium(y_potential=1000)

    print(f"\nLabor-dominated regime:")
    print(f"  Unemployment rate: {eq_labor['unemployment_rate']*100:.1f}%")
    print(f"  Government spending: {eq_labor['government_spending']:.1f}")

    print(f"\nKalecki's insight: Full employment politically unfeasible")
    print(f"under capitalism due to business opposition")

    # Visualization
    print("\n5. Generating Visualizations")
    print("-" * 70)

    fig = plot_kalecki_business_cycle(df_cycle, title="Kalecki Business Cycle Model")
    fig.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/kalecki_business_cycle.png',
                dpi=150, bbox_inches='tight')

    print("Saved: kalecki_business_cycle.png")

    print("\n" + "=" * 70)
    print("Kalecki's key contributions demonstrated:")
    print("1. Profits determined by capitalist spending (profit equation)")
    print("2. Endogenous business cycles from investment-profit dynamics")
    print("3. Distribution determined by monopoly power, not productivity")
    print("4. Political constraints on full employment")
    print("5. Class conflict central to macroeconomic dynamics")
    print("=" * 70)
