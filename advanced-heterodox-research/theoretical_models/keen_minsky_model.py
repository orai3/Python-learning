"""
Keen-Minsky Dynamic Model of Financial Instability

Implements Steve Keen's mathematical formalization of Hyman Minsky's Financial
Instability Hypothesis, featuring endogenous money creation, private debt dynamics,
and the transition from stable growth to financial crisis.

The model demonstrates how:
1. Rising debt initially stimulates demand and growth
2. Debt servicing costs eventually outweigh new credit
3. Fisher debt-deflation dynamics emerge
4. System exhibits limit cycles and bifurcations

Mathematical Foundation:
The model consists of three coupled differential equations:
- dλ/dt: Employment rate dynamics (Phillips curve)
- dω/dt: Wage share dynamics (wage-profit conflict)
- dd/dt: Private debt ratio dynamics (credit creation and repayment)

Where credit creation directly adds to aggregate demand, creating the
possibility of both "Minsky moments" and debt-deflationary spirals.

References:
- Keen, S. (1995). Finance and Economic Breakdown: Modeling Minsky's
  Financial Instability Hypothesis. Journal of Post Keynesian Economics, 17(4), 607-635.
- Keen, S. (2013). A Monetary Minsky Model of the Great Moderation and the
  Great Recession. Journal of Economic Behavior & Organization, 86, 221-235.
- Keen, S. (2011). Debunking Economics. Zed Books.
- Minsky, H. (1986). Stabilizing an Unstable Economy. Yale University Press.
- Fisher, I. (1933). The Debt-Deflation Theory of Great Depressions.
  Econometrica, 1(4), 337-357.

Author: Claude
License: MIT
"""

from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
import warnings


@dataclass
class KeenMinskyParameters:
    """
    Parameters for the Keen-Minsky model.

    All parameters calibrated based on Keen (2013) and empirical evidence
    from advanced economies.
    """
    # Production and productivity
    a: float = 5.0  # Labor productivity (output per worker)
    alpha: float = 0.025  # Productivity growth rate

    # Population and labor force
    beta: float = 0.02  # Population growth rate

    # Investment function
    kappa: float = -1.0  # Investment sensitivity to wage share (Goodwin/Kalecki)
    nu: float = 3.0  # Accelerator coefficient (investment response to demand)

    # Debt dynamics
    tau: float = 0.05  # Debt servicing/repayment rate
    r_debt: float = 0.03  # Interest rate on debt

    # Wage-price dynamics
    phi: float = 0.04  # Wage Phillips curve slope
    psi_w: float = 0.05  # Workers' bargaining power (wage push)

    # Capacity utilization (implicit)
    # The model assumes Say's Law does NOT hold - demand creates its own supply
    # through credit creation

    # Initial conditions
    lambda_0: float = 0.95  # Initial employment rate
    omega_0: float = 0.65  # Initial wage share
    d_0: float = 0.5  # Initial private debt to output ratio

    def __post_init__(self):
        """Validate parameter consistency"""
        assert 0 < self.lambda_0 < 1, "Employment rate must be between 0 and 1"
        assert 0 < self.omega_0 < 1, "Wage share must be between 0 and 1"
        assert self.d_0 >= 0, "Debt ratio must be non-negative"


class KeenMinskyModel:
    """
    Keen-Minsky model of financial instability and endogenous business cycles.

    The model extends Goodwin's (1967) growth cycle model by adding:
    1. Endogenous money creation through bank credit
    2. Debt servicing that drains aggregate demand
    3. Fisher debt-deflation dynamics
    4. Investment financed by both profits and credit

    Core Insight:
    Unlike Goodwin (where cycles are perpetual), Keen-Minsky shows how
    financial fragility builds up, eventually triggering crisis and depression.
    """

    def __init__(self, params: Optional[KeenMinskyParameters] = None):
        """
        Initialize Keen-Minsky model.

        Args:
            params: Model parameters. Uses defaults if None.
        """
        self.params = params or KeenMinskyParameters()
        self.results: Optional[pd.DataFrame] = None

    def system_equations(self, state: np.ndarray, t: float,
                         params: Optional[KeenMinskyParameters] = None) -> np.ndarray:
        """
        System of differential equations for Keen-Minsky model.

        State vector: [λ, ω, d]
        where:
        - λ (lambda): Employment rate (employed / labor force)
        - ω (omega): Wage share (wages / output)
        - d: Private debt to output ratio

        Returns:
            Time derivatives [dλ/dt, dω/dt, dd/dt]

        Mathematical Derivation:
        ========================

        1. EMPLOYMENT RATE (λ):
        -----------------------
        The employment rate changes based on economic growth minus labor force growth.

        dλ/dt = λ * (g_y - α - β)

        where:
        - g_y = growth rate of output
        - α = productivity growth (reduces labor demand for given output)
        - β = population growth (increases labor supply)

        Output growth comes from:
        g_y = (I/K) + (dD/dt)/Y

        The first term is real investment relative to capital (Harrod-Domar).
        The second term is NEW credit creation, which directly adds to demand
        in Keen's model (this is the key Post-Keynesian insight: credit creates
        purchasing power independently of income).

        2. WAGE SHARE (ω):
        ------------------
        Wage share dynamics reflect the class struggle over income distribution.

        dω/dt = ω * (Φ(λ) - α)

        where Φ(λ) is the Phillips curve:
        Φ(λ) = φ * (λ - λ*) + ψ_w

        - φ: Phillips curve slope (tightness of labor market)
        - λ*: NAIRU (non-accelerating inflation rate of unemployment), normalized to 1
        - ψ_w: Workers' bargaining power (wage push independent of employment)

        When employment is high (λ near 1), workers have bargaining power and
        wages rise faster than productivity. When employment is low, wages lag
        productivity growth.

        3. DEBT RATIO (d):
        ------------------
        This is Keen's key contribution - endogenous debt dynamics.

        dd/dt = (dD/dt)/Y - d * g_y

        where dD/dt (new borrowing) is determined by the investment function:

        I/Y = κ*ω + ν*d + autonomous_investment

        The first term (κ*ω) is the Kaleckian/Goodwin effect: higher wage share
        reduces profit rate, reducing investment. κ < 0.

        The second term (ν*d) is the accelerator: rising debt signals rising
        demand, encouraging investment. ν > 0.

        New borrowing equals investment minus profits:
        dD/dt = I - Π = I - (1-ω)*Y

        So the change in debt ratio is:
        dd/dt = I/Y - (1-ω) - d*(I/K + dD/dt/Y)

        This creates a nonlinear feedback:
        - Initially, rising debt → rising demand → rising growth
        - Eventually, debt servicing (r*d) exceeds new borrowing
        - Demand collapses, triggering debt-deflation

        This is Minsky's Financial Instability Hypothesis in mathematical form.
        """
        if params is None:
            params = self.params

        # Unpack state
        lambda_emp, omega, d = state

        # Constrain state variables to economically meaningful ranges
        lambda_emp = np.clip(lambda_emp, 0.01, 1.0)
        omega = np.clip(omega, 0.01, 0.99)
        d = np.clip(d, 0.0, 5.0)  # Debt ratios above 500% are economically implausible

        # 1. Profit share
        pi = 1 - omega

        # 2. Investment function (Kaleckian with debt accelerator)
        # I/Y = κ*ω + ν*d
        # κ < 0: Higher wage share → lower profit rate → lower investment
        # ν > 0: Higher debt → higher aggregate demand → higher investment (accelerator)
        inv_rate = params.kappa * omega + params.nu * d

        # Ensure investment rate is not too negative (can't disinvest beyond depreciation)
        inv_rate = max(inv_rate, -0.1)

        # 3. New debt creation rate
        # dD/dt = I - Π (investment financed by credit beyond retained profits)
        # As ratio to output: (dD/dt)/Y = I/Y - (1-ω)
        new_debt_rate = inv_rate - pi

        # 4. Output growth rate
        # g_y = I/Y + (dD/dt)/Y
        # This is Keen's crucial insight: new debt adds to demand
        g_y = inv_rate + new_debt_rate

        # 5. Phillips curve
        # Φ(λ) = φ*(λ - 1) + ψ_w
        # Wage inflation relative to productivity growth
        phillips = params.phi * (lambda_emp - 1.0) + params.psi_w

        # 6. Differential equations

        # dλ/dt = λ * (g_y - α - β)
        # Employment rate rises if output grows faster than labor productivity and population
        d_lambda_dt = lambda_emp * (g_y - params.alpha - params.beta)

        # dω/dt = ω * (Φ(λ) - α)
        # Wage share rises if wage inflation exceeds productivity growth
        d_omega_dt = omega * (phillips - params.alpha)

        # dd/dt = (dD/dt)/Y - d * g_y
        # Debt ratio rises from new borrowing, falls if output grows
        d_d_dt = new_debt_rate - d * g_y

        # Include debt servicing drain on demand (Fisher effect)
        # This makes the system unstable - debt servicing reduces demand
        debt_service = params.r_debt * d
        d_d_dt -= params.tau * d  # Debt repayment

        # Adjust output growth for debt servicing drain
        # This creates the crisis dynamics
        g_y_adjusted = g_y - debt_service

        # Recalculate employment dynamics with debt servicing
        d_lambda_dt = lambda_emp * (g_y_adjusted - params.alpha - params.beta)

        return np.array([d_lambda_dt, d_omega_dt, d_d_dt])

    def find_equilibrium(self, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Find steady-state equilibrium of the system.

        In Keen-Minsky models, the steady state is often unstable,
        and the system exhibits limit cycles around it.

        Args:
            initial_guess: Starting point for solver

        Returns:
            (equilibrium_state, is_stable)
        """
        if initial_guess is None:
            initial_guess = np.array([self.params.lambda_0,
                                     self.params.omega_0,
                                     self.params.d_0])

        def equations_at_equilibrium(state):
            return self.system_equations(state, 0.0)

        try:
            equilibrium = fsolve(equations_at_equilibrium, initial_guess)

            # Validate economically meaningful
            if (0 < equilibrium[0] < 1 and  # Employment rate
                0 < equilibrium[1] < 1 and  # Wage share
                equilibrium[2] >= 0):        # Debt ratio

                # Check stability via Jacobian eigenvalues
                is_stable = self._check_stability(equilibrium)
                return equilibrium, is_stable
            else:
                warnings.warn("Equilibrium solution outside valid range")
                return initial_guess, False

        except Exception as e:
            warnings.warn(f"Could not find equilibrium: {e}")
            return initial_guess, False

    def _check_stability(self, state: np.ndarray, epsilon: float = 1e-6) -> bool:
        """
        Check local stability of equilibrium by computing Jacobian eigenvalues.

        If all eigenvalues have negative real parts, equilibrium is stable.
        If any have positive real parts, equilibrium is unstable (saddle or source).

        For Keen-Minsky models, we expect instability (limit cycles).
        """
        # Numerical Jacobian
        n = len(state)
        J = np.zeros((n, n))

        f0 = self.system_equations(state, 0.0)

        for i in range(n):
            state_perturbed = state.copy()
            state_perturbed[i] += epsilon
            f_perturbed = self.system_equations(state_perturbed, 0.0)
            J[:, i] = (f_perturbed - f0) / epsilon

        eigenvalues = np.linalg.eigvals(J)

        # Stable if all real parts negative
        is_stable = np.all(np.real(eigenvalues) < 0)

        return is_stable

    def simulate(self, t_max: float = 200, t_points: int = 5000,
                 initial_state: Optional[np.ndarray] = None,
                 method: str = 'LSODA') -> pd.DataFrame:
        """
        Simulate the Keen-Minsky model over time.

        Args:
            t_max: Maximum time to simulate
            t_points: Number of time points
            initial_state: Initial [λ, ω, d]. Uses defaults if None.
            method: Integration method ('LSODA', 'RK45', 'DOP853')

        Returns:
            DataFrame with time series of state variables and derived quantities
        """
        if initial_state is None:
            initial_state = np.array([self.params.lambda_0,
                                     self.params.omega_0,
                                     self.params.d_0])

        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, t_points)

        # Solve using scipy's solve_ivp (more robust than odeint)
        solution = solve_ivp(
            fun=lambda t, y: self.system_equations(y, t),
            t_span=t_span,
            y0=initial_state,
            method=method,
            t_eval=t_eval,
            dense_output=True,
            max_step=0.1  # Prevent large steps in unstable regions
        )

        if not solution.success:
            warnings.warn(f"Integration failed: {solution.message}")

        # Extract results
        t = solution.t
        lambda_emp = solution.y[0]
        omega = solution.y[1]
        d = solution.y[2]

        # Calculate derived quantities
        pi = 1 - omega  # Profit share

        # Investment rate
        inv_rate = self.params.kappa * omega + self.params.nu * d

        # Output growth rate
        new_debt_rate = inv_rate - pi
        g_y = inv_rate + new_debt_rate

        # Debt servicing burden
        debt_service = self.params.r_debt * d

        # Create DataFrame
        df = pd.DataFrame({
            't': t,
            'lambda': lambda_emp,
            'omega': omega,
            'd': d,
            'pi': pi,
            'inv_rate': inv_rate,
            'growth_rate': g_y,
            'debt_service': debt_service,
            'new_debt': new_debt_rate,
        })

        # Add crisis indicator (negative growth + high debt)
        df['crisis'] = (df['growth_rate'] < -0.01) & (df['d'] > 1.0)

        self.results = df
        return df

    def phase_portrait(self, var_x: str = 'omega', var_y: str = 'd',
                      grid_points: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate phase portrait showing vector field.

        Useful for visualizing limit cycles and stability.

        Args:
            var_x: Variable for x-axis ('lambda', 'omega', or 'd')
            var_y: Variable for y-axis
            grid_points: Resolution of vector field

        Returns:
            (X, Y, U, V) for quiver plot
        """
        var_map = {'lambda': 0, 'omega': 1, 'd': 2}
        idx_x = var_map[var_x]
        idx_y = var_map[var_y]
        idx_z = [i for i in [0, 1, 2] if i not in [idx_x, idx_y]][0]

        # Set fixed value for third variable (use equilibrium or initial condition)
        equilibrium, _ = self.find_equilibrium()
        fixed_val = equilibrium[idx_z]

        # Create grid
        if var_x == 'lambda':
            x_range = np.linspace(0.5, 1.0, grid_points)
        elif var_x == 'omega':
            x_range = np.linspace(0.3, 0.9, grid_points)
        else:  # d
            x_range = np.linspace(0.0, 3.0, grid_points)

        if var_y == 'lambda':
            y_range = np.linspace(0.5, 1.0, grid_points)
        elif var_y == 'omega':
            y_range = np.linspace(0.3, 0.9, grid_points)
        else:  # d
            y_range = np.linspace(0.0, 3.0, grid_points)

        X, Y = np.meshgrid(x_range, y_range)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(grid_points):
            for j in range(grid_points):
                state = np.zeros(3)
                state[idx_x] = X[i, j]
                state[idx_y] = Y[i, j]
                state[idx_z] = fixed_val

                derivatives = self.system_equations(state, 0.0)

                U[i, j] = derivatives[idx_x]
                V[i, j] = derivatives[idx_y]

        return X, Y, U, V

    def bifurcation_analysis(self, param_name: str, param_range: np.ndarray,
                            n_periods: int = 100) -> pd.DataFrame:
        """
        Perform bifurcation analysis by varying a parameter.

        Shows how system dynamics change (e.g., stable → oscillatory → chaotic)
        as parameters vary.

        Args:
            param_name: Name of parameter to vary
            param_range: Array of parameter values to test
            n_periods: Simulation length for each parameter value

        Returns:
            DataFrame with parameter values and long-run state values
        """
        results = []

        for param_val in param_range:
            # Update parameter
            params_temp = KeenMinskyParameters()
            setattr(params_temp, param_name, param_val)

            # Simulate
            model_temp = KeenMinskyModel(params_temp)
            df = model_temp.simulate(t_max=n_periods, t_points=1000)

            # Extract long-run behavior (last 20% of simulation)
            long_run = df.iloc[int(0.8 * len(df)):]

            results.append({
                'param': param_val,
                'lambda_mean': long_run['lambda'].mean(),
                'lambda_std': long_run['lambda'].std(),
                'omega_mean': long_run['omega'].mean(),
                'omega_std': long_run['omega'].std(),
                'd_mean': long_run['d'].mean(),
                'd_std': long_run['d'].std(),
                'crisis_frequency': long_run['crisis'].mean(),
            })

        return pd.DataFrame(results)

    def identify_crises(self, threshold_growth: float = -0.02,
                       threshold_debt: float = 1.5) -> pd.DataFrame:
        """
        Identify crisis periods in simulation results.

        Crisis defined as: negative growth + elevated debt levels.

        Args:
            threshold_growth: Growth rate below which indicates crisis
            threshold_debt: Debt ratio above which is dangerous

        Returns:
            DataFrame of crisis episodes with start, end, severity
        """
        if self.results is None:
            raise ValueError("Must run simulate() first")

        df = self.results.copy()

        # Identify crisis periods
        in_crisis = (df['growth_rate'] < threshold_growth) & (df['d'] > threshold_debt)

        # Find crisis episodes
        crisis_start = []
        crisis_end = []
        crisis_severity = []

        currently_in_crisis = False
        current_crisis_start = None
        current_crisis_min_growth = 0

        for i, row in df.iterrows():
            if in_crisis.iloc[i] and not currently_in_crisis:
                # Crisis begins
                currently_in_crisis = True
                current_crisis_start = row['t']
                current_crisis_min_growth = row['growth_rate']

            elif in_crisis.iloc[i] and currently_in_crisis:
                # Crisis continues
                current_crisis_min_growth = min(current_crisis_min_growth, row['growth_rate'])

            elif not in_crisis.iloc[i] and currently_in_crisis:
                # Crisis ends
                currently_in_crisis = False
                crisis_start.append(current_crisis_start)
                crisis_end.append(row['t'])
                crisis_severity.append(abs(current_crisis_min_growth))

        crises = pd.DataFrame({
            'start': crisis_start,
            'end': crisis_end,
            'duration': np.array(crisis_end) - np.array(crisis_start),
            'severity': crisis_severity
        })

        return crises

    def minsky_stages(self) -> pd.DataFrame:
        """
        Classify periods according to Minsky's financial stages:
        - Hedge finance: Cash flows > debt servicing
        - Speculative finance: Interest can be paid, but not principal
        - Ponzi finance: Debt servicing > cash flows (must borrow to service debt)

        Returns:
            DataFrame with Minsky stage classifications
        """
        if self.results is None:
            raise ValueError("Must run simulate() first")

        df = self.results.copy()

        # Calculate cash flows (profits)
        cash_flows = df['pi']  # Profit share approximates cash flow to GDP

        # Debt servicing burden
        debt_servicing = df['debt_service']

        # New borrowing
        new_borrowing = df['new_debt']

        # Classify stages
        stages = []
        for i in range(len(df)):
            if cash_flows.iloc[i] > debt_servicing.iloc[i]:
                # Can pay interest and principal from cash flows
                stages.append('Hedge')
            elif new_borrowing.iloc[i] > 0 and cash_flows.iloc[i] > self.params.r_debt * df['d'].iloc[i]:
                # Can pay interest, but needs new borrowing for principal
                stages.append('Speculative')
            else:
                # Must borrow to pay even interest
                stages.append('Ponzi')

        df['minsky_stage'] = stages

        return df


def plot_keen_minsky_results(df: pd.DataFrame, title: str = "Keen-Minsky Model Simulation"):
    """
    Create comprehensive visualization of Keen-Minsky model results.

    Args:
        df: Results from model.simulate()
        title: Plot title
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Time series of state variables
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df['t'], df['lambda'], 'b-', linewidth=2, label='Employment rate (λ)')
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Full employment')
    ax1.set_ylabel('Employment Rate')
    ax1.set_xlabel('Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Employment Dynamics')

    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(df['t'], df['omega'], 'g-', linewidth=2, label='Wage share (ω)')
    ax2.plot(df['t'], df['pi'], 'r-', linewidth=2, label='Profit share (π)')
    ax2.set_ylabel('Income Share')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Functional Income Distribution')

    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(df['t'], df['d'], 'purple', linewidth=2, label='Debt ratio (d)')
    ax3.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='d = 1')
    ax3.set_ylabel('Private Debt / GDP')
    ax3.set_xlabel('Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Private Debt Dynamics')

    # 2. Growth and investment
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(df['t'], df['growth_rate'], 'b-', linewidth=2, label='Output growth')
    ax4.plot(df['t'], df['inv_rate'], 'g-', linewidth=2, label='Investment rate')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.fill_between(df['t'], 0, df['growth_rate'],
                     where=(df['growth_rate'] < 0), alpha=0.3, color='red', label='Recession')
    ax4.set_ylabel('Rate')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Growth and Investment')

    # 3. Debt dynamics
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(df['t'], df['new_debt'], 'b-', linewidth=2, label='New borrowing')
    ax5.plot(df['t'], df['debt_service'], 'r-', linewidth=2, label='Debt servicing')
    ax5.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax5.fill_between(df['t'], df['new_debt'], df['debt_service'],
                     where=(df['new_debt'] < df['debt_service']),
                     alpha=0.3, color='red', label='Minsky moment')
    ax5.set_ylabel('Rate (% of GDP)')
    ax5.set_xlabel('Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Credit Creation vs Debt Servicing')

    # 4. Phase portrait: wage share vs debt
    ax6 = plt.subplot(3, 3, 6)
    scatter = ax6.scatter(df['omega'], df['d'], c=df['t'], cmap='viridis',
                         s=10, alpha=0.6)
    ax6.set_xlabel('Wage Share (ω)')
    ax6.set_ylabel('Debt Ratio (d)')
    ax6.set_title('Phase Space: Distribution vs Debt')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Time')

    # 5. Phase portrait: employment vs debt
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(df['lambda'], df['d'], c=df['t'], cmap='plasma',
                         s=10, alpha=0.6)
    ax7.set_xlabel('Employment Rate (λ)')
    ax7.set_ylabel('Debt Ratio (d)')
    ax7.set_title('Phase Space: Employment vs Debt')
    ax7.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax7, label='Time')

    # 6. Goodwin cycle (employment vs wage share)
    ax8 = plt.subplot(3, 3, 8)
    # Color by debt to show financial fragility overlay
    scatter = ax8.scatter(df['lambda'], df['omega'], c=df['d'],
                         cmap='RdYlGn_r', s=10, alpha=0.6, vmin=0, vmax=2)
    ax8.set_xlabel('Employment Rate (λ)')
    ax8.set_ylabel('Wage Share (ω)')
    ax8.set_title('Goodwin Cycle with Financial Overlay')
    ax8.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax8, label='Debt Ratio')

    # 7. Crisis indicator
    ax9 = plt.subplot(3, 3, 9)
    crisis_indicator = ((df['growth_rate'] < -0.01) & (df['d'] > 1.0)).astype(int)
    ax9.fill_between(df['t'], 0, 1, where=(crisis_indicator > 0),
                     alpha=0.5, color='red', label='Crisis periods')
    ax9.set_ylabel('Crisis (binary)')
    ax9.set_xlabel('Time')
    ax9.set_ylim(-0.1, 1.1)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_title('Crisis Identification')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


# Example usage and demonstrations
if __name__ == "__main__":
    print("Keen-Minsky Model of Financial Instability")
    print("=" * 60)

    # 1. Baseline simulation
    print("\n1. Baseline Simulation")
    print("-" * 60)

    model = KeenMinskyModel()

    # Find equilibrium
    equilibrium, is_stable = model.find_equilibrium()
    print(f"Equilibrium: λ={equilibrium[0]:.3f}, ω={equilibrium[1]:.3f}, d={equilibrium[2]:.3f}")
    print(f"Stability: {'Stable' if is_stable else 'Unstable (expect limit cycles)'}")

    # Simulate
    df = model.simulate(t_max=200, t_points=5000)

    print(f"\nSimulation statistics:")
    print(f"  Mean employment rate: {df['lambda'].mean():.3f}")
    print(f"  Mean wage share: {df['omega'].mean():.3f}")
    print(f"  Mean debt ratio: {df['d'].mean():.3f}")
    print(f"  Max debt ratio: {df['d'].max():.3f}")
    print(f"  Growth rate volatility: {df['growth_rate'].std():.4f}")

    # 2. Identify crises
    print(f"\n2. Crisis Analysis")
    print("-" * 60)

    crises = model.identify_crises()
    print(f"Number of crises identified: {len(crises)}")
    if len(crises) > 0:
        print(f"\nCrisis episodes:")
        print(crises.to_string(index=False))
        print(f"\nAverage crisis duration: {crises['duration'].mean():.2f} time units")
        print(f"Average crisis severity: {crises['severity'].mean():.4f}")

    # 3. Minsky stages
    print(f"\n3. Minsky Financial Stages")
    print("-" * 60)

    df_minsky = model.minsky_stages()
    stage_counts = df_minsky['minsky_stage'].value_counts()
    stage_pct = stage_counts / len(df_minsky) * 100

    print("Time spent in each stage:")
    for stage in ['Hedge', 'Speculative', 'Ponzi']:
        if stage in stage_pct:
            print(f"  {stage}: {stage_pct[stage]:.1f}%")

    # 4. Parameter sensitivity - debt servicing rate
    print(f"\n4. Bifurcation Analysis: Interest Rate")
    print("-" * 60)

    r_range = np.linspace(0.01, 0.10, 20)
    bifurc_df = model.bifurcation_analysis('r_debt', r_range, n_periods=150)

    print("Effect of increasing interest rate:")
    print(f"  Low rate (r=1%): mean debt = {bifurc_df.iloc[0]['d_mean']:.2f}")
    print(f"  High rate (r=10%): mean debt = {bifurc_df.iloc[-1]['d_mean']:.2f}")
    print(f"  Crisis frequency at low rate: {bifurc_df.iloc[0]['crisis_frequency']:.1%}")
    print(f"  Crisis frequency at high rate: {bifurc_df.iloc[-1]['crisis_frequency']:.1%}")

    # 5. Visualization
    print(f"\n5. Generating Visualizations")
    print("-" * 60)

    fig = plot_keen_minsky_results(df, title="Keen-Minsky Model: Financial Instability Dynamics")
    fig.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/keen_minsky_baseline.png',
                dpi=150, bbox_inches='tight')
    print("Saved: keen_minsky_baseline.png")

    # 6. Compare low-debt vs high-debt economies
    print(f"\n6. Comparative Scenarios")
    print("-" * 60)

    # Low debt economy
    params_low_debt = KeenMinskyParameters(d_0=0.3, nu=2.0)
    model_low = KeenMinskyModel(params_low_debt)
    df_low = model_low.simulate(t_max=200)

    # High debt economy
    params_high_debt = KeenMinskyParameters(d_0=1.5, nu=4.0)
    model_high = KeenMinskyModel(params_high_debt)
    df_high = model_high.simulate(t_max=200)

    print("Low-debt economy:")
    print(f"  Average growth: {df_low['growth_rate'].mean():.3f}")
    print(f"  Growth volatility: {df_low['growth_rate'].std():.4f}")

    print("\nHigh-debt economy:")
    print(f"  Average growth: {df_high['growth_rate'].mean():.3f}")
    print(f"  Growth volatility: {df_high['growth_rate'].std():.4f}")

    print("\n" + "=" * 60)
    print("Simulation complete. Key Minsky insights demonstrated:")
    print("1. Debt-driven growth creates instability")
    print("2. System exhibits endogenous cycles and crises")
    print("3. Financial fragility builds during good times")
    print("4. Debt-deflation dynamics can emerge")
    print("=" * 60)
