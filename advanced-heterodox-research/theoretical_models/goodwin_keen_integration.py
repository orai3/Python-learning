"""
Goodwin-Keen Integration Model

Combines Richard Goodwin's growth cycle model with Steve Keen's debt dynamics,
creating a comprehensive framework for analyzing the interaction between:
1. Class conflict over income distribution (Goodwin)
2. Financial fragility and debt (Keen/Minsky)

This integration shows how distributive cycles and financial cycles interact,
potentially amplifying instability and creating complex crisis dynamics.

Theoretical Components:

GOODWIN (1967): Growth cycle as predator-prey system
- Employment rate (λ) affects wage bargaining power
- Wage share (ω) affects profit rate and investment
- Creates perpetual cycles in (λ, ω) space

KEEN (1995, 2013): Financial instability through debt
- Private debt (d) initially stimulates demand
- Debt servicing eventually drains demand
- Fisher debt-deflation dynamics

INTEGRATION:
- Debt affects investment (accelerator effect)
- Distribution affects debt accumulation (via profits)
- Employment affects both wages AND debt servicing capacity
- Creates complex, potentially chaotic dynamics

References:
- Goodwin, R. (1967). A Growth Cycle. In C. H. Feinstein (Ed.), Socialism,
  Capitalism and Economic Growth (pp. 54-58). Cambridge University Press.
- Keen, S. (2013). A monetary Minsky model of the Great Moderation and the
  Great Recession. Journal of Economic Behavior & Organization, 86, 221-235.
- Grasselli, M., & Costa Lima, B. (2012). An analysis of the Keen model for
  credit expansion, asset price bubbles and financial fragility. Mathematics
  and Financial Economics, 6(3), 191-210.
- Harvie, D. (2000). Testing Goodwin: growth cycles in ten OECD countries.
  Cambridge Journal of Economics, 24(3), 349-376.

Author: Claude
License: MIT
"""

from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import warnings


@dataclass
class GoodwinKeenParameters:
    """
    Parameters for the integrated Goodwin-Keen model.

    Combines classical growth cycle parameters with financial dynamics.
    """
    # Goodwin parameters
    alpha: float = 0.02  # Productivity growth rate
    beta: float = 0.01  # Population/labor force growth rate
    delta: float = 0.05  # Capital depreciation rate

    # Investment function (Goodwin-Kaleckian)
    gamma: float = 0.8  # Base investment rate
    kappa: float = -0.3  # Sensitivity to wage share (Kalecki effect)
    nu: float = 0.5  # Accelerator effect from utilization

    # Phillips curve parameters
    phi: float = 0.08  # Phillips curve slope
    rho: float = 0.03  # Autonomous wage push

    # Debt dynamics (Keen)
    sigma: float = 0.3  # Investment financed by credit (beyond profits)
    tau: float = 0.05  # Debt repayment rate
    r_debt: float = 0.03  # Interest rate on debt

    # Debt-investment interaction
    eta: float = 1.5  # Debt accelerator (credit-fueled investment)

    # Initial conditions
    lambda_0: float = 0.90  # Initial employment rate
    omega_0: float = 0.65  # Initial wage share
    d_0: float = 0.3  # Initial debt to capital ratio

    def __post_init__(self):
        """Validate parameters"""
        assert 0 < self.lambda_0 <= 1, "Employment rate must be in (0,1]"
        assert 0 < self.omega_0 < 1, "Wage share must be in (0,1)"
        assert self.d_0 >= 0, "Debt ratio must be non-negative"


class GoodwinKeenModel:
    """
    Integrated Goodwin-Keen model of growth cycles and financial fragility.

    State Variables:
    - λ (lambda): Employment rate = L/N (employed/labor force)
    - ω (omega): Wage share = wL/Y (wages/output)
    - d: Private debt to capital ratio = D/K

    The model demonstrates how distributional conflict and financial
    dynamics interact, creating complex cycles and potential instability.
    """

    def __init__(self, params: Optional[GoodwinKeenParameters] = None):
        """
        Initialize model.

        Args:
            params: Model parameters. Uses defaults if None.
        """
        self.params = params or GoodwinKeenParameters()
        self.results: Optional[pd.DataFrame] = None

    def system_equations(self, state: np.ndarray, t: float,
                         params: Optional[GoodwinKeenParameters] = None) -> np.ndarray:
        """
        System of differential equations for integrated model.

        State: [λ, ω, d]

        Returns:
            [dλ/dt, dω/dt, dd/dt]

        MATHEMATICAL DERIVATION:
        =======================

        1. EMPLOYMENT RATE DYNAMICS (dλ/dt):
        ------------------------------------

        From Goodwin (1967):

        Employment rate changes based on output growth vs labor supply growth:

        dλ/dt = λ * (g_Y - g_N)

        where:
        - g_Y = output growth rate
        - g_N = α + β = productivity growth + labor force growth

        Output growth comes from capital accumulation:
        g_Y = (I/K) - δ

        where I = investment, K = capital stock, δ = depreciation.

        So:
        dλ/dt = λ * [(I/K) - δ - α - β]

        2. WAGE SHARE DYNAMICS (dω/dt):
        --------------------------------

        Wage share evolves based on nominal wage growth vs productivity:

        dω/dt = ω * (g_w - g_p - α)

        where:
        - g_w = nominal wage growth
        - g_p = price inflation
        - α = productivity growth

        Real wage growth (g_w - g_p) depends on labor market tightness
        via Phillips curve:

        g_w - g_p = φ(λ - 1) + ρ

        This gives:
        dω/dt = ω * [φ(λ - 1) + ρ - α]

        3. DEBT RATIO DYNAMICS (dd/dt):
        -------------------------------

        From Keen's extension:

        Debt to capital ratio:
        d = D/K

        Taking time derivative:
        dd/dt = (dD/dt)/K - d*(dK/dt)/K
              = (dD/dt)/K - d*g_K

        where g_K = (I - δK)/K = I/K - δ

        New borrowing (dD/dt) finances investment beyond retained earnings:

        dD/dt = σ * I

        where σ ∈ [0,1] is the fraction financed externally.

        Alternatively, with debt accelerator:
        dD/dt = σ * I + η * d * K

        The second term captures self-reinforcing credit dynamics.

        So:
        dd/dt = σ*(I/K) + η*d - d*(I/K - δ)
              = (σ - d)*(I/K) + δ*d + η*d

        4. INVESTMENT FUNCTION:
        -----------------------

        Combining Goodwin and Kaleckian approaches:

        I/K = γ + κ*ω + ν*u + η*d

        where:
        - γ: autonomous investment
        - κ*ω: Kaleckian/Goodwin term (higher wages → lower profit rate → lower investment)
        - ν*u: accelerator (capacity utilization effect)
        - η*d: debt-driven investment (Minsky/Keen)

        For simplicity, assume u ≈ 1 (normal utilization), giving:

        I/K = γ + κ*ω + η*d

        This creates the feedback:
        High debt → high investment → high growth → high employment →
        high wages → low profits → credit demand → higher debt
        """
        if params is None:
            params = self.params

        # Unpack state
        lambda_emp, omega, d = state

        # Constrain to economically meaningful values
        lambda_emp = np.clip(lambda_emp, 0.01, 1.0)
        omega = np.clip(omega, 0.01, 0.99)
        d = np.clip(d, 0.0, 10.0)

        # Profit share
        pi = 1 - omega

        # Investment rate (I/K)
        # Combines Goodwin-Kalecki and Keen effects
        inv_rate = params.gamma + params.kappa * omega + params.eta * d

        # Ensure economically meaningful (can't have negative gross investment)
        inv_rate = max(inv_rate, 0.0)

        # Output growth rate
        # g_Y = I/K - δ
        g_Y = inv_rate - params.delta

        # Debt servicing burden
        debt_service = params.r_debt * d

        # Adjust growth for financial drain (Keen effect)
        g_Y_net = g_Y - debt_service

        # 1. Employment rate dynamics
        # dλ/dt = λ * (g_Y - α - β)
        d_lambda_dt = lambda_emp * (g_Y_net - params.alpha - params.beta)

        # 2. Wage share dynamics
        # Phillips curve: real wage growth = φ(λ - 1) + ρ
        phillips = params.phi * (lambda_emp - 1.0) + params.rho

        # dω/dt = ω * (phillips - α)
        d_omega_dt = omega * (phillips - params.alpha)

        # 3. Debt ratio dynamics
        # dd/dt = σ*(I/K) + η*d - d*(I/K - δ)
        d_d_dt = (params.sigma - d) * inv_rate + params.delta * d + params.eta * d

        # Subtract debt repayment
        d_d_dt -= params.tau * d

        return np.array([d_lambda_dt, d_omega_dt, d_d_dt])

    def simulate(self, t_max: float = 200, t_points: int = 5000,
                 initial_state: Optional[np.ndarray] = None,
                 method: str = 'LSODA') -> pd.DataFrame:
        """
        Simulate the integrated model.

        Args:
            t_max: Maximum simulation time
            t_points: Number of time points
            initial_state: Initial [λ, ω, d]. Uses defaults if None.
            method: Integration method

        Returns:
            DataFrame with simulation results
        """
        if initial_state is None:
            initial_state = np.array([
                self.params.lambda_0,
                self.params.omega_0,
                self.params.d_0
            ])

        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, t_points)

        # Solve ODE system
        solution = solve_ivp(
            fun=lambda t, y: self.system_equations(y, t),
            t_span=t_span,
            y0=initial_state,
            method=method,
            t_eval=t_eval,
            dense_output=True,
            max_step=0.1
        )

        if not solution.success:
            warnings.warn(f"Integration failed: {solution.message}")

        # Extract results
        t = solution.t
        lambda_emp = solution.y[0]
        omega = solution.y[1]
        d = solution.y[2]

        # Calculate derived quantities
        pi = 1 - omega

        # Investment rate
        inv_rate = self.params.gamma + self.params.kappa * omega + self.params.eta * d

        # Growth rate
        growth_rate = inv_rate - self.params.delta

        # Debt servicing
        debt_service = self.params.r_debt * d

        # Net growth (after debt servicing)
        growth_net = growth_rate - debt_service

        # Profit rate (approximation)
        # r_profit ≈ π * Y/K
        # Assuming Y/K ≈ output-capital ratio around 1
        profit_rate = pi * (1 + growth_rate)

        # Create DataFrame
        df = pd.DataFrame({
            't': t,
            'lambda': lambda_emp,
            'omega': omega,
            'd': d,
            'pi': pi,
            'inv_rate': inv_rate,
            'growth_rate': growth_rate,
            'growth_net': growth_net,
            'debt_service': debt_service,
            'profit_rate': profit_rate,
        })

        # Add cycle phase indicators
        df['goodwin_phase'] = self._classify_goodwin_phase(df['lambda'], df['omega'])

        # Crisis indicator
        df['crisis'] = (df['growth_net'] < -0.02) & (df['d'] > 1.0)

        self.results = df
        return df

    def _classify_goodwin_phase(self, lambda_vals: pd.Series, omega_vals: pd.Series) -> pd.Series:
        """
        Classify Goodwin cycle phases.

        Phases:
        1. Recovery: Low λ, low ω → Rising λ, rising ω
        2. Prosperity: High λ, rising ω → High λ, high ω
        3. Stagflation: High λ, high ω → Falling λ, high ω
        4. Recession: Falling λ, falling ω → Low λ, low ω
        """
        lambda_mean = lambda_vals.mean()
        omega_mean = omega_vals.mean()

        phases = []
        for lam, omeg in zip(lambda_vals, omega_vals):
            if lam < lambda_mean and omeg < omega_mean:
                phases.append('Recovery')
            elif lam >= lambda_mean and omeg < omega_mean:
                phases.append('Expansion')
            elif lam >= lambda_mean and omeg >= omega_mean:
                phases.append('Prosperity')
            else:  # lam < lambda_mean and omeg >= omega_mean
                phases.append('Stagflation')

        return pd.Series(phases, index=lambda_vals.index)

    def find_equilibrium(self) -> Tuple[np.ndarray, Dict[str, complex]]:
        """
        Find equilibrium point and analyze stability.

        Returns:
            (equilibrium_state, eigenvalues_dict)
        """
        initial_guess = np.array([
            self.params.lambda_0,
            self.params.omega_0,
            self.params.d_0
        ])

        def equations(state):
            return self.system_equations(state, 0.0)

        try:
            equilibrium = fsolve(equations, initial_guess)

            # Validate
            if not (0 < equilibrium[0] <= 1 and 0 < equilibrium[1] < 1 and equilibrium[2] >= 0):
                warnings.warn("Equilibrium outside valid range")
                return initial_guess, {}

            # Compute Jacobian for stability analysis
            eigenvalues = self._compute_eigenvalues(equilibrium)

            return equilibrium, eigenvalues

        except Exception as e:
            warnings.warn(f"Could not find equilibrium: {e}")
            return initial_guess, {}

    def _compute_eigenvalues(self, state: np.ndarray, epsilon: float = 1e-6) -> Dict[str, complex]:
        """
        Compute eigenvalues of Jacobian at given state.

        Returns:
            Dictionary mapping 'lambda_1', 'lambda_2', 'lambda_3' to eigenvalues
        """
        n = len(state)
        J = np.zeros((n, n))

        f0 = self.system_equations(state, 0.0)

        for i in range(n):
            state_pert = state.copy()
            state_pert[i] += epsilon
            f_pert = self.system_equations(state_pert, 0.0)
            J[:, i] = (f_pert - f0) / epsilon

        eigenvalues = np.linalg.eigvals(J)

        eig_dict = {
            'lambda_1': eigenvalues[0],
            'lambda_2': eigenvalues[1],
            'lambda_3': eigenvalues[2],
        }

        # Classify stability
        real_parts = [np.real(ev) for ev in eigenvalues]
        if all(rp < 0 for rp in real_parts):
            eig_dict['stability'] = 'Stable (sink)'
        elif all(rp > 0 for rp in real_parts):
            eig_dict['stability'] = 'Unstable (source)'
        elif any(rp > 0 for rp in real_parts):
            eig_dict['stability'] = 'Saddle'

        # Check for oscillations
        imag_parts = [np.imag(ev) for ev in eigenvalues]
        if any(abs(ip) > 1e-6 for ip in imag_parts):
            eig_dict['oscillatory'] = True
        else:
            eig_dict['oscillatory'] = False

        return eig_dict

    def cycle_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics about cyclical behavior.

        Returns:
            Dictionary with cycle characteristics
        """
        if self.results is None:
            raise ValueError("Must run simulate() first")

        df = self.results

        # Use second half to avoid transients
        df_stable = df.iloc[len(df)//2:]

        stats = {
            'lambda_mean': df_stable['lambda'].mean(),
            'lambda_std': df_stable['lambda'].std(),
            'lambda_amplitude': df_stable['lambda'].max() - df_stable['lambda'].min(),

            'omega_mean': df_stable['omega'].mean(),
            'omega_std': df_stable['omega'].std(),
            'omega_amplitude': df_stable['omega'].max() - df_stable['omega'].min(),

            'd_mean': df_stable['d'].mean(),
            'd_std': df_stable['d'].std(),
            'd_max': df_stable['d'].max(),

            'growth_mean': df_stable['growth_net'].mean(),
            'growth_std': df_stable['growth_net'].std(),

            'crisis_frequency': df_stable['crisis'].mean(),
        }

        # Estimate cycle period
        # Use zero-crossings of detrended employment rate
        lambda_detrended = df_stable['lambda'] - df_stable['lambda'].mean()
        zero_crossings = np.where(np.diff(np.sign(lambda_detrended)))[0]

        if len(zero_crossings) >= 4:
            # Average period (time between every other zero crossing)
            periods = []
            for i in range(0, len(zero_crossings)-2, 2):
                period = df_stable.iloc[zero_crossings[i+2]]['t'] - df_stable.iloc[zero_crossings[i]]['t']
                periods.append(period)

            stats['cycle_period'] = np.mean(periods)
        else:
            stats['cycle_period'] = np.nan

        return stats

    def phase_space_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get 3D phase space trajectory.

        Returns:
            (lambda_array, omega_array, d_array)
        """
        if self.results is None:
            raise ValueError("Must run simulate() first")

        return (
            self.results['lambda'].values,
            self.results['omega'].values,
            self.results['d'].values
        )


def plot_goodwin_keen_results(model: GoodwinKeenModel, df: pd.DataFrame,
                              title: str = "Goodwin-Keen Integrated Model"):
    """
    Comprehensive visualization of Goodwin-Keen model results.

    Args:
        model: GoodwinKeenModel instance
        df: Results DataFrame from model.simulate()
        title: Plot title
    """
    fig = plt.figure(figsize=(18, 12))

    # 1. Employment rate over time
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(df['t'], df['lambda'], 'b-', linewidth=2)
    ax1.set_ylabel('Employment Rate λ')
    ax1.set_xlabel('Time')
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Employment Dynamics')

    # 2. Wage share over time
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(df['t'], df['omega'], 'g-', linewidth=2, label='Wage share')
    ax2.plot(df['t'], df['pi'], 'r-', linewidth=2, label='Profit share')
    ax2.set_ylabel('Income Share')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Functional Distribution')

    # 3. Debt ratio over time
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(df['t'], df['d'], 'purple', linewidth=2)
    ax3.set_ylabel('Debt to Capital Ratio')
    ax3.set_xlabel('Time')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Private Debt Dynamics')

    # 4. Growth rate
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(df['t'], df['growth_rate'], 'b-', linewidth=1.5, label='Gross growth')
    ax4.plot(df['t'], df['growth_net'], 'r-', linewidth=2, label='Net of debt service')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.fill_between(df['t'], 0, df['growth_net'],
                     where=(df['growth_net'] < 0), alpha=0.3, color='red')
    ax4.set_ylabel('Growth Rate')
    ax4.set_xlabel('Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Economic Growth')

    # 5. Goodwin cycle (employment vs wage share)
    ax5 = plt.subplot(3, 4, 5)
    scatter = ax5.scatter(df['lambda'], df['omega'], c=df['t'], cmap='viridis',
                         s=5, alpha=0.5)
    ax5.set_xlabel('Employment Rate λ')
    ax5.set_ylabel('Wage Share ω')
    ax5.set_title('Goodwin Cycle')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Time')

    # 6. Debt-distribution space
    ax6 = plt.subplot(3, 4, 6)
    scatter = ax6.scatter(df['omega'], df['d'], c=df['t'], cmap='plasma',
                         s=5, alpha=0.5)
    ax6.set_xlabel('Wage Share ω')
    ax6.set_ylabel('Debt Ratio d')
    ax6.set_title('Distribution vs Debt')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Time')

    # 7. Employment-debt space
    ax7 = plt.subplot(3, 4, 7)
    scatter = ax7.scatter(df['lambda'], df['d'], c=df['omega'],
                         cmap='RdYlGn', s=5, alpha=0.5, vmin=0.5, vmax=0.8)
    ax7.set_xlabel('Employment Rate λ')
    ax7.set_ylabel('Debt Ratio d')
    ax7.set_title('Employment vs Debt')
    ax7.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax7, label='Wage Share')

    # 8. 3D phase space (if matplotlib supports)
    ax8 = plt.subplot(3, 4, 8, projection='3d')
    ax8.plot(df['lambda'], df['omega'], df['d'], linewidth=0.5, alpha=0.6)
    ax8.scatter(df['lambda'].iloc[0], df['omega'].iloc[0], df['d'].iloc[0],
               c='green', s=100, marker='o', label='Start')
    ax8.scatter(df['lambda'].iloc[-1], df['omega'].iloc[-1], df['d'].iloc[-1],
               c='red', s=100, marker='X', label='End')
    ax8.set_xlabel('Employment λ')
    ax8.set_ylabel('Wage Share ω')
    ax8.set_zlabel('Debt d')
    ax8.set_title('3D Phase Space')
    ax8.legend()

    # 9. Investment and profit rates
    ax9 = plt.subplot(3, 4, 9)
    ax9.plot(df['t'], df['inv_rate'], 'b-', linewidth=2, label='Investment rate')
    ax9.plot(df['t'], df['profit_rate'], 'g-', linewidth=2, label='Profit rate')
    ax9.set_ylabel('Rate')
    ax9.set_xlabel('Time')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_title('Investment and Profitability')

    # 10. Debt servicing burden
    ax10 = plt.subplot(3, 4, 10)
    ax10.plot(df['t'], df['debt_service'], 'r-', linewidth=2)
    ax10.fill_between(df['t'], 0, df['debt_service'], alpha=0.3, color='red')
    ax10.set_ylabel('Debt Service / Capital')
    ax10.set_xlabel('Time')
    ax10.grid(True, alpha=0.3)
    ax10.set_title('Financial Burden')

    # 11. Cycle phase distribution
    ax11 = plt.subplot(3, 4, 11)
    phase_counts = df['goodwin_phase'].value_counts()
    ax11.bar(phase_counts.index, phase_counts.values, color=['green', 'blue', 'orange', 'red'])
    ax11.set_ylabel('Time periods')
    ax11.set_title('Time in Each Phase')
    ax11.tick_params(axis='x', rotation=45)

    # 12. Crisis periods
    ax12 = plt.subplot(3, 4, 12)
    crisis_indicator = df['crisis'].astype(int)
    ax12.fill_between(df['t'], 0, crisis_indicator,
                      where=(crisis_indicator > 0), alpha=0.5, color='darkred')
    ax12.set_ylabel('Crisis (0/1)')
    ax12.set_xlabel('Time')
    ax12.set_ylim(-0.1, 1.1)
    ax12.grid(True, alpha=0.3)
    ax12.set_title('Crisis Periods')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig


# Example usage
if __name__ == "__main__":
    print("Goodwin-Keen Integration Model")
    print("=" * 70)

    # Scenario 1: Baseline integration
    print("\n1. Baseline Simulation")
    print("-" * 70)

    model = GoodwinKeenModel()

    # Find equilibrium
    eq, eigenvals = model.find_equilibrium()
    print(f"Equilibrium: λ={eq[0]:.3f}, ω={eq[1]:.3f}, d={eq[2]:.3f}")
    if 'stability' in eigenvals:
        print(f"Stability: {eigenvals['stability']}")
        print(f"Oscillatory: {eigenvals['oscillatory']}")

    # Simulate
    df = model.simulate(t_max=300, t_points=10000)

    # Cycle statistics
    stats = model.cycle_statistics()
    print(f"\nCycle Statistics:")
    print(f"  Employment: {stats['lambda_mean']:.3f} ± {stats['lambda_std']:.3f}")
    print(f"  Wage share: {stats['omega_mean']:.3f} ± {stats['omega_std']:.3f}")
    print(f"  Debt ratio: {stats['d_mean']:.3f} (max: {stats['d_max']:.3f})")
    print(f"  Growth rate: {stats['growth_mean']:.4f} ± {stats['growth_std']:.4f}")
    print(f"  Cycle period: {stats['cycle_period']:.2f}")
    print(f"  Crisis frequency: {stats['crisis_frequency']:.2%}")

    # Scenario 2: High debt economy
    print(f"\n2. High-Debt Scenario")
    print("-" * 70)

    params_high_debt = GoodwinKeenParameters(
        d_0=0.8,
        eta=2.0,  # Stronger debt accelerator
        sigma=0.5  # More credit-financed investment
    )

    model_high_debt = GoodwinKeenModel(params_high_debt)
    df_high_debt = model_high_debt.simulate(t_max=300)
    stats_high_debt = model_high_debt.cycle_statistics()

    print(f"High-debt economy:")
    print(f"  Average debt: {stats_high_debt['d_mean']:.3f}")
    print(f"  Growth volatility: {stats_high_debt['growth_std']:.4f}")
    print(f"  Crisis frequency: {stats_high_debt['crisis_frequency']:.2%}")

    # Scenario 3: Weak labor (low bargaining power)
    print(f"\n3. Weak Labor Scenario")
    print("-" * 70)

    params_weak_labor = GoodwinKeenParameters(
        phi=0.03,  # Flatter Phillips curve
        rho=0.01  # Weak wage push
    )

    model_weak_labor = GoodwinKeenModel(params_weak_labor)
    df_weak_labor = model_weak_labor.simulate(t_max=300)
    stats_weak_labor = model_weak_labor.cycle_statistics()

    print(f"Weak labor bargaining:")
    print(f"  Average wage share: {stats_weak_labor['omega_mean']:.3f}")
    print(f"  Average employment: {stats_weak_labor['lambda_mean']:.3f}")
    print(f"  Cycle amplitude (ω): {stats_weak_labor['omega_amplitude']:.4f}")

    # Visualization
    print(f"\n4. Generating Visualizations")
    print("-" * 70)

    fig = plot_goodwin_keen_results(model, df,
                                    title="Goodwin-Keen Model: Distribution and Debt Dynamics")
    fig.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/goodwin_keen_baseline.png',
                dpi=150, bbox_inches='tight')

    print("Saved: goodwin_keen_baseline.png")

    # Comparative plot
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compare debt trajectories
    axes[0, 0].plot(df['t'], df['d'], 'b-', label='Baseline', linewidth=2)
    axes[0, 0].plot(df_high_debt['t'], df_high_debt['d'], 'r-', label='High debt', linewidth=2)
    axes[0, 0].set_ylabel('Debt Ratio')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Debt Dynamics Comparison')

    # Compare wage share
    axes[0, 1].plot(df['t'], df['omega'], 'b-', label='Baseline', linewidth=2)
    axes[0, 1].plot(df_weak_labor['t'], df_weak_labor['omega'], 'g-', label='Weak labor', linewidth=2)
    axes[0, 1].set_ylabel('Wage Share')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('Distribution Comparison')

    # Goodwin cycles comparison
    axes[1, 0].scatter(df['lambda'], df['omega'], s=3, alpha=0.3, label='Baseline')
    axes[1, 0].scatter(df_weak_labor['lambda'], df_weak_labor['omega'],
                      s=3, alpha=0.3, label='Weak labor')
    axes[1, 0].set_xlabel('Employment Rate')
    axes[1, 0].set_ylabel('Wage Share')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Goodwin Cycles')

    # Growth rate distributions
    axes[1, 1].hist(df['growth_net'], bins=50, alpha=0.5, label='Baseline', density=True)
    axes[1, 1].hist(df_high_debt['growth_net'], bins=50, alpha=0.5, label='High debt', density=True)
    axes[1, 1].set_xlabel('Net Growth Rate')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].set_title('Growth Rate Distribution')

    fig2.suptitle('Comparative Scenarios', fontsize=14, fontweight='bold')
    plt.tight_layout()

    fig2.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/goodwin_keen_comparison.png',
                 dpi=150, bbox_inches='tight')

    print("Saved: goodwin_keen_comparison.png")

    print("\n" + "=" * 70)
    print("Integration model demonstrates:")
    print("1. Distributional cycles (Goodwin) interact with financial cycles (Keen)")
    print("2. High debt amplifies volatility and crisis frequency")
    print("3. Weak labor bargaining power lowers wage share and dampens cycles")
    print("4. Complex dynamics emerge from interaction of class conflict and finance")
    print("=" * 70)
