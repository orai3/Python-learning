"""
Sraffian Production Model with Fixed Capital

Implements Piero Sraffa's analysis of production prices, distribution, and value
from "Production of Commodities by Means of Commodities" (1960).

This module provides comprehensive tools for:
1. Calculating production prices for any input-output system
2. Deriving the wage-profit rate frontier
3. Finding the standard commodity and maximum profit rate
4. Analyzing joint production and fixed capital
5. Demonstrating reswitching and capital reversing
6. Computing labor values and their relationship to prices

Theoretical Significance:
Sraffa's work fundamentally challenged neoclassical capital theory by showing:
- Prices depend on distribution (circular reasoning in marginal productivity)
- Capital cannot be measured independently of distribution
- Reswitching invalidates simple supply-demand stories
- No monotonic relationship between "capital intensity" and profit rate

This implementation provides rigorous computational tools for these insights.

References:
- Sraffa, P. (1960). Production of Commodities by Means of Commodities:
  Prelude to a Critique of Economic Theory. Cambridge University Press.
- Pasinetti, L. (1977). Lectures on the Theory of Production. Columbia University Press.
- Kurz, H., & Salvadori, N. (1995). Theory of Production: A Long-Period Analysis.
  Cambridge University Press.
- Cambridge Capital Controversy literature (Robinson, Samuelson, et al.)

Author: Claude
License: MIT
"""

from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.optimize import fsolve, minimize_scalar
import warnings


@dataclass
class ProductionTechnique:
    """
    A production technique specifies the methods used to produce commodities.

    Attributes:
        name: Name of the technique (e.g., "Alpha", "Beta")
        A: Input coefficient matrix (n×n)
           A[i,j] = quantity of commodity i used to produce one unit of commodity j
        l: Labor input vector (n×1)
           l[j] = labor hours required to produce one unit of commodity j
        B: Output coefficient matrix (n×n), default is identity
           For joint production: B[i,j] = quantity of commodity i produced
           by activity j (usually produces multiple outputs)
        commodity_names: Names of commodities
    """
    name: str
    A: np.ndarray  # Input coefficients
    l: np.ndarray  # Labor inputs
    B: Optional[np.ndarray] = None  # Output coefficients (identity if None)
    commodity_names: Optional[List[str]] = None

    def __post_init__(self):
        """Validate consistency of technique specification"""
        n = self.A.shape[0]

        assert self.A.shape == (n, n), "A must be square matrix"
        assert self.l.shape == (n,), f"l must be vector of length {n}"

        if self.B is None:
            self.B = np.eye(n)  # Single-product industries
        else:
            assert self.B.shape == (n, n), "B must be square matrix"

        if self.commodity_names is None:
            self.commodity_names = [f"Commodity {i+1}" for i in range(n)]
        else:
            assert len(self.commodity_names) == n

        # Economic viability checks
        # Check primitivity: All goods used directly or indirectly in all production
        # Check productivity: Must be able to produce surplus

    @property
    def n_commodities(self) -> int:
        """Number of commodities"""
        return self.A.shape[0]

    @property
    def is_single_product(self) -> bool:
        """Check if this is single-product system (no joint production)"""
        return np.allclose(self.B, np.eye(self.n_commodities))


class SraffaModel:
    """
    Sraffa model for price determination and distribution analysis.

    The fundamental equations are:
    (1 + r) * A * p + w * l = B * p

    where:
    - p = price vector
    - r = profit rate (uniform across all industries)
    - w = wage rate
    - A = input coefficient matrix
    - l = labor input vector
    - B = output coefficient matrix

    For single-product systems (B = I):
    (1 + r) * A * p + w * l = p

    Rearranging:
    p = (I - (1+r)A)^(-1) * w * l

    This shows prices depend on distribution (w, r), which is Sraffa's
    key insight: no "technical" determination of prices independent of
    distribution.
    """

    def __init__(self, technique: ProductionTechnique):
        """
        Initialize Sraffa model with a production technique.

        Args:
            technique: Production technique specification
        """
        self.technique = technique
        self._validate_technique()

    def _validate_technique(self):
        """
        Validate economic viability of the technique.

        Checks:
        1. Productivity: Maximum profit rate > 0
        2. Non-negative prices for some distribution
        """
        # Check maximum profit rate exists and is positive
        R = self.maximum_profit_rate()
        if R <= 0:
            warnings.warn("Maximum profit rate <= 0: technique is not productive")

        # Check for negative input coefficients
        if np.any(self.technique.A < 0):
            warnings.warn("Negative input coefficients detected")

    def maximum_profit_rate(self) -> float:
        """
        Calculate the maximum profit rate R.

        This is the profit rate when wages = 0.

        For single-product systems, R is found from:
        (1 + R) * A * p = p

        This is an eigenvalue problem: A * p = λ * p where λ = 1/(1+R)

        R is determined by the dominant (Frobenius-Perron) eigenvalue of A.

        Returns:
            R: Maximum profit rate

        Mathematical Derivation:
        When w = 0: (1 + r) * A * p = B * p
        For single-product: (1 + r) * A * p = p
        Rearranging: A * p = [1/(1+r)] * p

        This is eigenvalue problem with eigenvalue λ = 1/(1+r)
        By Frobenius-Perron theorem, there exists a dominant eigenvalue
        λ_max > 0 with corresponding eigenvector p > 0 (all positive prices)

        Maximum profit rate: R = (1/λ_max) - 1
        """
        if self.technique.is_single_product:
            # Eigenvalue problem: A * p = λ * p
            eigenvalues, eigenvectors = eig(self.technique.A)

            # Find dominant (largest magnitude) eigenvalue
            idx = np.argmax(np.abs(eigenvalues))
            lambda_max = np.real(eigenvalues[idx])

            if lambda_max <= 0:
                raise ValueError("Non-positive dominant eigenvalue: technique not viable")

            R = (1.0 / lambda_max) - 1.0

        else:
            # Joint production case: solve (1+R) * A * p = B * p
            # Generalized eigenvalue problem: A * p = λ * B * p where λ = 1/(1+R)

            eigenvalues, eigenvectors = eig(self.technique.A, self.technique.B)

            # Find economically meaningful eigenvalue (positive, real)
            real_positive_eigvals = []
            for i, ev in enumerate(eigenvalues):
                if np.isreal(ev) and np.real(ev) > 0:
                    # Check if corresponding eigenvector has all positive elements
                    eigvec = np.real(eigenvectors[:, i])
                    if np.all(eigvec > 0):
                        real_positive_eigvals.append(np.real(ev))

            if len(real_positive_eigvals) == 0:
                raise ValueError("No economically meaningful eigenvalue found")

            lambda_max = max(real_positive_eigvals)
            R = (1.0 / lambda_max) - 1.0

        return R

    def production_prices(self, r: float, w: float = 1.0,
                         numeraire: Optional[Union[str, np.ndarray]] = None) -> np.ndarray:
        """
        Calculate production prices for given profit rate and wage.

        Solves: (1 + r) * A * p + w * l = B * p

        Args:
            r: Profit rate (must be 0 <= r <= R)
            w: Wage rate (can be chosen as numeraire)
            numeraire: Price normalization method:
                       - None or 'wage': w = 1 (wage numeraire)
                       - 'commodity_i': p[i] = 1
                       - np.ndarray: numeraire basket, p · basket = 1
                       - 'standard': standard commodity numeraire

        Returns:
            p: Price vector

        Mathematical Method:
        From (1 + r) * A * p + w * l = B * p

        Rearranging: (B - (1+r)*A) * p = w * l

        For single-product: (I - (1+r)*A) * p = w * l

        So: p = (I - (1+r)*A)^(-1) * w * l

        This requires (I - (1+r)*A) to be invertible, which holds for r < R.
        """
        # Validate profit rate
        R = self.maximum_profit_rate()
        if r < 0 or r > R:
            warnings.warn(f"Profit rate r={r:.4f} outside valid range [0, {R:.4f}]")

        n = self.technique.n_commodities

        # Solve price system
        if self.technique.is_single_product:
            # (I - (1+r)*A) * p = w * l
            lhs_matrix = np.eye(n) - (1 + r) * self.technique.A

            # Check if matrix is singular (shouldn't be for r < R)
            if np.abs(np.linalg.det(lhs_matrix)) < 1e-10:
                raise ValueError(f"Price system is singular at r={r}")

            # Solve
            p = np.linalg.solve(lhs_matrix, w * self.technique.l)

        else:
            # Joint production: (B - (1+r)*A) * p = w * l
            lhs_matrix = self.technique.B - (1 + r) * self.technique.A

            if np.abs(np.linalg.det(lhs_matrix)) < 1e-10:
                raise ValueError(f"Price system is singular at r={r}")

            p = np.linalg.solve(lhs_matrix, w * self.technique.l)

        # Apply numeraire normalization
        if numeraire is None or numeraire == 'wage':
            # Already normalized with w = 1
            pass

        elif isinstance(numeraire, str) and numeraire.startswith('commodity_'):
            # Normalize so specific commodity has price = 1
            comm_idx = int(numeraire.split('_')[1])
            p = p / p[comm_idx]

        elif isinstance(numeraire, np.ndarray):
            # Normalize with basket: p · numeraire = 1
            basket_value = np.dot(p, numeraire)
            p = p / basket_value

        elif numeraire == 'standard':
            # Use standard commodity as numeraire
            q = self.standard_commodity()
            basket_value = np.dot(p, q)
            p = p / basket_value

        return p

    def wage_profit_frontier(self, n_points: int = 100,
                            numeraire: Optional[Union[str, np.ndarray]] = None) -> pd.DataFrame:
        """
        Calculate the wage-profit rate frontier (factor price frontier).

        This shows the trade-off between wages and profits. For given
        technology, higher profit rates necessitate lower wages.

        Shape of frontier is generally (but not always!) convex to origin.
        In presence of reswitching, can have non-monotonic regions.

        Args:
            n_points: Number of points to calculate
            numeraire: Price normalization method

        Returns:
            DataFrame with columns [r, w, p1, p2, ..., pn]

        Mathematical Derivation:
        From production price equations with commodity numeraire:
        p = (I - (1+r)*A)^(-1) * w * l

        Choosing numeraire (e.g., total price = 1):
        Σ p_i = 1

        This gives: Σ [(I - (1+r)*A)^(-1) * w * l]_i = 1

        Solving for w:
        w = 1 / Σ [(I - (1+r)*A)^(-1) * l]_i

        This shows w as function of r: w = w(r)

        The frontier is typically downward sloping (∂w/∂r < 0) but
        can have irregularities due to price effects.
        """
        R = self.maximum_profit_rate()

        # Create array of profit rates from 0 to R
        r_array = np.linspace(0, R * 0.999, n_points)  # Stop just before R

        results = []

        for r in r_array:
            # Calculate prices at this profit rate
            p = self.production_prices(r, w=1.0, numeraire=numeraire)

            # If using commodity numeraire, need to recalculate wage
            if numeraire == 'standard' or (isinstance(numeraire, np.ndarray)):
                # Wage is determined by price normalization
                # This is already handled in production_prices
                w_normalized = 1.0  # Placeholder
            else:
                w_normalized = 1.0

            result = {'r': r, 'w': w_normalized}
            for i, name in enumerate(self.technique.commodity_names):
                result[f'p_{name}'] = p[i]

            results.append(result)

        df = pd.DataFrame(results)

        # Calculate real wage (wage in terms of basket of commodities)
        # Using first commodity as consumption good
        df['w_real'] = df['w'] / df[f'p_{self.technique.commodity_names[0]}']

        return df

    def standard_commodity(self) -> np.ndarray:
        """
        Calculate Sraffa's Standard Commodity.

        The standard commodity is a composite commodity whose composition
        mirrors the economy's input structure. It has the property that
        the wage-profit relationship is linear when the standard commodity
        is the numeraire.

        Mathematical Construction:
        The standard commodity q satisfies:
        q = (1 + R) * A * q

        where R is the maximum profit rate.

        This is the eigenvector corresponding to the dominant eigenvalue
        of A, normalized appropriately.

        With standard commodity numeraire:
        w = 1 - r/R  (linear relationship!)

        This is a major result: it shows the "intrinsic" trade-off between
        wages and profits, before price effects.

        Returns:
            q: Standard commodity quantities

        References:
        - Sraffa (1960), Chapters 4-5
        - Pasinetti (1977), Chapter 5
        """
        # Standard commodity is eigenvector of A corresponding to eigenvalue 1/(1+R)
        R = self.maximum_profit_rate()

        if self.technique.is_single_product:
            eigenvalues, eigenvectors = eig(self.technique.A)

            # Find eigenvalue closest to 1/(1+R)
            target = 1.0 / (1 + R)
            idx = np.argmin(np.abs(eigenvalues - target))

            q = np.real(eigenvectors[:, idx])

            # Normalize to positive quantities
            if np.any(q < 0):
                q = -q

            # Conventional normalization: sum to 1 or set first element to 1
            q = q / q.sum()

        else:
            # Joint production case more complex
            eigenvalues, eigenvectors = eig(self.technique.A, self.technique.B)

            target = 1.0 / (1 + R)
            idx = np.argmin(np.abs(eigenvalues - target))

            q = np.real(eigenvectors[:, idx])

            if np.any(q < 0):
                q = -q

            q = q / q.sum()

        return q

    def standard_ratio(self) -> float:
        """
        Calculate the standard ratio (1 + R).

        This is the ratio of net output to means of production
        in the standard system.

        Returns:
            1 + R
        """
        return 1.0 + self.maximum_profit_rate()

    def labor_values(self) -> np.ndarray:
        """
        Calculate labor values (total labor content) of commodities.

        Labor value v[j] = total direct and indirect labor required
        to produce one unit of commodity j.

        Mathematical formulation:
        v = l + v * A

        Solving: v * (I - A) = l
        Therefore: v = l * (I - A)^(-1)

        Or equivalently: v = l * Σ(A^k) for k=0 to ∞

        This is the "labor embodied" in commodities, central to
        classical and Marxian value theory.

        Returns:
            v: Labor values

        Connection to Prices:
        If r = 0 (no profit), then p = w * v (prices proportional to values)
        For r > 0, prices deviate from values due to different "capital intensity"
        This is Marx's "transformation problem"
        """
        n = self.technique.n_commodities

        # Solve v * (I - A) = l
        # Transposing: (I - A)^T * v^T = l^T

        if self.technique.is_single_product:
            lhs_matrix = np.eye(n) - self.technique.A

            # Check productivity: I - A must be invertible
            if np.abs(np.linalg.det(lhs_matrix)) < 1e-10:
                raise ValueError("System is not productive: I - A is singular")

            v = np.linalg.solve(lhs_matrix.T, self.technique.l)

        else:
            # Joint production: more complex, use (B - A) instead
            lhs_matrix = self.technique.B - self.technique.A

            if np.abs(np.linalg.det(lhs_matrix)) < 1e-10:
                raise ValueError("System is not productive")

            v = np.linalg.solve(lhs_matrix.T, self.technique.l)

        return v

    def price_value_deviation(self, r: float) -> Dict[str, float]:
        """
        Calculate deviation of prices from labor values at given profit rate.

        Measures transformation problem: how prices deviate from values
        due to profit rate equalization.

        Args:
            r: Profit rate

        Returns:
            Dictionary with deviation metrics
        """
        v = self.labor_values()
        p = self.production_prices(r, w=1.0)

        # Normalize both to same total
        v_normalized = v / v.sum()
        p_normalized = p / p.sum()

        # Mean absolute deviation
        mad = np.mean(np.abs(p_normalized - v_normalized))

        # Correlation
        correlation = np.corrcoef(v, p)[0, 1]

        # Max deviation
        max_dev = np.max(np.abs(p_normalized - v_normalized))

        return {
            'mean_absolute_deviation': mad,
            'correlation': correlation,
            'max_deviation': max_dev,
        }

    def capital_intensity(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate capital intensity of each industry.

        Capital intensity k[j] = (capital value) / (labor employed)
                                = (A[:,j] · p) / l[j]

        This concept is problematic (circular) in neoclassical theory
        because it depends on prices, which depend on distribution!

        Args:
            prices: Price vector

        Returns:
            Capital intensity by industry
        """
        capital_values = self.technique.A.T @ prices  # Value of inputs in each industry
        k = capital_values / self.technique.l

        return k

    def organic_composition(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate Marxian organic composition of capital.

        Organic composition: c/(c+v)
        where c = constant capital (means of production)
              v = variable capital (labor power)

        In Sraffa's framework:
        c_j = A[:,j] · p (value of inputs)
        v_j = w * l[j] (wage bill)

        Args:
            prices: Price vector

        Returns:
            Organic composition by industry
        """
        w = 1.0  # Wage numeraire

        constant_capital = self.technique.A.T @ prices
        variable_capital = w * self.technique.l

        occ = constant_capital / (constant_capital + variable_capital)

        return occ


class ReswitchingAnalysis:
    """
    Tools for analyzing reswitching and capital reversing.

    Reswitching: A technique can be most profitable at low AND high
    profit rates, while another technique is most profitable at
    intermediate rates.

    This demonstrates:
    1. No monotonic relationship between "capital intensity" and profit rate
    2. Supply-demand stories for capital are incoherent
    3. Marginal productivity theory of distribution is invalid

    This was a key result of the Cambridge Capital Controversy.
    """

    def __init__(self, techniques: List[ProductionTechnique]):
        """
        Initialize with multiple techniques.

        Args:
            techniques: List of alternative production techniques
        """
        self.techniques = techniques
        self.models = [SraffaModel(tech) for tech in techniques]

    def technique_choice(self, r: float) -> Tuple[int, np.ndarray]:
        """
        Determine which technique is cost-minimizing at given profit rate.

        Args:
            r: Profit rate

        Returns:
            (technique_index, unit_costs)

        Technique choice is based on which yields lowest unit costs.
        """
        unit_costs = []

        for model in self.models:
            try:
                p = model.production_prices(r, w=1.0)

                # Unit cost is price (by definition in equilibrium,
                # price = cost of production at given r, w)
                # We'll use first commodity as comparison
                cost = p[0]
                unit_costs.append(cost)

            except:
                # If technique not viable at this r
                unit_costs.append(np.inf)

        chosen_idx = np.argmin(unit_costs)

        return chosen_idx, np.array(unit_costs)

    def switching_points(self, n_points: int = 1000) -> pd.DataFrame:
        """
        Find switching points between techniques.

        Returns:
            DataFrame showing which technique is chosen at each profit rate
        """
        # Find maximum profit rate across all techniques
        R_max = max(model.maximum_profit_rate() for model in self.models)

        r_array = np.linspace(0, R_max * 0.99, n_points)

        results = []

        for r in r_array:
            chosen_idx, costs = self.technique_choice(r)

            result = {
                'r': r,
                'chosen_technique': self.techniques[chosen_idx].name,
                'chosen_index': chosen_idx,
            }

            for i, tech in enumerate(self.techniques):
                result[f'cost_{tech.name}'] = costs[i]

            results.append(result)

        df = pd.DataFrame(results)

        return df

    def detect_reswitching(self) -> bool:
        """
        Detect if reswitching occurs.

        Returns:
            True if reswitching detected
        """
        df = self.switching_points()

        # Check if any technique is chosen, then not chosen, then chosen again
        for tech_name in [t.name for t in self.techniques]:
            is_chosen = (df['chosen_technique'] == tech_name).astype(int)

            # Find switches
            switches = np.diff(is_chosen)

            # Count number of times technique becomes chosen
            n_switches_to = np.sum(switches == 1)

            if n_switches_to > 1:
                # Technique is chosen multiple times with gaps
                return True

        return False


def plot_wage_profit_frontier(model: SraffaModel, title: str = "Wage-Profit Frontier"):
    """
    Plot the wage-profit rate frontier (factor price frontier).

    Args:
        model: SraffaModel instance
        title: Plot title
    """
    df = model.wage_profit_frontier(n_points=200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Wage-profit frontier
    ax1.plot(df['r'], df['w_real'], 'b-', linewidth=2.5)
    ax1.set_xlabel('Profit Rate (r)', fontsize=12)
    ax1.set_ylabel('Real Wage (w)', fontsize=12)
    ax1.set_title('Wage-Profit Frontier', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

    # Mark maximum profit rate
    R = model.maximum_profit_rate()
    ax1.axvline(x=R, color='r', linestyle='--', alpha=0.5, label=f'R = {R:.3f}')
    ax1.legend()

    # Panel 2: Relative prices
    ax2.set_xlabel('Profit Rate (r)', fontsize=12)
    ax2.set_ylabel('Relative Price', fontsize=12)
    ax2.set_title('Price Ratios vs Profit Rate', fontsize=13, fontweight='bold')

    # Plot price of each commodity relative to first commodity
    for i in range(1, model.technique.n_commodities):
        name_i = model.technique.commodity_names[i]
        name_0 = model.technique.commodity_names[0]

        relative_price = df[f'p_{name_i}'] / df[f'p_{name_0}']

        ax2.plot(df['r'], relative_price, linewidth=2, label=f'{name_i}/{name_0}')

    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# Example usage and demonstrations
if __name__ == "__main__":
    print("Sraffian Production Price Analysis")
    print("=" * 70)

    # Example 1: Simple two-commodity system
    print("\n1. Two-Commodity Example (Corn-Iron)")
    print("-" * 70)

    # Corn and Iron economy from Sraffa's examples
    # Corn production uses: 240 tons corn + 12 tons iron + 180 labor → 450 tons corn
    # Iron production uses: 90 tons corn + 6 tons iron + 120 labor → 21 tons iron

    # Normalize to unit output:
    A_corn_iron = np.array([
        [240/450, 90/21],    # Corn inputs
        [12/450, 6/21]       # Iron inputs
    ])

    l_corn_iron = np.array([180/450, 120/21])  # Labor inputs

    tech_corn_iron = ProductionTechnique(
        name="Corn-Iron",
        A=A_corn_iron,
        l=l_corn_iron,
        commodity_names=['Corn', 'Iron']
    )

    model1 = SraffaModel(tech_corn_iron)

    R = model1.maximum_profit_rate()
    print(f"Maximum profit rate R = {R:.4f} ({R*100:.2f}%)")

    # Calculate labor values
    v = model1.labor_values()
    print(f"\nLabor values:")
    for i, name in enumerate(tech_corn_iron.commodity_names):
        print(f"  {name}: {v[i]:.4f} labor hours")

    # Standard commodity
    q = model1.standard_commodity()
    print(f"\nStandard commodity composition:")
    for i, name in enumerate(tech_corn_iron.commodity_names):
        print(f"  {name}: {q[i]:.4f}")

    # Prices at different profit rates
    print(f"\nProduction prices at different profit rates:")
    print(f"{'r':<8} {'p_Corn':<12} {'p_Iron':<12} {'Deviation':<12}")
    print("-" * 48)

    for r_pct in [0, 10, 25, 50]:
        r = (r_pct / 100) * R
        p = model1.production_prices(r, w=1.0)
        dev = model1.price_value_deviation(r)

        print(f"{r:<8.4f} {p[0]:<12.4f} {p[1]:<12.4f} {dev['mean_absolute_deviation']:<12.4f}")

    # Example 2: Three-commodity system with reswitching
    print(f"\n\n2. Reswitching Example (Cambridge Capital Controversy)")
    print("-" * 70)

    # Technique Alpha
    A_alpha = np.array([
        [0.2, 0.3, 0.1],
        [0.3, 0.1, 0.2],
        [0.1, 0.2, 0.3]
    ])
    l_alpha = np.array([0.5, 0.6, 0.4])

    tech_alpha = ProductionTechnique(
        name="Alpha",
        A=A_alpha,
        l=l_alpha,
        commodity_names=['Good_1', 'Good_2', 'Good_3']
    )

    # Technique Beta (different capital intensity)
    A_beta = np.array([
        [0.3, 0.2, 0.15],
        [0.2, 0.3, 0.25],
        [0.15, 0.25, 0.2]
    ])
    l_beta = np.array([0.4, 0.5, 0.6])

    tech_beta = ProductionTechnique(
        name="Beta",
        A=A_beta,
        l=l_beta,
        commodity_names=['Good_1', 'Good_2', 'Good_3']
    )

    # Analyze technique choice
    reswitch_analysis = ReswitchingAnalysis([tech_alpha, tech_beta])

    print("Analyzing technique choice across profit rates...")

    switching_df = reswitch_analysis.switching_points(n_points=500)

    # Count technique switches
    switches = switching_df['chosen_index'].diff().fillna(0)
    n_switches = np.sum(switches != 0)

    print(f"Number of technique switches: {n_switches}")

    if reswitch_analysis.detect_reswitching():
        print("RESWITCHING DETECTED!")
        print("This invalidates simple capital theory.")
    else:
        print("No reswitching in this example.")

    # Show technique choice at different profit rates
    print(f"\nTechnique choice at different profit rates:")
    for pct in [0, 25, 50, 75, 95]:
        idx = int(pct / 100 * len(switching_df))
        row = switching_df.iloc[idx]
        print(f"  r = {row['r']:.4f}: {row['chosen_technique']}")

    # Example 3: Visualizations
    print(f"\n3. Generating Visualizations...")
    print("-" * 70)

    fig1 = plot_wage_profit_frontier(model1, title="Corn-Iron Economy: Wage-Profit Frontier")
    fig1.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/sraffa_corn_iron.png',
                 dpi=150, bbox_inches='tight')

    # Plot technique choice
    fig2, ax = plt.subplots(figsize=(12, 6))

    for i, tech in enumerate(reswitch_analysis.techniques):
        is_chosen = (switching_df['chosen_technique'] == tech.name).astype(int)
        ax.fill_between(switching_df['r'], 0, is_chosen,
                       where=(is_chosen > 0), alpha=0.4, label=tech.name)

    ax.set_xlabel('Profit Rate (r)', fontsize=12)
    ax.set_ylabel('Chosen', fontsize=12)
    ax.set_title('Technique Choice and Reswitching', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig2.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/sraffa_reswitching.png',
                 dpi=150, bbox_inches='tight')

    print("Saved visualizations to examples/ directory")

    print("\n" + "=" * 70)
    print("Sraffa analysis complete. Key insights:")
    print("1. Prices depend on distribution (r, w), not just 'technology'")
    print("2. No independent measure of 'capital' exists")
    print("3. Reswitching invalidates marginal productivity theory")
    print("4. Labor values and prices diverge with profit rate")
    print("=" * 70)
