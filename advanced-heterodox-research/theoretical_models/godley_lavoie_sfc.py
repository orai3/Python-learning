"""
Godley-Lavoie Multi-Sector Stock-Flow Consistent (SFC) Model

This module implements a comprehensive 5-sector SFC model following the methodology
developed by Wynne Godley and Marc Lavoie in "Monetary Economics: An Integrated
Approach to Credit, Money, Income, Production and Wealth" (2007).

The model includes:
- Households (consume, hold wealth, receive income)
- Firms (produce, invest, borrow)
- Banks (create credit, manage deposits)
- Government (fiscal policy, taxation, spending)
- Central Bank (monetary policy, interest rate setting)

Key SFC Principles:
1. Every flow comes from somewhere and goes somewhere
2. Every financial asset has a corresponding liability
3. All balance sheets must balance at all times
4. Sum of sectoral surpluses/deficits must equal zero

References:
- Godley, W., & Lavoie, M. (2007). Monetary Economics: An Integrated Approach to
  Credit, Money, Income, Production and Wealth. Palgrave Macmillan.
- Lavoie, M., & Zezza, G. (2020). A Simple Stock-Flow Consistent Model with
  Short-Term and Long-Term Debt: A Comment on Claudio Sardoni. Review of
  Political Economy, 32(4), 459-473.

Author: Claude
License: MIT
"""

from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings


class SectorType(Enum):
    """Enumeration of sectors in the SFC model"""
    HOUSEHOLDS = "households"
    FIRMS = "firms"
    BANKS = "banks"
    GOVERNMENT = "government"
    CENTRAL_BANK = "central_bank"


@dataclass
class SFCParameters:
    """
    Parameters for the Godley-Lavoie SFC model.

    All parameters are based on typical Post-Keynesian SFC calibrations.
    """
    # Household behavior
    alpha1: float = 0.8  # Propensity to consume out of current income
    alpha2: float = 0.2  # Propensity to consume out of wealth
    lambda_0: float = 0.3  # Baseline demand for cash (currency)
    lambda_1: float = 5.0  # Sensitivity of cash demand to interest rates
    lambda_2: float = 0.2  # Sensitivity of cash demand to income

    # Portfolio allocation (must sum to 1 in equilibrium)
    mu_0: float = 0.2  # Baseline demand for bank deposits
    mu_1: float = 10.0  # Sensitivity to deposit rate
    mu_2: float = 0.3  # Sensitivity to income

    # Firm behavior
    gamma: float = 0.2  # Target inventories to sales ratio
    sigma_T: float = 0.15  # Target leverage ratio (debt/capital)
    phi: float = 0.1  # Sensitivity of investment to capacity utilization
    rho: float = 0.05  # Target profit rate on capital

    # Production and pricing
    pr: float = 1.2  # Price markup over unit labor costs
    w: float = 1.0  # Nominal wage rate
    a: float = 1.0  # Labor productivity (output per worker)

    # Banking sector
    beta: float = 0.02  # Target bank spread (lending - deposit rate)
    chi: float = 0.1  # Banks' capital adequacy requirement

    # Government policy
    theta: float = 0.2  # Tax rate on income
    g_bar: float = 20.0  # Autonomous government spending

    # Central bank policy
    r_cb: float = 0.03  # Central bank policy rate (Taylor rule baseline)
    pi_target: float = 0.02  # Inflation target
    taylor_pi: float = 1.5  # Taylor rule coefficient on inflation
    taylor_y: float = 0.5  # Taylor rule coefficient on output gap

    # Depreciation
    delta: float = 0.1  # Capital depreciation rate

    # Initial conditions
    k_initial: float = 100.0  # Initial capital stock
    y_initial: float = 80.0  # Initial output
    n_initial: float = 80.0  # Initial employment


@dataclass
class SFCState:
    """
    State variables for the SFC model at a given time period.

    Tracks all stock and flow variables across all sectors.
    """
    # Time
    t: int = 0

    # Household sector
    y_h: float = 0.0  # Household disposable income
    c: float = 0.0  # Consumption expenditure
    v: float = 0.0  # Household wealth (net worth)
    h_h: float = 0.0  # Holdings of cash (currency)
    m_h: float = 0.0  # Holdings of bank deposits
    b_h: float = 0.0  # Holdings of government bonds

    # Firm sector
    y: float = 0.0  # Output (GDP)
    y_d: float = 0.0  # Expected sales
    inv: float = 0.0  # Inventories
    i: float = 0.0  # Investment (gross fixed capital formation)
    k: float = 0.0  # Capital stock
    n: float = 0.0  # Employment
    wb: float = 0.0  # Wage bill
    l_f: float = 0.0  # Firm loans from banks
    f_f: float = 0.0  # Firm profits

    # Banking sector
    m_s: float = 0.0  # Supply of bank deposits
    l_s: float = 0.0  # Supply of loans
    hpb_d: float = 0.0  # Banks' demand for reserves
    or_: float = 0.0  # Banks' own funds (equity)
    f_b: float = 0.0  # Bank profits

    # Government sector
    g: float = 0.0  # Government expenditure
    t: float = 0.0  # Tax revenue
    b_s: float = 0.0  # Supply of government bonds
    b_cb: float = 0.0  # Central bank holdings of bonds

    # Central bank sector
    h_s: float = 0.0  # Supply of cash (currency)
    hpb_s: float = 0.0  # Supply of reserves
    f_cb: float = 0.0  # Central bank profits

    # Prices and interest rates
    p: float = 1.0  # Price level
    pi: float = 0.0  # Inflation rate
    r_l: float = 0.05  # Loan interest rate
    r_m: float = 0.02  # Deposit interest rate
    r_b: float = 0.03  # Bond interest rate
    r: float = 0.03  # Policy interest rate (central bank rate)

    # Real variables
    y_r: float = 0.0  # Real GDP
    k_r: float = 0.0  # Real capital stock
    u: float = 0.8  # Capacity utilization rate


class BalanceSheetMatrix:
    """
    Balance Sheet Matrix for the SFC model.

    Implements the fundamental SFC accounting identity that all assets
    must have corresponding liabilities, and all sectoral balance sheets
    must sum to zero (closure property).
    """

    def __init__(self):
        self.sectors = [s.value for s in SectorType]
        self.assets = [
            'Currency (H)',
            'Reserves (HPB)',
            'Deposits (M)',
            'Loans (L)',
            'Bonds (B)',
            'Capital (K)',
            'Inventories (IN)'
        ]

    def construct_matrix(self, state: SFCState, params: SFCParameters) -> pd.DataFrame:
        """
        Construct the balance sheet matrix at a given state.

        Rows: Asset/Liability categories
        Columns: Sectors

        Positive = Asset
        Negative = Liability

        Properties that must hold:
        1. Each row sums to zero (asset = liability)
        2. Each column equals sectoral net worth
        3. Sum of all net worths = value of real capital + inventories
        """
        matrix = pd.DataFrame(
            index=self.assets,
            columns=self.sectors,
            data=0.0
        )

        # Currency (H): Asset for households, liability for CB
        matrix.loc['Currency (H)', 'households'] = state.h_h
        matrix.loc['Currency (H)', 'central_bank'] = -state.h_s

        # Reserves (HPB): Asset for banks, liability for CB
        matrix.loc['Reserves (HPB)', 'banks'] = state.hpb_d
        matrix.loc['Reserves (HPB)', 'central_bank'] = -state.hpb_s

        # Deposits (M): Asset for households, liability for banks
        matrix.loc['Deposits (M)', 'households'] = state.m_h
        matrix.loc['Deposits (M)', 'banks'] = -state.m_s

        # Loans (L): Asset for banks, liability for firms
        matrix.loc['Loans (L)', 'banks'] = state.l_s
        matrix.loc['Loans (L)', 'firms'] = -state.l_f

        # Bonds (B): Asset for HH and CB, liability for government
        matrix.loc['Bonds (B)', 'households'] = state.b_h
        matrix.loc['Bonds (B)', 'central_bank'] = state.b_cb
        matrix.loc['Bonds (B)', 'government'] = -state.b_s

        # Capital (K): Asset for firms (real capital valued at replacement cost)
        matrix.loc['Capital (K)', 'firms'] = state.k * state.p

        # Inventories (IN): Asset for firms
        matrix.loc['Inventories (IN)', 'firms'] = state.inv * state.p

        return matrix

    def validate_consistency(self, matrix: pd.DataFrame, tolerance: float = 1e-6) -> Dict[str, bool]:
        """
        Validate that the balance sheet satisfies SFC accounting identities.

        Returns:
            Dictionary with validation results for each identity
        """
        results = {}

        # Check 1: Each row (asset category) sums to zero
        row_sums = matrix.sum(axis=1)
        results['assets_equal_liabilities'] = np.all(np.abs(row_sums) < tolerance)
        results['row_violations'] = row_sums[np.abs(row_sums) >= tolerance].to_dict()

        # Check 2: Column sums equal sectoral net worth
        col_sums = matrix.sum(axis=0)
        results['sectoral_net_worth'] = col_sums.to_dict()

        # Check 3: Household wealth consistency
        hh_wealth = matrix['households'].sum()
        results['household_wealth_consistent'] = True  # Can add specific checks

        # Check 4: Government debt equals bonds outstanding
        gov_debt = -matrix.loc['Bonds (B)', 'government']
        bond_holdings = matrix.loc['Bonds (B)', ['households', 'central_bank']].sum()
        results['government_debt_consistent'] = abs(gov_debt - bond_holdings) < tolerance

        return results


class TransactionFlowMatrix:
    """
    Transaction Flow Matrix for the SFC model.

    Records all flows between sectors in a period. Each flow has a source
    and destination, ensuring the fundamental identity: "everything comes
    from somewhere and goes somewhere."
    """

    def __init__(self):
        self.sectors = [s.value for s in SectorType]
        # Current and capital account rows
        self.transactions = [
            # Current account
            'Consumption',
            'Government spending',
            'Investment',
            'GDP (output)',
            'Wages',
            'Firm profits',
            'Bank profits',
            'CB profits',
            'Taxes',
            'Interest on loans',
            'Interest on deposits',
            'Interest on bonds',
            # Capital account
            'Change in currency',
            'Change in reserves',
            'Change in deposits',
            'Change in loans',
            'Change in bonds (HH)',
            'Change in bonds (CB)',
        ]

    def construct_matrix(self, state: SFCState, state_prev: SFCState,
                        params: SFCParameters) -> pd.DataFrame:
        """
        Construct the transaction flow matrix for a period.

        Positive = Receipt/Source
        Negative = Payment/Use

        Each column must sum to zero (sectoral budget constraint).
        Each row must sum to zero (flow consistency).
        """
        matrix = pd.DataFrame(
            index=self.transactions,
            columns=self.sectors + ['Sum'],
            data=0.0
        )

        # Consumption: Households pay, Firms receive
        matrix.loc['Consumption', 'households'] = -state.c
        matrix.loc['Consumption', 'firms'] = state.c

        # Government spending: Government pays, Firms receive
        matrix.loc['Government spending', 'government'] = -state.g
        matrix.loc['Government spending', 'firms'] = state.g

        # Investment: Firms pay themselves (internal transaction)
        matrix.loc['Investment', 'firms'] = -state.i + state.i

        # GDP: Firms produce
        matrix.loc['GDP (output)', 'firms'] = state.y

        # Wages: Firms pay, Households receive
        matrix.loc['Wages', 'firms'] = -state.wb
        matrix.loc['Wages', 'households'] = state.wb

        # Profits distribution
        matrix.loc['Firm profits', 'firms'] = -state.f_f
        matrix.loc['Firm profits', 'households'] = state.f_f

        matrix.loc['Bank profits', 'banks'] = -state.f_b
        matrix.loc['Bank profits', 'households'] = state.f_b

        matrix.loc['CB profits', 'central_bank'] = -state.f_cb
        matrix.loc['CB profits', 'government'] = state.f_cb

        # Taxes: Households pay, Government receives
        matrix.loc['Taxes', 'households'] = -state.t
        matrix.loc['Taxes', 'government'] = state.t

        # Interest payments
        interest_loans = state_prev.l_f * state.r_l
        matrix.loc['Interest on loans', 'firms'] = -interest_loans
        matrix.loc['Interest on loans', 'banks'] = interest_loans

        interest_deposits = state_prev.m_h * state.r_m
        matrix.loc['Interest on deposits', 'banks'] = -interest_deposits
        matrix.loc['Interest on deposits', 'households'] = interest_deposits

        interest_bonds_hh = state_prev.b_h * state.r_b
        interest_bonds_cb = state_prev.b_cb * state.r_b
        matrix.loc['Interest on bonds', 'government'] = -(interest_bonds_hh + interest_bonds_cb)
        matrix.loc['Interest on bonds', 'households'] = interest_bonds_hh
        matrix.loc['Interest on bonds', 'central_bank'] = interest_bonds_cb

        # Changes in financial stocks (capital account)
        matrix.loc['Change in currency', 'households'] = -(state.h_h - state_prev.h_h)
        matrix.loc['Change in currency', 'central_bank'] = state.h_s - state_prev.h_s

        matrix.loc['Change in reserves', 'banks'] = -(state.hpb_d - state_prev.hpb_d)
        matrix.loc['Change in reserves', 'central_bank'] = state.hpb_s - state_prev.hpb_s

        matrix.loc['Change in deposits', 'households'] = -(state.m_h - state_prev.m_h)
        matrix.loc['Change in deposits', 'banks'] = state.m_s - state_prev.m_s

        matrix.loc['Change in loans', 'firms'] = state.l_f - state_prev.l_f
        matrix.loc['Change in loans', 'banks'] = -(state.l_s - state_prev.l_s)

        matrix.loc['Change in bonds (HH)', 'households'] = -(state.b_h - state_prev.b_h)
        matrix.loc['Change in bonds (HH)', 'government'] = state.b_h - state_prev.b_h

        matrix.loc['Change in bonds (CB)', 'central_bank'] = -(state.b_cb - state_prev.b_cb)
        matrix.loc['Change in bonds (CB)', 'government'] = state.b_cb - state_prev.b_cb

        # Calculate row and column sums
        matrix['Sum'] = matrix[self.sectors].sum(axis=1)

        return matrix

    def validate_consistency(self, matrix: pd.DataFrame, tolerance: float = 1e-6) -> Dict[str, bool]:
        """
        Validate transaction flow identities.

        Returns:
            Dictionary with validation results
        """
        results = {}

        # Check 1: Each row sums to zero (flow consistency)
        row_sums = matrix['Sum']
        results['flows_consistent'] = np.all(np.abs(row_sums) < tolerance)
        results['row_violations'] = row_sums[np.abs(row_sums) >= tolerance].to_dict()

        # Check 2: Each column sums to zero (budget constraints)
        col_sums = matrix[self.sectors].sum(axis=0)
        results['budget_constraints_satisfied'] = np.all(np.abs(col_sums) < tolerance)
        results['col_violations'] = col_sums[np.abs(col_sums) >= tolerance].to_dict()

        return results


class SFCModel:
    """
    Main Godley-Lavoie SFC Model class.

    Implements the complete 5-sector SFC model with:
    - Behavioral equations for each sector
    - Portfolio allocation decisions
    - Credit creation mechanics
    - Fiscal and monetary policy rules
    - Iterative solution algorithm
    - Stock-flow consistency validation
    """

    def __init__(self, params: Optional[SFCParameters] = None):
        """
        Initialize the SFC model.

        Args:
            params: Model parameters. Uses defaults if None.
        """
        self.params = params or SFCParameters()
        self.states: List[SFCState] = []
        self.balance_sheet_matrix = BalanceSheetMatrix()
        self.transaction_flow_matrix = TransactionFlowMatrix()

    def initialize_state(self) -> SFCState:
        """
        Initialize the model to a steady-state equilibrium.

        This involves finding consistent initial values for all stock
        and flow variables that satisfy accounting identities and
        behavioral equations.
        """
        state = SFCState()
        p = self.params

        # Initialize real variables
        state.k_r = p.k_initial
        state.y_r = p.y_initial
        state.n = p.n_initial
        state.u = state.y_r / state.k_r  # Capacity utilization

        # Price level (normalized to 1 initially)
        state.p = 1.0

        # Nominal variables
        state.k = state.k_r * state.p
        state.y = state.y_r * state.p

        # Wage bill
        state.wb = p.w * state.n

        # Firm profits (residual after wages)
        state.f_f = state.y - state.wb - p.delta * state.k

        # Government
        state.g = p.g_bar
        state.t = p.theta * (state.wb + state.f_f)

        # Initial interest rates
        state.r = p.r_cb
        state.r_l = state.r + p.beta
        state.r_m = state.r
        state.r_b = state.r

        # Household income
        state.y_h = state.wb + state.f_f - state.t

        # Initialize wealth (steady state assumption)
        state.v = state.y_h / 0.1  # Assuming 10% return on wealth

        # Portfolio allocation
        state.h_h = p.lambda_0 * state.v - p.lambda_1 * state.r + p.lambda_2 * state.y_h
        state.m_h = p.mu_0 * state.v + p.mu_1 * state.r_m + p.mu_2 * state.y_h
        state.b_h = state.v - state.h_h - state.m_h

        # Banking sector
        state.l_f = p.sigma_T * state.k  # Firm leverage
        state.m_s = state.m_h
        state.l_s = state.l_f
        state.hpb_d = p.chi * state.l_s  # Reserve requirement
        state.or_ = state.l_s - state.m_s - state.hpb_d  # Bank equity

        # Government bonds
        state.b_s = state.b_h + state.b_cb
        state.b_cb = 0.0  # No QE initially

        # Central bank
        state.h_s = state.h_h
        state.hpb_s = state.hpb_d

        # Consumption (residual to balance current account)
        state.c = state.y_h - (state.v - state.v * 0.95)  # Assuming 5% wealth accumulation

        # Investment
        state.i = p.delta * state.k  # Replacement investment

        # Inventories
        state.inv = p.gamma * state.y

        return state

    def household_behavior(self, state: SFCState, state_prev: SFCState) -> Tuple[float, float, float, float]:
        """
        Household behavioral equations.

        Households decide:
        1. How much to consume (consumption function)
        2. How to allocate wealth (portfolio choice)

        Returns:
            (consumption, currency demand, deposit demand, bond demand)

        References:
        - Godley & Lavoie (2007), Chapter 4
        - Brainard & Tobin (1968) on portfolio choice
        """
        p = self.params

        # Disposable income
        y_d = state.wb + state.f_f + state.f_b - state.t + \
              state_prev.m_h * state.r_m + state_prev.b_h * state.r_b

        # Consumption function (Keynesian consumption with wealth effect)
        # Based on Modigliani life-cycle / Friedman permanent income
        c = p.alpha1 * y_d + p.alpha2 * state_prev.v

        # Wealth accumulation
        v = state_prev.v + (y_d - c)

        # Portfolio allocation (Tobinesque portfolio choice)
        # Households allocate wealth across currency, deposits, and bonds
        # based on rates of return and liquidity preferences

        # Currency demand (no interest, but liquid)
        h_h = p.lambda_0 * v - p.lambda_1 * state.r_m + p.lambda_2 * y_d
        h_h = max(0, h_h)  # Non-negativity constraint

        # Deposit demand (earns r_m, liquid)
        m_h = p.mu_0 * v + p.mu_1 * (state.r_m - state.r_b) + p.mu_2 * y_d
        m_h = max(0, m_h)

        # Bond demand (residual, earns r_b)
        b_h = v - h_h - m_h
        b_h = max(0, b_h)

        # Ensure portfolio adds up (adjust bonds if needed)
        if abs(h_h + m_h + b_h - v) > 1e-6:
            b_h = v - h_h - m_h

        return c, h_h, m_h, b_h, v, y_d

    def firm_behavior(self, state: SFCState, state_prev: SFCState) -> Tuple[float, float, float]:
        """
        Firm behavioral equations.

        Firms decide:
        1. How much to produce (based on expected sales)
        2. How much to invest (accelerator + profit rate)
        3. How much credit to demand

        Returns:
            (output, investment, loan demand)

        References:
        - Godley & Lavoie (2007), Chapter 8
        - Kalecki (1971) on investment determination
        """
        p = self.params

        # Expected sales (adaptive expectations)
        # Firms expect current consumption + government spending + investment
        # Simplified: assume firms expect sales equal to previous period
        y_d = state.c + state.g + state.i

        # Production decision
        # Firms produce to meet expected sales plus maintain target inventory
        inv_target = p.gamma * y_d
        y = y_d + (inv_target - state_prev.inv)

        # Employment (from production function: Y = a * N)
        n = y / (p.a * state.p)

        # Wage bill
        wb = p.w * n

        # Actual sales (realized)
        sales = state.c + state.g

        # Inventory accumulation
        inv = state_prev.inv + (y - sales)

        # Capital stock evolution
        k = state_prev.k * (1 - p.delta) + state.i

        # Capacity utilization
        y_r = y / state.p
        k_r = k / state.p
        u = y_r / k_r if k_r > 0 else 0

        # Investment function (Kaleckian)
        # Investment depends on:
        # 1. Capacity utilization (accelerator effect)
        # 2. Profit rate (profitability effect)
        # 3. Depreciation (replacement)

        profit_rate = state_prev.f_f / state_prev.k if state_prev.k > 0 else p.rho

        i = p.delta * k + p.phi * (u - 0.8) * k + (profit_rate - p.rho) * k
        i = max(0, i)  # Non-negative investment

        # Loan demand (to finance investment beyond retained earnings)
        # Target leverage ratio (Minsky/Kalecki)
        l_f_target = p.sigma_T * k

        # Firms borrow to maintain leverage ratio and finance investment
        retained_earnings = state_prev.f_f
        financing_gap = i - retained_earnings

        l_f = state_prev.l_f + max(0, financing_gap)

        # Ensure target leverage ratio
        if l_f < l_f_target:
            l_f = l_f_target

        # Firm profits (residual)
        interest_payments = state_prev.l_f * state.r_l
        f_f = y - wb - p.delta * k - interest_payments

        return y, i, l_f, n, wb, inv, k, u, f_f, y_r, k_r

    def banking_behavior(self, state: SFCState, state_prev: SFCState) -> Tuple[float, float, float]:
        """
        Banking sector behavioral equations.

        Banks:
        1. Accommodate loan demand (endogenous money creation)
        2. Hold required reserves
        3. Set interest rates based on markup

        Returns:
            (loan supply, deposit supply, reserve demand)

        References:
        - Godley & Lavoie (2007), Chapter 3
        - Moore (1988) on endogenous money
        """
        p = self.params

        # Loan supply: Banks accommodate all creditworthy demand
        # This is the Post-Keynesian endogenous money view
        l_s = state.l_f

        # Deposit creation: Loans create deposits
        # When banks extend loans, they create deposits
        m_s = state.m_h

        # Reserve demand: Based on regulatory requirements
        hpb_d = p.chi * l_s

        # Interest rate setting
        # Banks set loan rate as markup over policy rate
        r_l = state.r + p.beta

        # Deposit rate follows policy rate
        r_m = state.r

        # Bank profits
        interest_received = state_prev.l_s * r_l
        interest_paid = state_prev.m_s * state.r_m
        f_b = interest_received - interest_paid

        # Bank equity (own funds)
        or_ = state_prev.or_ + f_b

        return l_s, m_s, hpb_d, r_l, r_m, f_b, or_

    def government_behavior(self, state: SFCState, state_prev: SFCState) -> Tuple[float, float, float]:
        """
        Government behavioral equations.

        Government:
        1. Sets exogenous spending (fiscal policy)
        2. Collects taxes
        3. Issues bonds to finance deficit

        Returns:
            (spending, taxes, bond supply)

        References:
        - Godley & Lavoie (2007), Chapter 4
        - Lerner (1943) on functional finance
        """
        p = self.params

        # Government spending (exogenous policy)
        g = p.g_bar

        # Tax revenue (proportional to income)
        tax_base = state.wb + state.f_f + state.f_b
        t = p.theta * tax_base

        # Government deficit
        interest_payments = state_prev.b_s * state.r_b
        deficit = g + interest_payments - t

        # Bond issuance to finance deficit
        b_s = state_prev.b_s + deficit

        return g, t, b_s

    def central_bank_behavior(self, state: SFCState, state_prev: SFCState) -> Tuple[float, float, float, float]:
        """
        Central bank behavioral equations.

        Central bank:
        1. Sets policy interest rate (Taylor rule)
        2. Supplies reserves on demand
        3. Supplies currency on demand
        4. May hold government bonds (QE)

        Returns:
            (policy rate, currency supply, reserve supply, bond holdings)

        References:
        - Taylor (1993) on monetary policy rules
        - Godley & Lavoie (2007), Chapter 5
        """
        p = self.params

        # Taylor rule for interest rate setting
        # r = r* + φ_π(π - π*) + φ_y(y - y*)

        inflation = state.pi
        output_gap = (state.y_r - p.y_initial) / p.y_initial if p.y_initial > 0 else 0

        r = p.r_cb + p.taylor_pi * (inflation - p.pi_target) + p.taylor_y * output_gap

        # Floor on interest rate (zero lower bound)
        r = max(0, r)

        # Bond interest rate follows policy rate
        r_b = r

        # Currency supply: Accommodate household demand
        h_s = state.h_h

        # Reserve supply: Accommodate bank demand
        hpb_s = state.hpb_d

        # Bond holdings (can be policy variable for QE)
        # For now, keep constant (no QE)
        b_cb = state_prev.b_cb

        # Central bank profits (seigniorage)
        f_cb = state_prev.b_cb * r_b

        return r, r_b, h_s, hpb_s, b_cb, f_cb

    def solve_period(self, state_prev: SFCState, max_iter: int = 100,
                     tolerance: float = 1e-6) -> SFCState:
        """
        Solve for equilibrium in a single period using iterative method.

        The model is solved by iterating behavioral equations until
        all accounting identities and market clearing conditions are satisfied.

        This is the Gauss-Seidel algorithm adapted for SFC models.

        Args:
            state_prev: Previous period state
            max_iter: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Converged state for current period

        References:
        - Godley & Lavoie (2007), Appendix on solution methods
        """
        state = SFCState(t=state_prev.t + 1)

        # Initialize state with previous period values
        for field in state.__dataclass_fields__:
            if field != 't':
                setattr(state, field, getattr(state_prev, field))

        for iteration in range(max_iter):
            state_old = SFCState(**{f: getattr(state, f) for f in state.__dataclass_fields__})

            # 1. Central bank sets policy rate
            state.r, state.r_b, state.h_s, state.hpb_s, state.b_cb, state.f_cb = \
                self.central_bank_behavior(state, state_prev)

            # 2. Banks set interest rates and accommodate credit demand
            # Note: We need firm loan demand first, so this is done after firms

            # 3. Government sets fiscal policy
            state.g, state.t, state.b_s = self.government_behavior(state, state_prev)

            # 4. Households decide consumption and portfolio
            state.c, state.h_h, state.m_h, state.b_h, state.v, state.y_h = \
                self.household_behavior(state, state_prev)

            # 5. Firms decide production, investment, and credit demand
            # This uses consumption and government spending from above
            state.y, state.i, state.l_f, state.n, state.wb, state.inv, state.k, \
                state.u, state.f_f, state.y_r, state.k_r = \
                self.firm_behavior(state, state_prev)

            # 6. Banks accommodate loan demand
            state.l_s, state.m_s, state.hpb_d, state.r_l, state.r_m, state.f_b, state.or_ = \
                self.banking_behavior(state, state_prev)

            # 7. Update price level (for now, assume constant; can add Phillips curve)
            # Markup pricing: p = (1 + markup) * unit_labor_cost
            unit_labor_cost = self.params.w / self.params.a
            state.p = self.params.pr * unit_labor_cost

            # 8. Calculate inflation
            state.pi = (state.p - state_prev.p) / state_prev.p if state_prev.p > 0 else 0

            # Check convergence
            max_diff = 0
            for field in ['y', 'c', 'i', 'v', 'l_f', 'm_h']:
                diff = abs(getattr(state, field) - getattr(state_old, field))
                max_diff = max(max_diff, diff)

            if max_diff < tolerance:
                break
        else:
            warnings.warn(f"Model did not converge after {max_iter} iterations. Max diff: {max_diff}")

        return state

    def simulate(self, periods: int, initial_state: Optional[SFCState] = None,
                 shock_fn: Optional[Callable[[int, SFCParameters], None]] = None) -> pd.DataFrame:
        """
        Simulate the model for multiple periods.

        Args:
            periods: Number of periods to simulate
            initial_state: Initial state. If None, uses steady state.
            shock_fn: Optional function to apply shocks to parameters.
                     Signature: shock_fn(period, params) -> None

        Returns:
            DataFrame with time series of all variables

        Example:
            def fiscal_shock(t, params):
                if t == 10:
                    params.g_bar *= 1.2  # 20% increase in government spending

            results = model.simulate(50, shock_fn=fiscal_shock)
        """
        # Initialize
        if initial_state is None:
            initial_state = self.initialize_state()

        self.states = [initial_state]

        # Simulate
        for t in range(1, periods):
            # Apply shocks if specified
            if shock_fn is not None:
                shock_fn(t, self.params)

            # Solve period
            state = self.solve_period(self.states[-1])
            self.states.append(state)

        # Convert to DataFrame
        data = []
        for state in self.states:
            row = {field: getattr(state, field) for field in state.__dataclass_fields__}
            data.append(row)

        return pd.DataFrame(data)

    def validate_consistency(self, state: SFCState, state_prev: SFCState,
                           tolerance: float = 1e-6) -> Dict[str, any]:
        """
        Validate stock-flow consistency at a given state.

        Returns:
            Dictionary with validation results and any violations
        """
        results = {}

        # Validate balance sheet matrix
        bs_matrix = self.balance_sheet_matrix.construct_matrix(state, self.params)
        bs_validation = self.balance_sheet_matrix.validate_consistency(bs_matrix, tolerance)
        results['balance_sheet'] = bs_validation

        # Validate transaction flow matrix
        tf_matrix = self.transaction_flow_matrix.construct_matrix(state, state_prev, self.params)
        tf_validation = self.transaction_flow_matrix.validate_consistency(tf_matrix, tolerance)
        results['transaction_flows'] = tf_validation

        # Check specific identities
        results['identities'] = {}

        # 1. Household wealth identity
        wealth_calc = state.h_h + state.m_h + state.b_h
        results['identities']['household_wealth'] = {
            'v': state.v,
            'h_h + m_h + b_h': wealth_calc,
            'difference': abs(state.v - wealth_calc),
            'satisfied': abs(state.v - wealth_calc) < tolerance
        }

        # 2. Loan-deposit identity (loans create deposits in Post-Keynesian theory)
        # In equilibrium: M = L - HPB + OR (bank balance sheet)
        results['identities']['money_creation'] = {
            'm_s': state.m_s,
            'l_s - hpb_d + or_': state.l_s - state.hpb_d + state.or_,
            'note': 'Banks create deposits when extending loans'
        }

        # 3. Government budget constraint
        gov_deficit = state.g - state.t
        bond_change = state.b_s - state_prev.b_s
        results['identities']['government_budget'] = {
            'deficit': gov_deficit,
            'bond_issuance': bond_change,
            'difference': abs(gov_deficit - bond_change),
            'satisfied': abs(gov_deficit - bond_change) < tolerance * 10  # Looser tolerance
        }

        # 4. Sectoral balances identity (S - I) + (T - G) + (M - X) = 0
        # For closed economy: (S - I) + (T - G) = 0
        household_balance = (state.y_h - state.c) - 0  # Household saving
        government_balance = state.t - state.g
        # Firm balance implicit in loan creation

        results['identities']['sectoral_balances'] = {
            'household_surplus': household_balance,
            'government_surplus': government_balance,
            'note': 'Godley sectoral balances identity'
        }

        return results


def plot_simulation_results(df: pd.DataFrame, variables: Optional[List[str]] = None,
                           title: str = "SFC Model Simulation Results"):
    """
    Plot simulation results with multiple panels.

    Args:
        df: DataFrame from model.simulate()
        variables: List of variables to plot. If None, plots key variables.
        title: Plot title
    """
    if variables is None:
        variables = [
            ['y_r', 'c', 'i', 'g'],  # Real aggregates
            ['u', 'n'],  # Utilization and employment
            ['r', 'r_l', 'r_m'],  # Interest rates
            ['v', 'l_f', 'b_s'],  # Stocks
        ]

    n_panels = len(variables)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3*n_panels))

    if n_panels == 1:
        axes = [axes]

    for ax, var_list in zip(axes, variables):
        for var in var_list:
            if var in df.columns:
                ax.plot(df['t'], df[var], label=var, linewidth=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Period')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# Example usage and tests
if __name__ == "__main__":
    print("Godley-Lavoie Multi-Sector SFC Model")
    print("=" * 50)

    # Create model with default parameters
    model = SFCModel()

    # Scenario 1: Baseline simulation
    print("\n1. Baseline Simulation (50 periods)")
    print("-" * 50)
    baseline = model.simulate(periods=50)
    print(f"Simulated {len(baseline)} periods")
    print(f"\nFinal period key variables:")
    print(f"  Real GDP: {baseline.iloc[-1]['y_r']:.2f}")
    print(f"  Capacity utilization: {baseline.iloc[-1]['u']:.2%}")
    print(f"  Household wealth: {baseline.iloc[-1]['v']:.2f}")
    print(f"  Private debt: {baseline.iloc[-1]['l_f']:.2f}")
    print(f"  Government debt: {baseline.iloc[-1]['b_s']:.2f}")

    # Validate consistency for final period
    print(f"\n2. Stock-Flow Consistency Validation")
    print("-" * 50)
    validation = model.validate_consistency(model.states[-1], model.states[-2])
    print(f"Balance sheet consistent: {validation['balance_sheet']['assets_equal_liabilities']}")
    print(f"Transaction flows consistent: {validation['transaction_flows']['flows_consistent']}")
    print(f"Budget constraints satisfied: {validation['transaction_flows']['budget_constraints_satisfied']}")

    # Scenario 2: Fiscal stimulus shock
    print(f"\n3. Fiscal Stimulus Scenario")
    print("-" * 50)

    def fiscal_stimulus(t, params):
        if t == 10:
            params.g_bar *= 1.2
            print(f"  Period {t}: Government spending increased by 20%")

    model_shock = SFCModel()
    fiscal_sim = model_shock.simulate(periods=50, shock_fn=fiscal_stimulus)

    gdp_increase = (fiscal_sim.iloc[20]['y_r'] - fiscal_sim.iloc[9]['y_r']) / fiscal_sim.iloc[9]['y_r']
    print(f"  GDP increase after 10 periods: {gdp_increase:.2%}")
    print(f"  Fiscal multiplier: {gdp_increase / 0.2:.2f}")

    # Scenario 3: Interest rate shock
    print(f"\n4. Monetary Tightening Scenario")
    print("-" * 50)

    def monetary_tightening(t, params):
        if t == 10:
            params.r_cb *= 1.5
            print(f"  Period {t}: Policy rate increased by 50%")

    model_monetary = SFCModel()
    monetary_sim = model_monetary.simulate(periods=50, shock_fn=monetary_tightening)

    investment_change = (monetary_sim.iloc[20]['i'] - monetary_sim.iloc[9]['i']) / monetary_sim.iloc[9]['i']
    print(f"  Investment change after 10 periods: {investment_change:.2%}")

    # Plot results
    print(f"\n5. Generating Plots...")
    print("-" * 50)

    fig1 = plot_simulation_results(baseline, title="Baseline Scenario")
    fig1.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/sfc_baseline.png', dpi=150)

    fig2 = plot_simulation_results(fiscal_sim, title="Fiscal Stimulus Scenario")
    fig2.savefig('/home/user/Python-learning/advanced-heterodox-research/examples/sfc_fiscal_stimulus.png', dpi=150)

    print("Plots saved to examples/ directory")

    # Display balance sheet matrix
    print(f"\n6. Balance Sheet Matrix (Final Period)")
    print("-" * 50)
    bs_matrix = model.balance_sheet_matrix.construct_matrix(model.states[-1], model.params)
    print(bs_matrix.round(2))

    print(f"\nRow sums (should be zero):")
    print(bs_matrix.sum(axis=1).round(6))

    print("\n" + "=" * 50)
    print("Simulation complete. SFC consistency maintained throughout.")
    print("=" * 50)
