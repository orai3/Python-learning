"""
Economic Calculations Utilities

Provides calculation methods for various heterodox economic indicators.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy import stats


class EconomicCalculations:
    """
    Collection of economic calculation methods with proper academic citations.
    """

    @staticmethod
    def calculate_gini(data: np.ndarray) -> float:
        """
        Calculate Gini coefficient.

        Formula: G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n

        Args:
            data: Array of income/wealth values

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)

        Reference:
        - Gini, C. (1912). Variability and Mutability.
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return np.nan

        # Sort data
        sorted_data = np.sort(data)
        n = len(sorted_data)

        # Calculate Gini
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_data)) / (n * np.sum(sorted_data)) - (n + 1) / n

        return gini

    @staticmethod
    def calculate_palma_ratio(data: np.ndarray) -> float:
        """
        Calculate Palma ratio: ratio of top 10% to bottom 40% income share.

        The Palma ratio captures changes in inequality better than Gini
        for policy purposes.

        Args:
            data: Array of income values

        Returns:
            Palma ratio

        Reference:
        - Palma, J. G. (2011). Homogeneous middles vs. heterogeneous tails.
          Development and Change, 42(1), 87-153.
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]

        if len(data) == 0:
            return np.nan

        sorted_data = np.sort(data)
        n = len(sorted_data)

        # Top 10%
        top_10_idx = int(n * 0.9)
        top_10_share = np.sum(sorted_data[top_10_idx:]) / np.sum(sorted_data)

        # Bottom 40%
        bottom_40_idx = int(n * 0.4)
        bottom_40_share = np.sum(sorted_data[:bottom_40_idx]) / np.sum(sorted_data)

        if bottom_40_share == 0:
            return np.nan

        return top_10_share / bottom_40_share

    @staticmethod
    def calculate_theil_index(data: np.ndarray) -> float:
        """
        Calculate Theil entropy index.

        Useful for decomposing inequality between and within groups.

        Args:
            data: Array of income values

        Returns:
            Theil index

        Reference:
        - Theil, H. (1967). Economics and Information Theory.
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]
        data = data[data > 0]  # Theil requires positive values

        if len(data) == 0:
            return np.nan

        mean_income = np.mean(data)
        n = len(data)

        theil = np.sum((data / mean_income) * np.log(data / mean_income)) / n

        return theil

    @staticmethod
    def calculate_rate_of_profit(profits: pd.Series,
                                 capital_stock: pd.Series) -> pd.Series:
        """
        Calculate Marxian rate of profit.

        r = Surplus Value / (Constant Capital + Variable Capital)
        Approximation: r = Profits / Capital Stock

        Args:
            profits: Time series of profits
            capital_stock: Time series of capital stock

        Returns:
            Rate of profit time series

        Reference:
        - Marx, K. (1894/1991). Capital, Volume III. Penguin Classics.
        - Shaikh, A. (2016). Capitalism: Competition, Conflict, Crises.
        """
        return (profits / capital_stock) * 100

    @staticmethod
    def calculate_rate_of_surplus_value(profits: pd.Series,
                                       wages: pd.Series) -> pd.Series:
        """
        Calculate rate of surplus value (rate of exploitation).

        e = Surplus Value / Variable Capital
        Approximation: e = Profits / Wages

        Args:
            profits: Time series of profits
            wages: Time series of wages

        Returns:
            Rate of surplus value

        Reference:
        - Marx, K. (1867/1990). Capital, Volume I.
        """
        return (profits / wages) * 100

    @staticmethod
    def calculate_organic_composition(capital_stock: pd.Series,
                                     employment: pd.Series) -> pd.Series:
        """
        Calculate organic composition of capital.

        OCC = Constant Capital / Variable Capital
        Approximation: Capital per Worker

        Args:
            capital_stock: Time series of capital stock
            employment: Time series of employment

        Returns:
            Organic composition of capital

        Reference:
        - Marx, K. (1867/1990). Capital, Volume I.
        """
        return capital_stock / employment

    @staticmethod
    def calculate_capacity_utilization(output: pd.Series,
                                      potential_output: pd.Series) -> pd.Series:
        """
        Calculate capacity utilization rate.

        CU = Actual Output / Potential Output * 100

        Key indicator for Post-Keynesian analysis of demand constraints.

        Args:
            output: Actual output
            potential_output: Potential/full capacity output

        Returns:
            Capacity utilization rate

        Reference:
        - Lavoie, M. (2014). Post-Keynesian Economics: New Foundations.
        """
        return (output / potential_output) * 100

    @staticmethod
    def kalecki_profit_equation(investment: float,
                                capitalist_consumption: float,
                                worker_saving: float,
                                government_deficit: float = 0,
                                trade_surplus: float = 0) -> float:
        """
        Calculate profits using Kalecki's profit equation.

        Profits = Investment + Capitalist Consumption - Worker Saving
                 + Government Deficit + Trade Surplus

        "Workers spend what they earn; capitalists earn what they spend."

        Args:
            investment: Gross investment
            capitalist_consumption: Consumption by capitalists
            worker_saving: Saving by workers
            government_deficit: Government deficit
            trade_surplus: Trade surplus

        Returns:
            Total profits

        Reference:
        - Kalecki, M. (1971). Selected Essays on the Dynamics of the
          Capitalist Economy.
        """
        profits = (investment + capitalist_consumption - worker_saving +
                  government_deficit + trade_surplus)
        return profits

    @staticmethod
    def check_sectoral_balances_identity(private_balance: float,
                                         government_balance: float,
                                         foreign_balance: float,
                                         tolerance: float = 0.01) -> Tuple[bool, float]:
        """
        Verify Godley's sectoral balances identity.

        Private + Government + Foreign = 0 (accounting identity)

        Args:
            private_balance: Private sector balance
            government_balance: Government sector balance
            foreign_balance: Foreign sector balance
            tolerance: Acceptable deviation

        Returns:
            Tuple of (identity_holds, sum_of_balances)

        Reference:
        - Godley, W., & Lavoie, M. (2007). Monetary Economics.
        """
        balance_sum = private_balance + government_balance + foreign_balance
        identity_holds = abs(balance_sum) < tolerance

        return identity_holds, balance_sum

    @staticmethod
    def calculate_goodwin_cycle_variables(wage_share: pd.Series,
                                         employment_rate: pd.Series) -> Dict:
        """
        Calculate variables for Goodwin growth cycle analysis.

        The Goodwin model shows cyclical dynamics between distribution
        and employment.

        Args:
            wage_share: Time series of wage share
            employment_rate: Time series of employment rate

        Returns:
            Dictionary with cycle characteristics

        Reference:
        - Goodwin, R. M. (1967). A growth cycle. In C.H. Feinstein (Ed.),
          Socialism, Capitalism and Economic Growth.
        """
        results = {}

        # Calculate correlation
        if len(wage_share) == len(employment_rate):
            correlation = wage_share.corr(employment_rate)
            results['correlation'] = correlation

        # Identify cycles using peaks and troughs
        from scipy.signal import find_peaks

        wage_peaks, _ = find_peaks(wage_share.values)
        wage_troughs, _ = find_peaks(-wage_share.values)

        results['num_cycles'] = min(len(wage_peaks), len(wage_troughs))

        if len(wage_peaks) > 0 and len(wage_troughs) > 0:
            avg_cycle_length = np.mean(np.diff(wage_peaks))
            results['avg_cycle_length'] = avg_cycle_length

        return results

    @staticmethod
    def calculate_minsky_fragility_index(debt_gdp: float,
                                        debt_service_income: float,
                                        asset_price_growth: float) -> Tuple[str, float]:
        """
        Assess financial fragility using Minsky's classification.

        Minsky identified three financing regimes:
        - Hedge: Can pay interest and principal from income
        - Speculative: Can pay interest but must roll over principal
        - Ponzi: Cannot pay interest, relies on asset appreciation

        Args:
            debt_gdp: Debt-to-GDP ratio
            debt_service_income: Debt service to income ratio
            asset_price_growth: Asset price growth rate

        Returns:
            Tuple of (regime_type, fragility_score)

        Reference:
        - Minsky, H. (1986). Stabilizing an Unstable Economy.
        """
        fragility_score = 0

        # Score based on debt levels
        if debt_gdp > 100:
            fragility_score += 3
        elif debt_gdp > 70:
            fragility_score += 2
        elif debt_gdp > 50:
            fragility_score += 1

        # Score based on debt service
        if debt_service_income > 40:
            fragility_score += 3
        elif debt_service_income > 30:
            fragility_score += 2
        elif debt_service_income > 20:
            fragility_score += 1

        # Regime classification
        if debt_service_income < 20 and debt_gdp < 50:
            regime = "Hedge (Stable)"
        elif debt_service_income < 35 or asset_price_growth > 5:
            regime = "Speculative (Moderately Fragile)"
        else:
            regime = "Ponzi (Highly Fragile)"

        return regime, fragility_score

    @staticmethod
    def calculate_financialization_index(financial_sector_gdp: float,
                                        financial_profits_total: float,
                                        household_debt_income: float) -> float:
        """
        Calculate index of financialization.

        Combines multiple indicators of financial sector dominance.

        Args:
            financial_sector_gdp: Financial sector share of GDP (%)
            financial_profits_total: Financial profits share of total (%)
            household_debt_income: Household debt to income ratio (%)

        Returns:
            Financialization index (0-100)

        Reference:
        - Epstein, G. A. (Ed.). (2005). Financialization and the World Economy.
        """
        # Normalize each component to 0-100 scale
        fin_gdp_score = min(financial_sector_gdp * 3, 100)  # >33% = max
        fin_profit_score = min(financial_profits_total * 2, 100)  # >50% = max
        debt_score = min(household_debt_income / 2, 100)  # >200% = max

        # Weighted average
        index = (fin_gdp_score * 0.4 + fin_profit_score * 0.3 + debt_score * 0.3)

        return index

    @staticmethod
    def calculate_trend_and_cycle(series: pd.Series,
                                 method: str = 'hp_filter') -> Tuple[pd.Series, pd.Series]:
        """
        Decompose series into trend and cycle components.

        Args:
            series: Time series to decompose
            method: Decomposition method ('hp_filter', 'moving_average')

        Returns:
            Tuple of (trend, cycle)

        Reference:
        - Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. business cycles.
        """
        if method == 'hp_filter':
            try:
                from statsmodels.tsa.filters.hp_filter import hpfilter
                cycle, trend = hpfilter(series.dropna(), lamb=1600)
                return trend, cycle
            except ImportError:
                # Fallback to moving average
                method = 'moving_average'

        if method == 'moving_average':
            window = min(8, len(series) // 4)
            trend = series.rolling(window=window, center=True).mean()
            cycle = series - trend
            return trend, cycle

        raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def calculate_structural_break(series: pd.Series,
                                   min_size: int = 10) -> Optional[int]:
        """
        Detect structural break in time series.

        Uses simple Chow test approach.

        Args:
            series: Time series to test
            min_size: Minimum size of each segment

        Returns:
            Index of break point (or None if no significant break)

        Reference:
        - Chow, G. C. (1960). Tests of equality between sets of coefficients.
        """
        clean_series = series.dropna()
        n = len(clean_series)

        if n < 2 * min_size:
            return None

        best_break = None
        min_sse = float('inf')

        # Test each potential break point
        for i in range(min_size, n - min_size):
            # Fit linear trends to each segment
            x1 = np.arange(i)
            y1 = clean_series.iloc[:i].values

            x2 = np.arange(n - i)
            y2 = clean_series.iloc[i:].values

            # Calculate SSE for each segment
            try:
                p1 = np.polyfit(x1, y1, 1)
                p2 = np.polyfit(x2, y2, 1)

                sse1 = np.sum((y1 - np.polyval(p1, x1)) ** 2)
                sse2 = np.sum((y2 - np.polyval(p2, x2)) ** 2)

                total_sse = sse1 + sse2

                if total_sse < min_sse:
                    min_sse = total_sse
                    best_break = i

            except:
                continue

        return best_break
