"""
Prebisch-Singer Thesis: Terms of Trade Deterioration

The Prebisch-Singer hypothesis argues that prices of primary commodities
decline relative to manufactured goods over time, causing a secular
deterioration in the terms of trade for developing countries.

Key Arguments:
1. Income elasticity of demand for primary goods < manufactured goods
2. Productivity gains in manufacturing → lower prices
3. Productivity gains in primary production → lower prices (no monopoly power)
4. Core captures productivity gains; periphery passes them to consumers
5. Result: Continuous income transfer from periphery to core

Mechanisms:
- Engel's Law: As incomes rise, demand for food rises less than proportionally
- Monopoly power in manufacturing vs competitive primary markets
- Labor unions in core capture productivity gains as wages
- Lack of labor organization in periphery

References:
Prebisch, R. (1950). The Economic Development of Latin America.
Singer, H. (1950). The Distribution of Gains from Trade.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats

from ..core.theoretical_base import UnequaExchangeModel, CountryCategory


@dataclass
class PrebischSingerParameters:
    """Parameters for Prebisch-Singer model"""
    # Elasticity parameters
    income_elasticity_primary: float = 0.6  # < 1 (Engel's Law)
    income_elasticity_manufactures: float = 1.3  # > 1

    # Productivity growth rates (annual)
    productivity_growth_primary: float = 0.02  # 2% per year
    productivity_growth_manufactures: float = 0.03  # 3% per year

    # Market structure
    market_power_manufactures: float = 0.7  # Degree of monopoly (0-1)
    market_power_primary: float = 0.2  # Competitive markets

    # Pass-through rates
    productivity_to_wages_core: float = 0.6  # Core workers capture gains
    productivity_to_wages_periphery: float = 0.2  # Periphery workers don't

    # Base year
    base_year: int = 1960


class PrebischSingerModel(UnequaExchangeModel):
    """
    Implementation of Prebisch-Singer terms of trade analysis.

    Models long-run deterioration in terms of trade for primary exporters.
    """

    def __init__(self, parameters: Optional[PrebischSingerParameters] = None):
        """
        Initialize Prebisch-Singer model.

        Args:
            parameters: Model parameters (uses defaults if None)
        """
        super().__init__(name="Prebisch-Singer Terms of Trade Model")
        self.params = parameters or PrebischSingerParameters()

    def calculate_terms_of_trade(self, export_prices: pd.Series,
                                 import_prices: pd.Series,
                                 base_year: Optional[int] = None) -> pd.Series:
        """
        Calculate terms of trade index.

        ToT = (Export Price Index / Import Price Index) * 100

        Base year = 100. Declining ToT means periphery receives less imports
        for same volume of exports.

        Args:
            export_prices: Price index for exports
            import_prices: Price index for imports
            base_year: Year to use as base (default: params.base_year)

        Returns:
            Terms of trade index
        """
        if base_year is None:
            base_year = self.params.base_year

        # Normalize to base year
        export_base = export_prices[export_prices.index == base_year].values[0]
        import_base = import_prices[import_prices.index == base_year].values[0]

        export_normalized = (export_prices / export_base) * 100
        import_normalized = (import_prices / import_base) * 100

        tot = (export_normalized / import_normalized) * 100

        return tot

    def simulate_long_run_tot(self, years: int = 60,
                             start_year: int = 1960) -> pd.DataFrame:
        """
        Simulate long-run terms of trade dynamics.

        Models interaction of:
        - Differential productivity growth
        - Differential income elasticities
        - Differential market power
        - Differential wage growth

        Args:
            years: Number of years to simulate
            start_year: Starting year

        Returns:
            DataFrame with price indices and ToT over time
        """
        time_periods = range(start_year, start_year + years)

        # Initialize prices at 100
        primary_price = 100.0
        manufactures_price = 100.0

        # Initialize incomes
        core_income = 100.0
        periphery_income = 100.0

        results = []

        for t, year in enumerate(time_periods):
            # Productivity growth
            primary_productivity = (1 + self.params.productivity_growth_primary) ** t
            manufactures_productivity = (1 + self.params.productivity_growth_manufactures) ** t

            # Income growth (exogenous, simplified)
            core_income_growth = 0.03  # 3% per year
            periphery_income_growth = 0.025  # 2.5% per year

            core_income = 100 * (1 + core_income_growth) ** t
            periphery_income = 100 * (1 + periphery_income_growth) ** t

            # Demand growth based on income elasticities
            primary_demand_growth = (periphery_income_growth *
                                    self.params.income_elasticity_primary)
            manufactures_demand_growth = (core_income_growth *
                                         self.params.income_elasticity_manufactures)

            # Price dynamics: P = (1 + demand_growth) / (1 + productivity_growth)
            # Modified by market power

            # Primary prices: competitive markets, productivity gains → price drops
            primary_price_change = (
                (1 + primary_demand_growth) / (1 + self.params.productivity_growth_primary) - 1
            )
            # In competitive markets, most productivity gain passed to consumers
            primary_price_change *= (1 - self.params.market_power_primary)

            # Manufactures prices: monopolistic markets, retain productivity gains
            manufactures_price_change = (
                (1 + manufactures_demand_growth) / (1 + self.params.productivity_growth_manufactures) - 1
            )
            # With market power, prices don't fall as much
            manufactures_price_change *= self.params.market_power_manufactures

            # Update prices
            primary_price *= (1 + primary_price_change)
            manufactures_price *= (1 + manufactures_price_change)

            # Calculate ToT (periphery perspective: primary/manufactures)
            tot = (primary_price / manufactures_price) * 100

            # Calculate income transfer
            # Base trade at start_year
            base_trade_volume = 100
            trade_volume = base_trade_volume * (1 + 0.04) ** t  # 4% annual growth

            # Income transfer from ToT deterioration
            # If ToT = 90, need to export 10% more to buy same imports
            tot_loss_pct = (100 - tot)
            income_transfer = trade_volume * (tot_loss_pct / 100)

            results.append({
                'year': year,
                'primary_price_index': primary_price,
                'manufactures_price_index': manufactures_price,
                'terms_of_trade': tot,
                'tot_vs_base': tot - 100,  # Cumulative change from base
                'primary_productivity': primary_productivity * 100,
                'manufactures_productivity': manufactures_productivity * 100,
                'core_income': core_income,
                'periphery_income': periphery_income,
                'trade_volume': trade_volume,
                'income_transfer_from_tot': income_transfer,
                'cumulative_transfer': 0  # Will calculate below
            })

        df = pd.DataFrame(results)

        # Calculate cumulative transfer
        df['cumulative_transfer'] = df['income_transfer_from_tot'].cumsum()

        return df

    def estimate_historical_trend(self, price_data: pd.DataFrame,
                                  primary_col: str = 'primary_prices',
                                  manufactures_col: str = 'manufactures_prices') -> Dict[str, float]:
        """
        Estimate historical trend in terms of trade.

        Uses linear regression to test Prebisch-Singer hypothesis.

        Args:
            price_data: DataFrame with price indices
            primary_col: Column name for primary commodity prices
            manufactures_col: Column name for manufactured goods prices

        Returns:
            Dictionary with trend statistics
        """
        # Calculate ToT
        tot = (price_data[primary_col] / price_data[manufactures_col]) * 100

        # Time trend
        years = np.arange(len(tot))

        # Run regression: ToT = a + b*time + error
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, tot)

        # Annual deterioration rate
        annual_deterioration = slope

        # Total deterioration over period
        total_deterioration = slope * len(years)

        return {
            'annual_deterioration_rate': annual_deterioration,
            'total_deterioration': total_deterioration,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'intercept': intercept,
            'periods': len(tot),
            'tot_start': tot.iloc[0],
            'tot_end': tot.iloc[-1]
        }

    def calculate_income_effect(self, tot_series: pd.Series,
                                export_volume: pd.Series) -> pd.DataFrame:
        """
        Calculate income effect of ToT changes.

        The income effect shows how much additional exports are needed
        to maintain same import capacity.

        Args:
            tot_series: Terms of trade index
            export_volume: Export volume index

        Returns:
            DataFrame with income effects
        """
        results = []

        base_tot = tot_series.iloc[0]
        base_exports = export_volume.iloc[0]

        for i, (year, tot) in enumerate(tot_series.items()):
            exports = export_volume.iloc[i]

            # Import capacity with current ToT
            import_capacity = exports * (tot / 100)

            # Import capacity with base ToT (counterfactual)
            import_capacity_base = exports * (base_tot / 100)

            # Loss in import capacity
            import_capacity_loss = import_capacity_base - import_capacity

            # Additional exports needed to maintain base import capacity
            additional_exports_needed = (import_capacity_base / (tot / 100)) - exports

            results.append({
                'year': year,
                'tot': tot,
                'exports': exports,
                'import_capacity': import_capacity,
                'import_capacity_base': import_capacity_base,
                'import_capacity_loss': import_capacity_loss,
                'additional_exports_needed': additional_exports_needed,
                'income_transfer_pct': (import_capacity_loss / import_capacity_base * 100)
                                      if import_capacity_base > 0 else 0
            })

        return pd.DataFrame(results)

    def decompose_tot_change(self, tot_series: pd.Series) -> Dict[str, float]:
        """
        Decompose changes in ToT into components.

        Args:
            tot_series: Terms of trade series

        Returns:
            Dictionary with decomposition
        """
        # Calculate log changes
        log_tot = np.log(tot_series)
        log_changes = log_tot.diff()

        return {
            'mean_annual_change': log_changes.mean() * 100,  # Convert to percentage
            'volatility': log_changes.std() * 100,
            'total_change': (tot_series.iloc[-1] / tot_series.iloc[0] - 1) * 100,
            'trend_component': log_changes.mean() * len(tot_series) * 100,
            'number_periods': len(tot_series)
        }

    def calculate_value_transfers(self) -> pd.DataFrame:
        """
        Calculate value transfers from ToT deterioration.

        Returns:
            DataFrame with transfers by country
        """
        # This is a placeholder - actual implementation would need
        # historical price data for each country's export basket

        results = []

        for country in self.countries.keys():
            category = self.countries[country]

            # Simulate country-specific ToT
            if category == CountryCategory.PERIPHERY:
                # Primary exporters face deteriorating ToT
                tot_trend = self.simulate_long_run_tot()
                tot_loss = tot_trend['income_transfer_from_tot'].iloc[-1]

            elif category == CountryCategory.CORE:
                # Manufactures exporters gain from improving ToT
                tot_trend = self.simulate_long_run_tot()
                tot_loss = -tot_trend['income_transfer_from_tot'].iloc[-1]  # Gain

            else:  # Semi-periphery
                # Mixed export basket
                tot_loss = 0

            results.append({
                'country': country,
                'category': category.value,
                'tot_transfer': tot_loss
            })

        return pd.DataFrame(results)
