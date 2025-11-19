"""
Value Transfer Analysis

Comprehensive tools for analyzing value transfers in international trade
through multiple mechanisms:
1. Unequal exchange (Emmanuel, Amin)
2. Terms of trade effects
3. Transfer pricing manipulation
4. Intellectual property rents
5. Financial flows and debt service
6. Repatriated profits

References:
- Hickel et al. (2021): Plunder in the Post-Colonial Era
- Amin, S. (1974): Accumulation on a World Scale
- Patnaik & Patnaik (2016): A Theory of Imperialism
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..core.theoretical_base import ProductionData, CountryCategory
from ..io_framework.multi_country import MultiCountryIOTable


@dataclass
class TransferComponents:
    """Components of total value transfer"""
    unequal_exchange: float = 0.0
    terms_of_trade: float = 0.0
    transfer_pricing: float = 0.0
    ip_rents: float = 0.0
    profit_repatriation: float = 0.0
    debt_service: float = 0.0
    other: float = 0.0

    @property
    def total(self) -> float:
        """Total value transfer"""
        return (self.unequal_exchange + self.terms_of_trade +
                self.transfer_pricing + self.ip_rents +
                self.profit_repatriation + self.debt_service + self.other)


class ValueTransferAnalyzer:
    """
    Comprehensive value transfer analysis combining multiple mechanisms.
    """

    def __init__(self, io_table: Optional[MultiCountryIOTable] = None):
        """
        Initialize analyzer.

        Args:
            io_table: Multi-country IO table for GVC analysis
        """
        self.io_table = io_table
        self.transfer_results: Dict[str, TransferComponents] = {}

    def calculate_total_transfers(self,
                                  country: str,
                                  year: int,
                                  production_data: Optional[ProductionData] = None,
                                  trade_data: Optional[pd.DataFrame] = None,
                                  financial_data: Optional[Dict] = None) -> TransferComponents:
        """
        Calculate total value transfers for a country across all mechanisms.

        Args:
            country: Country identifier
            year: Year of analysis
            production_data: Production data for the country
            trade_data: Bilateral trade data
            financial_data: Financial flows data

        Returns:
            TransferComponents with breakdown
        """
        transfers = TransferComponents()

        # 1. Unequal exchange (if IO table available)
        if self.io_table is not None:
            labor_decomp = self.io_table.calculate_embodied_labor()
            # Extract labor exports vs imports for this country
            # (Simplified - would need wage data)
            transfers.unequal_exchange = self._estimate_ue_from_io(country, labor_decomp)

        # 2. Terms of trade effects (if trade data available)
        if trade_data is not None:
            transfers.terms_of_trade = self._calculate_tot_effect(country, trade_data, year)

        # 3. Transfer pricing (if available)
        if financial_data and 'transfer_pricing' in financial_data:
            transfers.transfer_pricing = financial_data['transfer_pricing']

        # 4. IP rents (if available)
        if financial_data and 'ip_royalties' in financial_data:
            transfers.ip_rents = financial_data['ip_royalties']

        # 5. Profit repatriation (if available)
        if financial_data and 'profit_repatriation' in financial_data:
            transfers.profit_repatriation = financial_data['profit_repatriation']

        # 6. Debt service (if available)
        if financial_data and 'debt_service' in financial_data:
            transfers.debt_service = financial_data['debt_service']

        self.transfer_results[country] = transfers

        return transfers

    def _estimate_ue_from_io(self, country: str, labor_decomp: pd.DataFrame) -> float:
        """
        Estimate unequal exchange from IO table.

        Args:
            country: Country identifier
            labor_decomp: Labor embodied in final demand

        Returns:
            Estimated unequal exchange transfer
        """
        # Simplified: would need wage differentials
        # This is a placeholder calculation
        return 0.0

    def _calculate_tot_effect(self, country: str, trade_data: pd.DataFrame, year: int) -> float:
        """
        Calculate terms of trade effect.

        Args:
            country: Country identifier
            trade_data: Trade data with price indices
            year: Year

        Returns:
            ToT-induced transfer
        """
        # Placeholder - would need actual price indices
        return 0.0

    def calculate_drain_rate(self, country: str, gdp: float) -> Dict[str, float]:
        """
        Calculate 'drain rate' - transfers as percentage of GDP.

        Concept from Utsa and Prabhat Patnaik's work on colonial drain.

        Args:
            country: Country identifier
            gdp: GDP of country

        Returns:
            Dictionary with drain metrics
        """
        if country not in self.transfer_results:
            return {}

        transfers = self.transfer_results[country]

        return {
            'total_drain': transfers.total,
            'drain_rate_pct': (transfers.total / gdp * 100) if gdp > 0 else 0,
            'ue_drain_pct': (transfers.unequal_exchange / gdp * 100) if gdp > 0 else 0,
            'tot_drain_pct': (transfers.terms_of_trade / gdp * 100) if gdp > 0 else 0,
            'transfer_pricing_pct': (transfers.transfer_pricing / gdp * 100) if gdp > 0 else 0,
            'ip_rents_pct': (transfers.ip_rents / gdp * 100) if gdp > 0 else 0,
            'financial_drain_pct': ((transfers.profit_repatriation + transfers.debt_service) / gdp * 100) if gdp > 0 else 0
        }

    def calculate_cumulative_drain(self, country: str,
                                   annual_transfers: pd.Series,
                                   discount_rate: float = 0.05) -> Dict[str, float]:
        """
        Calculate cumulative drain over time.

        Shows total value extracted over historical period.

        Args:
            country: Country identifier
            annual_transfers: Series of annual transfers
            discount_rate: Discount rate for present value

        Returns:
            Cumulative drain metrics
        """
        years = len(annual_transfers)

        # Nominal cumulative
        cumulative_nominal = annual_transfers.sum()

        # Present value
        discount_factors = [(1 + discount_rate) ** i for i in range(years)]
        cumulative_pv = sum(annual_transfers.values * discount_factors)

        # Compounded value (if reinvested)
        compound_factors = [(1 + 0.07) ** (years - i - 1) for i in range(years)]  # 7% return assumption
        cumulative_compounded = sum(annual_transfers.values * compound_factors)

        return {
            'cumulative_nominal': cumulative_nominal,
            'cumulative_present_value': cumulative_pv,
            'cumulative_compounded': cumulative_compounded,
            'years': years,
            'avg_annual_drain': cumulative_nominal / years if years > 0 else 0
        }

    def analyze_transfer_asymmetry(self, transfers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze asymmetry in bilateral transfers.

        Shows which country pairs have most unequal relationships.

        Args:
            transfers_df: DataFrame with bilateral transfers

        Returns:
            Asymmetry analysis
        """
        results = []

        countries = transfers_df['country_a'].unique()

        for i, country_a in enumerate(countries):
            for country_b in countries[i+1:]:
                # Transfer from A to B
                a_to_b = transfers_df[
                    (transfers_df['country_a'] == country_a) &
                    (transfers_df['country_b'] == country_b)
                ]['transfer'].sum()

                # Transfer from B to A
                b_to_a = transfers_df[
                    (transfers_df['country_a'] == country_b) &
                    (transfers_df['country_b'] == country_a)
                ]['transfer'].sum()

                # Net transfer
                net_transfer = a_to_b - b_to_a

                # Asymmetry ratio
                total_transfer = abs(a_to_b) + abs(b_to_a)
                asymmetry = abs(net_transfer) / total_transfer if total_transfer > 0 else 0

                results.append({
                    'country_pair': f"{country_a}-{country_b}",
                    'country_a': country_a,
                    'country_b': country_b,
                    'transfer_a_to_b': a_to_b,
                    'transfer_b_to_a': b_to_a,
                    'net_transfer': net_transfer,
                    'dominant_country': country_a if net_transfer > 0 else country_b,
                    'asymmetry_ratio': asymmetry
                })

        return pd.DataFrame(results).sort_values('asymmetry_ratio', ascending=False)

    def decompose_north_south_transfers(self,
                                       transfers_df: pd.DataFrame,
                                       country_categories: Dict[str, CountryCategory]) -> Dict[str, float]:
        """
        Decompose transfers between North (core) and South (periphery).

        Args:
            transfers_df: Bilateral transfers
            country_categories: Mapping of countries to categories

        Returns:
            North-South transfer decomposition
        """
        # Identify core and periphery
        core = [c for c, cat in country_categories.items() if cat == CountryCategory.CORE]
        periphery = [c for c, cat in country_categories.items() if cat == CountryCategory.PERIPHERY]

        # Core to periphery transfers
        core_to_periphery = transfers_df[
            transfers_df['country_a'].isin(core) &
            transfers_df['country_b'].isin(periphery)
        ]['transfer'].sum()

        # Periphery to core transfers
        periphery_to_core = transfers_df[
            transfers_df['country_a'].isin(periphery) &
            transfers_df['country_b'].isin(core)
        ]['transfer'].sum()

        # Net transfer (positive = North gains)
        net_north_south = periphery_to_core - core_to_periphery

        return {
            'core_to_periphery': core_to_periphery,
            'periphery_to_core': periphery_to_core,
            'net_north_south_transfer': net_north_south,
            'periphery_loss_per_capita': 0,  # Would need population data
            'core_gain_per_capita': 0
        }
