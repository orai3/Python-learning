"""
Theoretical Base Classes for Unequal Exchange Analysis

This module provides the foundational theoretical framework for analyzing
unequal exchange, value transfers, and labor exploitation in international trade.

Key Concepts:
- Labor Value Theory: Commodities embody socially necessary labor time
- Unequal Exchange: Transfer of value from periphery to core through trade
- Super-exploitation: Labor paid below value of labor-power
- Value Appropriation: Mechanisms of surplus extraction in global capitalism

References:
- Emmanuel, A. (1972). Unequal Exchange: A Study of the Imperialism of Trade
- Amin, S. (1974). Accumulation on a World Scale
- Cope, Z. (2019). The Wealth of (Some) Nations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class CountryCategory(Enum):
    """Classification of countries in world-system"""
    CORE = "core"
    SEMI_PERIPHERY = "semi_periphery"
    PERIPHERY = "periphery"


@dataclass
class ProductionData:
    """Encapsulates production data for a country/sector"""
    gross_output: float  # Total output value
    labor_hours: float  # Total labor hours employed
    wage_rate: float  # Hourly wage rate
    capital_stock: float  # Value of capital employed
    intermediate_inputs: float  # Cost of intermediate inputs

    @property
    def value_added(self) -> float:
        """Calculate value added (output - intermediate inputs)"""
        return self.gross_output - self.intermediate_inputs

    @property
    def labor_cost(self) -> float:
        """Total labor compensation"""
        return self.labor_hours * self.wage_rate

    @property
    def surplus_value(self) -> float:
        """Surplus value = Value added - Labor cost"""
        return self.value_added - self.labor_cost

    @property
    def rate_of_exploitation(self) -> float:
        """Rate of exploitation = Surplus value / Labor cost"""
        if self.labor_cost == 0:
            return 0.0
        return self.surplus_value / self.labor_cost

    @property
    def labor_productivity(self) -> float:
        """Output per labor hour"""
        if self.labor_hours == 0:
            return 0.0
        return self.gross_output / self.labor_hours

    @property
    def organic_composition(self) -> float:
        """Organic composition of capital = Capital / Labor cost"""
        if self.labor_cost == 0:
            return 0.0
        return self.capital_stock / self.labor_cost


class LaborValueCalculator:
    """
    Calculates labor values of commodities using various methods.

    Implements:
    - Direct labor content
    - Vertically integrated labor coefficients
    - Labor embodied in trade
    """

    def __init__(self, io_table: Optional[np.ndarray] = None,
                 labor_coefficients: Optional[np.ndarray] = None):
        """
        Initialize calculator with input-output data.

        Args:
            io_table: Technical coefficients matrix (A matrix)
            labor_coefficients: Direct labor coefficients per unit output
        """
        self.io_table = io_table
        self.labor_coefficients = labor_coefficients

    def direct_labor_content(self, production: ProductionData) -> float:
        """Calculate direct labor hours per unit of output"""
        if production.gross_output == 0:
            return 0.0
        return production.labor_hours / production.gross_output

    def vertically_integrated_labor(self) -> Optional[np.ndarray]:
        """
        Calculate total labor embodied (direct + indirect) using Leontief inverse.

        Formula: l = l_d * (I - A)^(-1)
        where:
        - l = vertically integrated labor coefficients
        - l_d = direct labor coefficients
        - A = technical coefficients matrix
        - (I - A)^(-1) = Leontief inverse

        Returns:
            Array of labor values per unit gross output
        """
        if self.io_table is None or self.labor_coefficients is None:
            return None

        n = self.io_table.shape[0]
        I = np.eye(n)

        # Calculate Leontief inverse
        try:
            leontief_inverse = np.linalg.inv(I - self.io_table)
            # Multiply direct labor by Leontief inverse
            labor_values = self.labor_coefficients @ leontief_inverse
            return labor_values
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in Leontief inverse calculation")
            return None

    def labor_embodied_in_trade(self, exports: np.ndarray,
                                 imports: np.ndarray) -> Tuple[float, float]:
        """
        Calculate labor embodied in exports and imports.

        Args:
            exports: Vector of export values by sector
            imports: Vector of import values by sector

        Returns:
            Tuple of (labor in exports, labor in imports)
        """
        labor_values = self.vertically_integrated_labor()
        if labor_values is None:
            return (0.0, 0.0)

        labor_exports = np.sum(labor_values * exports)
        labor_imports = np.sum(labor_values * imports)

        return (labor_exports, labor_imports)


class ValueTransferCalculator:
    """
    Calculates value transfers between countries through trade.

    Implements multiple methods:
    1. Emmanuel's unequal exchange (wage differentials)
    2. Amin's unequal exchange (productivity-adjusted)
    3. Direct labor content comparison
    4. Terms of trade adjusted transfers
    """

    def __init__(self, base_year: int = 2000):
        """
        Initialize value transfer calculator.

        Args:
            base_year: Reference year for deflating values
        """
        self.base_year = base_year

    def emmanuel_transfer(self,
                          south_production: ProductionData,
                          north_production: ProductionData,
                          trade_value: float) -> float:
        """
        Calculate Emmanuel's unequal exchange based on wage differentials.

        Emmanuel argued that equal exchange of labor values is prevented by
        wage differentials between North and South. Capital mobility equalizes
        profit rates but labor immobility maintains wage gaps.

        The value transfer equals the difference between:
        - Labor embodied in Southern exports at Northern wages
        - Labor embodied in Southern exports at Southern wages

        Args:
            south_production: Southern country production data
            north_production: Northern country production data
            trade_value: Value of South-North trade

        Returns:
            Value transferred from South to North
        """
        # Calculate labor content per unit value
        south_labor_content = south_production.labor_hours / south_production.gross_output

        # Calculate what the same labor would earn at Northern wages
        counterfactual_value = south_labor_content * trade_value * north_production.wage_rate
        actual_value = south_labor_content * trade_value * south_production.wage_rate

        # Transfer = difference between counterfactual and actual
        transfer = (counterfactual_value - actual_value)

        return transfer

    def amin_transfer(self,
                      south_production: ProductionData,
                      north_production: ProductionData,
                      trade_value: float) -> float:
        """
        Calculate Amin's unequal exchange incorporating productivity differences.

        Amin extended Emmanuel by incorporating productivity differences.
        Super-exploitation occurs when wages in the South are below the
        value of labor-power, even accounting for productivity differences.

        Args:
            south_production: Southern country production data
            north_production: Northern country production data
            trade_value: Value of South-North trade

        Returns:
            Value transferred through super-exploitation
        """
        # Productivity-adjusted wage comparison
        south_productivity = south_production.labor_productivity
        north_productivity = north_production.labor_productivity

        if north_productivity == 0:
            return 0.0

        # Expected Southern wage if productivity differences were the only factor
        productivity_ratio = south_productivity / north_productivity
        expected_south_wage = north_production.wage_rate * productivity_ratio

        # Actual wage gap beyond productivity differences
        super_exploitation_gap = expected_south_wage - south_production.wage_rate

        # Calculate labor content of exports
        labor_in_exports = (south_production.labor_hours / south_production.gross_output) * trade_value

        # Value transfer from super-exploitation
        transfer = labor_in_exports * super_exploitation_gap

        return transfer

    def terms_of_trade_transfer(self,
                                 export_prices: pd.Series,
                                 import_prices: pd.Series,
                                 base_tot: float = 100.0) -> pd.Series:
        """
        Calculate income transfers from terms of trade deterioration.

        Prebisch-Singer hypothesis: Terms of trade for primary commodities
        deteriorate over time, transferring income from periphery to core.

        Args:
            export_prices: Price index for exports
            import_prices: Price index for imports
            base_tot: Base year terms of trade (default 100)

        Returns:
            Series of value transfers over time
        """
        # Calculate terms of trade index
        tot = (export_prices / import_prices) * 100

        # Calculate loss from base year
        tot_loss = base_tot - tot

        # Positive values indicate transfer from South to North
        return tot_loss

    def total_transfer(self,
                       south_production: ProductionData,
                       north_production: ProductionData,
                       trade_value: float,
                       tot_effect: float = 0.0) -> Dict[str, float]:
        """
        Calculate total value transfer through multiple mechanisms.

        Args:
            south_production: Southern country production data
            north_production: Northern country production data
            trade_value: Value of South-North trade
            tot_effect: Terms of trade effect (percentage)

        Returns:
            Dictionary with breakdown of transfers
        """
        emmanuel = self.emmanuel_transfer(south_production, north_production, trade_value)
        amin = self.amin_transfer(south_production, north_production, trade_value)
        tot_transfer = trade_value * (tot_effect / 100)

        return {
            'emmanuel_transfer': emmanuel,
            'amin_transfer': amin,
            'terms_of_trade_transfer': tot_transfer,
            'total_transfer': emmanuel + amin + tot_transfer,
            'transfer_as_pct_of_trade': ((emmanuel + amin + tot_transfer) / trade_value * 100) if trade_value > 0 else 0
        }


class UnequaExchangeModel:
    """
    Base class for unequal exchange models.

    Provides common infrastructure for Emmanuel, Amin, and other models.
    """

    def __init__(self, name: str = "Base Model"):
        """
        Initialize base model.

        Args:
            name: Name of the model
        """
        self.name = name
        self.countries: Dict[str, CountryCategory] = {}
        self.production_data: Dict[str, ProductionData] = {}
        self.trade_flows: pd.DataFrame = pd.DataFrame()

    def add_country(self, country: str, category: CountryCategory,
                    production: ProductionData):
        """
        Add a country to the model.

        Args:
            country: Country identifier
            category: Core, semi-periphery, or periphery
            production: Production data for the country
        """
        self.countries[country] = category
        self.production_data[country] = production

    def set_trade_flows(self, trade_matrix: pd.DataFrame):
        """
        Set bilateral trade flows between countries.

        Args:
            trade_matrix: DataFrame with exporter rows, importer columns
        """
        self.trade_flows = trade_matrix

    def calculate_value_transfers(self) -> pd.DataFrame:
        """
        Calculate value transfers between all country pairs.

        Returns:
            DataFrame with value transfers by country pair
        """
        raise NotImplementedError("Subclasses must implement this method")

    def get_core_countries(self) -> List[str]:
        """Get list of core countries"""
        return [c for c, cat in self.countries.items() if cat == CountryCategory.CORE]

    def get_periphery_countries(self) -> List[str]:
        """Get list of periphery countries"""
        return [c for c, cat in self.countries.items() if cat == CountryCategory.PERIPHERY]

    def get_semiperiphery_countries(self) -> List[str]:
        """Get list of semi-periphery countries"""
        return [c for c, cat in self.countries.items() if cat == CountryCategory.SEMI_PERIPHERY]

    def aggregate_transfers_by_category(self) -> Dict[str, float]:
        """
        Aggregate net value transfers by country category.

        Returns:
            Dictionary with net transfers for each category
        """
        transfers = self.calculate_value_transfers()

        results = {
            'core_net_gain': 0.0,
            'periphery_net_loss': 0.0,
            'semi_periphery_net': 0.0
        }

        for country, category in self.countries.items():
            country_net = transfers[transfers['country'] == country]['net_transfer'].sum()

            if category == CountryCategory.CORE:
                results['core_net_gain'] += country_net
            elif category == CountryCategory.PERIPHERY:
                results['periphery_net_loss'] += country_net
            else:
                results['semi_periphery_net'] += country_net

        return results
