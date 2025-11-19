"""
Emmanuel's Unequal Exchange Model (1972)

Arghiri Emmanuel's theory of unequal exchange argues that international trade
systematically transfers value from low-wage to high-wage countries, even when
commodities exchange at their prices of production.

Key Arguments:
1. International capital mobility tends to equalize profit rates globally
2. International labor immobility maintains wage differentials
3. Equal profit rates + unequal wages = unequal exchange of labor
4. Core countries appropriate surplus labor from periphery through trade

Mathematical Framework:
- Commodities sell at prices of production: p = (1+r)(wl + pA)
- Equal profit rates (r) globally due to capital mobility
- Unequal wages (w) due to labor immobility
- Value transfer = labor exported - labor imported (wage-adjusted)

References:
Emmanuel, A. (1972). Unequal Exchange: A Study of the Imperialism of Trade.
Monthly Review Press.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..core.theoretical_base import (
    UnequaExchangeModel,
    ProductionData,
    CountryCategory,
    ValueTransferCalculator
)


@dataclass
class EmmanuelParameters:
    """Parameters for Emmanuel's model"""
    global_profit_rate: float = 0.15  # Equalized profit rate
    capital_mobility: float = 1.0  # Degree of capital mobility (0-1)
    labor_mobility: float = 0.0  # Degree of labor mobility (0-1)


class EmmanuelModel(UnequaExchangeModel):
    """
    Implementation of Emmanuel's unequal exchange model.

    This model calculates value transfers arising from wage differentials
    in the context of internationally equalized profit rates.
    """

    def __init__(self, parameters: Optional[EmmanuelParameters] = None):
        """
        Initialize Emmanuel model.

        Args:
            parameters: Model parameters (uses defaults if None)
        """
        super().__init__(name="Emmanuel Unequal Exchange Model")
        self.params = parameters or EmmanuelParameters()
        self.value_calculator = ValueTransferCalculator()

    def calculate_prices_of_production(self, production: ProductionData,
                                       wage_rate: float) -> float:
        """
        Calculate price of production for a commodity.

        Price of production = (1 + r)(wl + pA)
        where:
        - r = profit rate
        - w = wage rate
        - l = labor coefficient
        - p = prices of inputs
        - A = input coefficients

        Simplified version assuming p*A captured in intermediate_inputs.

        Args:
            production: Production data
            wage_rate: Wage rate to use

        Returns:
            Price of production per unit
        """
        # Cost of labor per unit
        labor_cost_per_unit = (production.labor_hours / production.gross_output) * wage_rate

        # Cost of inputs per unit
        input_cost_per_unit = production.intermediate_inputs / production.gross_output

        # Total cost per unit
        cost_per_unit = labor_cost_per_unit + input_cost_per_unit

        # Price of production with profit markup
        price = cost_per_unit * (1 + self.params.global_profit_rate)

        return price

    def calculate_counterfactual_prices(self, country: str,
                                       reference_wage: float) -> float:
        """
        Calculate what prices would be if country used reference wage.

        Args:
            country: Country identifier
            reference_wage: Wage rate to use for counterfactual

        Returns:
            Counterfactual price
        """
        production = self.production_data[country]
        return self.calculate_prices_of_production(production, reference_wage)

    def calculate_value_transfers(self) -> pd.DataFrame:
        """
        Calculate value transfers between all country pairs.

        The value transfer from country A to country B equals:
        - Labor A exports to B (valued at B's wages) minus
        - Labor A exports to B (valued at A's wages)

        Returns:
            DataFrame with columns: exporter, importer, trade_value,
                                   labor_content, value_transfer
        """
        results = []

        # Calculate average core wage for reference
        core_countries = self.get_core_countries()
        if core_countries:
            avg_core_wage = np.mean([self.production_data[c].wage_rate
                                    for c in core_countries])
        else:
            avg_core_wage = 0

        for exporter in self.countries.keys():
            for importer in self.countries.keys():
                if exporter == importer:
                    continue

                # Get trade value
                if exporter in self.trade_flows.index and importer in self.trade_flows.columns:
                    trade_value = self.trade_flows.loc[exporter, importer]
                else:
                    continue

                if trade_value == 0:
                    continue

                # Get production data
                exp_prod = self.production_data[exporter]
                imp_prod = self.production_data[importer]

                # Calculate labor content of exports
                labor_coefficient = exp_prod.labor_hours / exp_prod.gross_output
                labor_exported = labor_coefficient * trade_value

                # Calculate value at exporter's wages vs importer's wages
                value_at_exp_wages = labor_exported * exp_prod.wage_rate
                value_at_imp_wages = labor_exported * imp_prod.wage_rate

                # Value transfer (positive = transfer from exporter to importer)
                value_transfer = value_at_exp_wages - value_at_imp_wages

                # If exporter is periphery and importer is core, transfer is negative
                # (periphery loses value)
                if (self.countries[exporter] == CountryCategory.PERIPHERY and
                    self.countries[importer] == CountryCategory.CORE):
                    # Invert sign: negative transfer = loss for periphery
                    value_transfer = -value_transfer

                results.append({
                    'exporter': exporter,
                    'importer': importer,
                    'exporter_category': self.countries[exporter].value,
                    'importer_category': self.countries[importer].value,
                    'trade_value': trade_value,
                    'labor_exported_hours': labor_exported,
                    'exporter_wage': exp_prod.wage_rate,
                    'importer_wage': imp_prod.wage_rate,
                    'wage_ratio': exp_prod.wage_rate / imp_prod.wage_rate if imp_prod.wage_rate > 0 else 0,
                    'value_transfer': value_transfer,
                    'transfer_pct_of_trade': (abs(value_transfer) / trade_value * 100) if trade_value > 0 else 0
                })

        df = pd.DataFrame(results)

        # Add net transfers by country
        if not df.empty:
            net_transfers = []
            for country in self.countries.keys():
                outflows = df[df['exporter'] == country]['value_transfer'].sum()
                inflows = df[df['importer'] == country]['value_transfer'].sum()
                net_transfer = inflows - outflows

                net_transfers.append({
                    'country': country,
                    'category': self.countries[country].value,
                    'total_exports': df[df['exporter'] == country]['trade_value'].sum(),
                    'total_imports': df[df['importer'] == country]['trade_value'].sum(),
                    'value_transfer_outflow': outflows,
                    'value_transfer_inflow': inflows,
                    'net_transfer': net_transfer
                })

            self.net_transfers_df = pd.DataFrame(net_transfers)

        return df

    def analyze_wage_equalization_scenario(self, target_wage: float) -> Dict[str, float]:
        """
        Simulate scenario where all wages equalize to target level.

        This counterfactual shows how much value transfer would be eliminated
        if wage differentials were removed (Emmanuel's key policy implication).

        Args:
            target_wage: Wage rate for equalization scenario

        Returns:
            Dictionary with impact analysis
        """
        # Calculate current transfers
        current_transfers = self.calculate_value_transfers()
        current_total = current_transfers['value_transfer'].abs().sum()

        # Simulate equalized wages
        original_wages = {}
        for country in self.countries.keys():
            original_wages[country] = self.production_data[country].wage_rate
            # Temporarily set wage to target
            prod = self.production_data[country]
            self.production_data[country] = ProductionData(
                gross_output=prod.gross_output,
                labor_hours=prod.labor_hours,
                wage_rate=target_wage,
                capital_stock=prod.capital_stock,
                intermediate_inputs=prod.intermediate_inputs
            )

        # Recalculate transfers
        equalized_transfers = self.calculate_value_transfers()
        equalized_total = equalized_transfers['value_transfer'].abs().sum()

        # Restore original wages
        for country in self.countries.keys():
            prod = self.production_data[country]
            self.production_data[country] = ProductionData(
                gross_output=prod.gross_output,
                labor_hours=prod.labor_hours,
                wage_rate=original_wages[country],
                capital_stock=prod.capital_stock,
                intermediate_inputs=prod.intermediate_inputs
            )

        return {
            'current_total_transfers': current_total,
            'equalized_total_transfers': equalized_total,
            'transfers_eliminated': current_total - equalized_total,
            'pct_eliminated': ((current_total - equalized_total) / current_total * 100)
                            if current_total > 0 else 0,
            'target_wage': target_wage
        }

    def calculate_hidden_transfers(self) -> pd.DataFrame:
        """
        Calculate "hidden" transfers through price differentials.

        Emmanuel argued that standard trade statistics hide value transfers
        because they record commodities at market prices, not labor values.

        Returns:
            DataFrame showing hidden transfers
        """
        transfers = self.calculate_value_transfers()

        # Group by country category pairs
        category_transfers = transfers.groupby(
            ['exporter_category', 'importer_category']
        ).agg({
            'trade_value': 'sum',
            'labor_exported_hours': 'sum',
            'value_transfer': 'sum'
        }).reset_index()

        category_transfers['hidden_transfer_pct'] = (
            category_transfers['value_transfer'] /
            category_transfers['trade_value'] * 100
        )

        return category_transfers

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Generate summary statistics for Emmanuel model results.

        Returns:
            Dictionary with key model outputs
        """
        transfers = self.calculate_value_transfers()
        net_transfers = self.net_transfers_df if hasattr(self, 'net_transfers_df') else None

        stats = {
            'total_trade_value': transfers['trade_value'].sum(),
            'total_value_transfers': transfers['value_transfer'].abs().sum(),
            'avg_transfer_pct': transfers['transfer_pct_of_trade'].mean(),
        }

        if net_transfers is not None:
            core_gain = net_transfers[
                net_transfers['category'] == CountryCategory.CORE.value
            ]['net_transfer'].sum()

            periphery_loss = net_transfers[
                net_transfers['category'] == CountryCategory.PERIPHERY.value
            ]['net_transfer'].sum()

            stats.update({
                'core_net_gain': core_gain,
                'periphery_net_loss': periphery_loss,
                'core_gain_as_pct_gdp': 0,  # Would need GDP data
            })

        return stats
