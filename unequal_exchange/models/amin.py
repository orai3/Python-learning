"""
Samir Amin's Unequal Exchange Model (1974)

Samir Amin extended Emmanuel's analysis to incorporate:
1. Productivity differentials between core and periphery
2. Super-exploitation of labor in the periphery
3. Blocking of autonomous accumulation in the periphery
4. Value transfers through multiple mechanisms beyond wage gaps

Key Concepts:
- Super-exploitation: Wages below value of labor-power even accounting for productivity
- Blocked development: Periphery locked into low-productivity activities
- Autocentric vs extroverted accumulation
- Combined effect of productivity gaps AND super-exploitation

Amin's Critique of Emmanuel:
Emmanuel focused only on wage differentials with equal productivity.
Amin argues productivity differences matter, AND wages in periphery are
below what productivity would justify.

References:
Amin, S. (1974). Accumulation on a World Scale.
Amin, S. (1976). Unequal Development.
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
class AminParameters:
    """Parameters for Amin's model"""
    # Technological parameters
    core_technology_level: float = 1.0  # Normalized to 1
    periphery_technology_level: float = 0.4  # Lower productivity

    # Labor market parameters
    value_of_labor_power_core: float = 20.0  # Subsistence + historical component
    value_of_labor_power_periphery: float = 8.0  # Lower due to different consumption basket

    # Structural parameters
    export_orientation_periphery: float = 0.4  # Share of output exported
    import_dependence_periphery: float = 0.5  # Share of consumption imported


class AminModel(UnequaExchangeModel):
    """
    Implementation of Samir Amin's extended unequal exchange model.

    Incorporates both wage differentials and productivity differentials,
    plus super-exploitation of peripheral labor.
    """

    def __init__(self, parameters: Optional[AminParameters] = None):
        """
        Initialize Amin model.

        Args:
            parameters: Model parameters (uses defaults if None)
        """
        super().__init__(name="Amin Unequal Exchange Model")
        self.params = parameters or AminParameters()
        self.value_calculator = ValueTransferCalculator()

    def calculate_super_exploitation_rate(self, country: str) -> float:
        """
        Calculate rate of super-exploitation.

        Super-exploitation rate = (Value of labor power - Actual wage) / Actual wage

        Positive values indicate wages below value of labor-power.

        Args:
            country: Country identifier

        Returns:
            Super-exploitation rate
        """
        production = self.production_data[country]
        category = self.countries[country]

        # Determine value of labor-power based on country category
        if category == CountryCategory.CORE:
            value_labor_power = self.params.value_of_labor_power_core
        else:
            value_labor_power = self.params.value_of_labor_power_periphery

        # Calculate super-exploitation
        if production.wage_rate == 0:
            return 0.0

        super_exploitation = (value_labor_power - production.wage_rate) / production.wage_rate

        return super_exploitation

    def calculate_productivity_adjusted_wages(self, country: str) -> float:
        """
        Calculate what wages should be if only productivity differences mattered.

        If core wage is w_c and periphery productivity is p_p/p_c of core,
        then productivity-justified periphery wage is: w_p_justified = w_c * (p_p/p_c)

        Args:
            country: Country identifier

        Returns:
            Productivity-adjusted wage
        """
        category = self.countries[country]

        if category == CountryCategory.CORE:
            return self.production_data[country].wage_rate

        # Get average core wage
        core_countries = self.get_core_countries()
        if not core_countries:
            return self.production_data[country].wage_rate

        avg_core_wage = np.mean([self.production_data[c].wage_rate for c in core_countries])

        # Adjust for productivity
        production = self.production_data[country]
        productivity_ratio = production.labor_productivity / np.mean([
            self.production_data[c].labor_productivity for c in core_countries
        ])

        productivity_adjusted_wage = avg_core_wage * productivity_ratio

        return productivity_adjusted_wage

    def calculate_value_transfers(self) -> pd.DataFrame:
        """
        Calculate value transfers incorporating both Emmanuel and Amin mechanisms.

        Amin's total transfer = Emmanuel transfer + Super-exploitation transfer +
                                Productivity-blocking transfer

        Returns:
            DataFrame with detailed transfer decomposition
        """
        results = []

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

                # Calculate labor content
                labor_coefficient = exp_prod.labor_hours / exp_prod.gross_output
                labor_exported = labor_coefficient * trade_value

                # 1. Emmanuel-type transfer (wage differential)
                wage_differential = imp_prod.wage_rate - exp_prod.wage_rate
                emmanuel_transfer = labor_exported * wage_differential

                # 2. Productivity-adjusted transfer
                productivity_adjusted_wage = self.calculate_productivity_adjusted_wages(exporter)
                productivity_gap_transfer = labor_exported * (
                    productivity_adjusted_wage - exp_prod.wage_rate
                )

                # 3. Super-exploitation transfer
                exp_category = self.countries[exporter]
                if exp_category == CountryCategory.PERIPHERY:
                    value_labor_power = self.params.value_of_labor_power_periphery
                else:
                    value_labor_power = self.params.value_of_labor_power_core

                super_exploitation_transfer = labor_exported * (
                    value_labor_power - exp_prod.wage_rate
                )

                # 4. Technology transfer (productivity gap not reflected in prices)
                productivity_ratio = exp_prod.labor_productivity / imp_prod.labor_productivity if imp_prod.labor_productivity > 0 else 1
                technology_transfer = trade_value * (1 - productivity_ratio) * 0.5  # Simplified

                # Total Amin transfer
                total_transfer = (
                    emmanuel_transfer +
                    productivity_gap_transfer +
                    super_exploitation_transfer +
                    technology_transfer
                )

                results.append({
                    'exporter': exporter,
                    'importer': importer,
                    'exporter_category': self.countries[exporter].value,
                    'importer_category': self.countries[importer].value,
                    'trade_value': trade_value,
                    'labor_exported_hours': labor_exported,

                    # Transfers by mechanism
                    'emmanuel_transfer': emmanuel_transfer,
                    'productivity_gap_transfer': productivity_gap_transfer,
                    'super_exploitation_transfer': super_exploitation_transfer,
                    'technology_transfer': technology_transfer,
                    'total_transfer': total_transfer,

                    # Diagnostics
                    'exporter_wage': exp_prod.wage_rate,
                    'importer_wage': imp_prod.wage_rate,
                    'productivity_adjusted_wage': productivity_adjusted_wage,
                    'exporter_productivity': exp_prod.labor_productivity,
                    'importer_productivity': imp_prod.labor_productivity,
                    'productivity_ratio': productivity_ratio,

                    'transfer_pct_of_trade': (abs(total_transfer) / trade_value * 100) if trade_value > 0 else 0
                })

        df = pd.DataFrame(results)

        # Calculate net transfers by country
        if not df.empty:
            net_transfers = []
            for country in self.countries.keys():
                # Sum transfers where country is exporter (outflows)
                outflows = df[df['exporter'] == country]['total_transfer'].sum()
                # Sum transfers where country is importer (inflows)
                inflows = df[df['importer'] == country]['total_transfer'].sum()

                # Net transfer (negative = loss, positive = gain)
                net_transfer = inflows - outflows

                net_transfers.append({
                    'country': country,
                    'category': self.countries[country].value,
                    'total_exports': df[df['exporter'] == country]['trade_value'].sum(),
                    'total_imports': df[df['importer'] == country]['trade_value'].sum(),
                    'transfer_outflow': outflows,
                    'transfer_inflow': inflows,
                    'net_transfer': net_transfer,
                    'super_exploitation_rate': self.calculate_super_exploitation_rate(country)
                })

            self.net_transfers_df = pd.DataFrame(net_transfers)

        return df

    def analyze_blocked_development(self) -> Dict[str, any]:
        """
        Analyze how unequal exchange blocks peripheral development.

        Amin argued that value transfers prevent accumulation in periphery,
        forcing it into extroverted (export-oriented) development that
        perpetuates underdevelopment.

        Returns:
            Dictionary with blocked development analysis
        """
        periphery = self.get_periphery_countries()

        if not periphery:
            return {}

        results = {
            'periphery_countries': periphery,
            'country_analysis': {}
        }

        for country in periphery:
            production = self.production_data[country]

            # Calculate surplus that could be available for accumulation
            surplus_generated = production.surplus_value

            # Calculate value transferred out
            if hasattr(self, 'net_transfers_df'):
                net_transfer = self.net_transfers_df[
                    self.net_transfers_df['country'] == country
                ]['net_transfer'].values[0]
            else:
                net_transfer = 0

            # Potential accumulation without transfers
            potential_accumulation = surplus_generated - net_transfer
            actual_accumulation = surplus_generated + net_transfer  # net_transfer is negative for periphery

            # Accumulation blocked
            accumulation_blocked = potential_accumulation - actual_accumulation

            results['country_analysis'][country] = {
                'surplus_generated': surplus_generated,
                'value_transferred_out': -net_transfer,  # Make positive for clarity
                'potential_accumulation': potential_accumulation,
                'actual_accumulation': actual_accumulation,
                'accumulation_blocked_pct': (accumulation_blocked / surplus_generated * 100)
                                           if surplus_generated > 0 else 0,
                'export_orientation': self.params.export_orientation_periphery,
                'import_dependence': self.params.import_dependence_periphery
            }

        return results

    def simulate_autocentric_development(self, country: str,
                                        increase_wage_pct: float = 50,
                                        reduce_export_orientation: float = 0.5) -> Dict[str, float]:
        """
        Simulate Amin's autocentric development strategy.

        Autocentric development involves:
        1. Raising wages to reduce super-exploitation
        2. Reorienting production toward domestic market
        3. Building inter-peripheral linkages
        4. Reducing import dependence

        Args:
            country: Periphery country to simulate
            increase_wage_pct: Percentage increase in wages
            reduce_export_orientation: Factor to reduce export orientation

        Returns:
            Dictionary with simulation results
        """
        if country not in self.countries or self.countries[country] != CountryCategory.PERIPHERY:
            return {}

        # Current state
        production = self.production_data[country]
        current_wage = production.wage_rate
        current_super_exploitation = self.calculate_super_exploitation_rate(country)

        # Get current transfers
        transfers = self.calculate_value_transfers()
        current_transfer = transfers[transfers['exporter'] == country]['total_transfer'].sum()

        # Simulate new wages
        new_wage = current_wage * (1 + increase_wage_pct / 100)

        # Update production data temporarily
        original_prod = production
        self.production_data[country] = ProductionData(
            gross_output=production.gross_output * (1 - reduce_export_orientation + 1),
            labor_hours=production.labor_hours,
            wage_rate=new_wage,
            capital_stock=production.capital_stock,
            intermediate_inputs=production.intermediate_inputs
        )

        # Recalculate
        new_super_exploitation = self.calculate_super_exploitation_rate(country)
        new_transfers = self.calculate_value_transfers()
        new_transfer = new_transfers[new_transfers['exporter'] == country]['total_transfer'].sum()

        # Restore original data
        self.production_data[country] = original_prod

        return {
            'current_wage': current_wage,
            'new_wage': new_wage,
            'wage_increase_pct': increase_wage_pct,
            'current_super_exploitation_rate': current_super_exploitation,
            'new_super_exploitation_rate': new_super_exploitation,
            'current_value_transfer': current_transfer,
            'new_value_transfer': new_transfer,
            'transfer_reduction': current_transfer - new_transfer,
            'transfer_reduction_pct': ((current_transfer - new_transfer) / abs(current_transfer) * 100)
                                     if current_transfer != 0 else 0
        }

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Generate summary statistics for Amin model.

        Returns:
            Dictionary with key outputs
        """
        transfers = self.calculate_value_transfers()

        stats = {
            'total_trade_value': transfers['trade_value'].sum(),
            'total_emmanuel_transfers': transfers['emmanuel_transfer'].abs().sum(),
            'total_productivity_gap_transfers': transfers['productivity_gap_transfer'].abs().sum(),
            'total_super_exploitation_transfers': transfers['super_exploitation_transfer'].abs().sum(),
            'total_technology_transfers': transfers['technology_transfer'].abs().sum(),
            'total_all_transfers': transfers['total_transfer'].abs().sum(),
            'avg_transfer_pct': transfers['transfer_pct_of_trade'].mean(),
        }

        # Category-level statistics
        if hasattr(self, 'net_transfers_df'):
            for category in [CountryCategory.CORE, CountryCategory.PERIPHERY, CountryCategory.SEMI_PERIPHERY]:
                cat_data = self.net_transfers_df[
                    self.net_transfers_df['category'] == category.value
                ]
                if not cat_data.empty:
                    stats[f'{category.value}_net_transfer'] = cat_data['net_transfer'].sum()
                    stats[f'{category.value}_avg_super_exploitation'] = cat_data['super_exploitation_rate'].mean()

        return stats

    def compare_with_emmanuel(self, emmanuel_results: pd.DataFrame) -> pd.DataFrame:
        """
        Compare Amin model results with Emmanuel model.

        Shows additional transfers captured by Amin's extended framework.

        Args:
            emmanuel_results: Results from Emmanuel model

        Returns:
            Comparison DataFrame
        """
        amin_results = self.calculate_value_transfers()

        comparison = []

        for _, row in amin_results.iterrows():
            exporter = row['exporter']
            importer = row['importer']

            # Find corresponding Emmanuel result
            emmanuel_row = emmanuel_results[
                (emmanuel_results['exporter'] == exporter) &
                (emmanuel_results['importer'] == importer)
            ]

            if not emmanuel_row.empty:
                emmanuel_transfer = emmanuel_row['value_transfer'].values[0]
                amin_transfer = row['total_transfer']

                comparison.append({
                    'exporter': exporter,
                    'importer': importer,
                    'trade_value': row['trade_value'],
                    'emmanuel_transfer': emmanuel_transfer,
                    'amin_transfer': amin_transfer,
                    'additional_transfer': amin_transfer - emmanuel_transfer,
                    'amin_to_emmanuel_ratio': (amin_transfer / emmanuel_transfer)
                                             if emmanuel_transfer != 0 else 0
                })

        return pd.DataFrame(comparison)
