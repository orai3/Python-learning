"""
Policy Simulation Tools

Simulates alternative development strategies based on dependency theory:

1. South-South Cooperation
   - Regional trade blocs
   - Joint industrial development
   - Technology sharing
   - Collective bargaining

2. Delinking Strategies (Samir Amin)
   - Reduction of dependence on core markets
   - Import substitution industrialization
   - Autocentric development
   - New International Economic Order (NIEO)

3. Industrial Policy
   - Strategic sectors development
   - Infant industry protection
   - Export diversification
   - Value chain upgrading

4. Alternative Integration
   - Fair trade agreements
   - Technology transfer mechanisms
   - Debt relief
   - Capital controls

References:
- Amin, S. (1990): Delinking: Towards a Polycentric World
- UNCTAD (1974): New International Economic Order Declaration
- Chang, H-J. (2002): Kicking Away the Ladder
- Gallagher, K. (2005): Putting Development First
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy

from ..core.theoretical_base import ProductionData, CountryCategory
from ..models.emmanuel import EmmanuelModel
from ..models.amin import AminModel


@dataclass
class PolicyParameters:
    """Parameters for policy simulations"""
    # South-South cooperation
    south_south_trade_share: float = 0.3  # Target share of intra-South trade
    technology_transfer_rate: float = 0.05  # Annual productivity gain from cooperation

    # Delinking
    import_substitution_rate: float = 0.20  # Share of imports to substitute
    domestic_market_orientation: float = 0.60  # Share of production for domestic market

    # Industrial policy
    infant_industry_protection: float = 0.30  # Tariff rate for strategic sectors
    r_and_d_investment_rate: float = 0.03  # R&D as share of GDP
    skill_development_rate: float = 0.02  # Annual increase in labor productivity

    # Labor policy
    minimum_wage_floor: float = 0.70  # Minimum wage as share of value of labor-power
    labor_rights_enforcement: float = 0.80  # 0-1 scale

    # Capital controls
    capital_account_restrictions: float = 0.50  # 0-1 scale
    profit_repatriation_tax: float = 0.25  # Tax on repatriated profits


class PolicySimulator:
    """
    Simulates alternative development policies for peripheral countries.
    """

    def __init__(self, base_model: Optional[EmmanuelModel] = None):
        """
        Initialize policy simulator.

        Args:
            base_model: Baseline unequal exchange model
        """
        self.base_model = base_model
        self.scenarios: Dict[str, Dict] = {}

    def simulate_south_south_cooperation(self,
                                        south_countries: List[str],
                                        cooperation_intensity: float = 0.5,
                                        years: int = 20) -> pd.DataFrame:
        """
        Simulate South-South cooperation scenario.

        Models:
        1. Increased intra-South trade (substituting North-South trade)
        2. Technology transfer among Southern countries
        3. Joint bargaining power for terms of trade
        4. Regional value chains

        Args:
            south_countries: List of Southern countries cooperating
            cooperation_intensity: Degree of cooperation (0-1)
            years: Simulation time horizon

        Returns:
            DataFrame with simulation results over time
        """
        results = []

        for year in range(years + 1):
            # Baseline scenario
            if year == 0:
                baseline_transfers = self._calculate_baseline_transfers(south_countries)
                avg_productivity = self._calculate_avg_productivity(south_countries)

                results.append({
                    'year': year,
                    'scenario': 'baseline',
                    'total_value_transfer': baseline_transfers,
                    'avg_productivity': avg_productivity,
                    'intra_south_trade_share': 0.15,  # Starting point
                    'avg_wage_rate': self._calculate_avg_wage(south_countries)
                })

            # Cooperation scenario
            # Productivity gains from technology transfer
            productivity_gain = (year * 0.02 * cooperation_intensity)

            # Reduction in value transfers through:
            # 1. Intra-South trade (no unequal exchange)
            # 2. Improved bargaining power
            intra_south_share = min(0.50, 0.15 + year * 0.015 * cooperation_intensity)
            transfer_reduction = baseline_transfers * (1 - intra_south_share * 0.7)

            # Wage increases from better labor standards coordination
            wage_increase = year * 0.01 * cooperation_intensity

            # Terms of trade improvement from collective bargaining
            tot_improvement = year * 0.005 * cooperation_intensity

            results.append({
                'year': year,
                'scenario': 'south_south_cooperation',
                'total_value_transfer': transfer_reduction,
                'transfer_reduction_from_baseline': baseline_transfers - transfer_reduction,
                'avg_productivity': avg_productivity * (1 + productivity_gain),
                'intra_south_trade_share': intra_south_share,
                'avg_wage_rate': self._calculate_avg_wage(south_countries) * (1 + wage_increase),
                'tot_improvement': tot_improvement * 100
            })

        df = pd.DataFrame(results)

        # Calculate cumulative benefits
        cooperation_only = df[df['scenario'] == 'south_south_cooperation']
        df.loc[df['scenario'] == 'south_south_cooperation', 'cumulative_transfer_reduction'] = \
            cooperation_only['transfer_reduction_from_baseline'].cumsum().values

        return df

    def simulate_delinking(self,
                          country: str,
                          delinking_strategy: str = "moderate",
                          years: int = 30) -> pd.DataFrame:
        """
        Simulate delinking strategy (Samir Amin).

        Delinking ≠ Autarky. It means:
        - Subordinating external relations to internal development logic
        - Reducing dependence on core countries
        - Building autocentric accumulation
        - Selective engagement with world economy

        Strategies:
        - "moderate": Gradual reorientation, maintain some core linkages
        - "radical": Rapid import substitution, prioritize South-South
        - "selective": Strategic delinking in key sectors only

        Args:
            country: Country to simulate
            delinking_strategy: Type of delinking
            years: Simulation time horizon

        Returns:
            DataFrame with simulation results
        """
        # Strategy parameters
        if delinking_strategy == "moderate":
            import_sub_rate = 0.15  # 15% of imports substituted over period
            export_reduction = 0.10  # 10% reduction in exports to North
            productivity_penalty = 0.05  # 5% initial productivity hit
        elif delinking_strategy == "radical":
            import_sub_rate = 0.40
            export_reduction = 0.30
            productivity_penalty = 0.15
        else:  # selective
            import_sub_rate = 0.25
            export_reduction = 0.15
            productivity_penalty = 0.08

        results = []

        baseline_gdp = 1000  # Normalized
        baseline_imports = 300
        baseline_exports = 280
        baseline_value_transfer = 50

        for year in range(years + 1):
            if year == 0:
                # Baseline
                results.append({
                    'year': year,
                    'scenario': 'baseline',
                    'gdp': baseline_gdp,
                    'imports': baseline_imports,
                    'exports': baseline_exports,
                    'value_transfer': baseline_value_transfer,
                    'domestic_market_share': 0.40
                })

            # Delinking scenario
            # Short-term costs
            if year <= 5:
                # Initial productivity penalty
                gdp_effect = -productivity_penalty * (5 - year) / 5
            else:
                # Long-term gains from autocentric development
                gdp_effect = 0.02 * (year - 5)  # 2% annual growth bonus

            # Import substitution
            imports_reduced = baseline_imports * (1 - import_sub_rate * min(1, year / years))

            # Export reorientation (some exports redirected to South-South)
            exports_reduced = baseline_exports * (1 - export_reduction * min(1, year / years))

            # Value transfer reduction
            transfer_reduced = baseline_value_transfer * (
                1 - 0.6 * min(1, year / years)  # 60% reduction over period
            )

            # GDP growth
            gdp = baseline_gdp * (1 + gdp_effect)

            # Domestic market share increases
            domestic_share = 0.40 + 0.30 * min(1, year / years)

            results.append({
                'year': year,
                'scenario': f'delinking_{delinking_strategy}',
                'gdp': gdp,
                'gdp_vs_baseline_pct': ((gdp - baseline_gdp) / baseline_gdp * 100),
                'imports': imports_reduced,
                'exports': exports_reduced,
                'trade_balance': exports_reduced - imports_reduced,
                'value_transfer': transfer_reduced,
                'transfer_reduction': baseline_value_transfer - transfer_reduced,
                'domestic_market_share': domestic_share,
                'economic_sovereignty_index': (1 - imports_reduced / baseline_imports) * 100
            })

        return pd.DataFrame(results)

    def simulate_industrial_policy(self,
                                  country: str,
                                  policy_package: str = "comprehensive",
                                  years: int = 25) -> pd.DataFrame:
        """
        Simulate industrial policy interventions.

        Policy packages:
        - "infant_industry": Protection for strategic sectors
        - "export_diversification": Move from primary to manufactured exports
        - "value_chain_upgrading": Move up value chain in existing sectors
        - "comprehensive": Combination of above

        Args:
            country: Country to simulate
            policy_package: Type of industrial policy
            years: Simulation time horizon

        Returns:
            DataFrame with simulation results
        """
        results = []

        # Baseline
        baseline_manuf_share = 0.20  # Manufacturing as share of exports
        baseline_productivity = 100
        baseline_value_added_share = 0.25  # Share of value in GVCs

        for year in range(years + 1):
            if year == 0:
                results.append({
                    'year': year,
                    'scenario': 'baseline',
                    'manufacturing_export_share': baseline_manuf_share,
                    'productivity_index': baseline_productivity,
                    'value_added_share_gvc': baseline_value_added_share,
                    'technological_capability': 30
                })

            # Industrial policy effects
            if policy_package == "infant_industry":
                manuf_growth = 0.02  # 2% annual increase
                productivity_growth = 0.025
                vc_upgrading = 0.01
                tech_capability_growth = 1.5
            elif policy_package == "export_diversification":
                manuf_growth = 0.03
                productivity_growth = 0.02
                vc_upgrading = 0.015
                tech_capability_growth = 2.0
            elif policy_package == "value_chain_upgrading":
                manuf_growth = 0.015
                productivity_growth = 0.03
                vc_upgrading = 0.025
                tech_capability_growth = 2.5
            else:  # comprehensive
                manuf_growth = 0.035
                productivity_growth = 0.035
                vc_upgrading = 0.03
                tech_capability_growth = 3.0

            # Apply policies
            manuf_share = min(0.70, baseline_manuf_share + year * manuf_growth)
            productivity = baseline_productivity * ((1 + productivity_growth) ** year)
            vc_share = min(0.60, baseline_value_added_share + year * vc_upgrading)
            tech_cap = min(100, 30 + year * tech_capability_growth)

            # Calculate value retention (higher value-added = less leakage)
            value_retention = 0.30 + (vc_share - baseline_value_added_share) * 2

            results.append({
                'year': year,
                'scenario': f'industrial_policy_{policy_package}',
                'manufacturing_export_share': manuf_share,
                'productivity_index': productivity,
                'value_added_share_gvc': vc_share,
                'technological_capability': tech_cap,
                'value_retention_rate': value_retention,
                'terms_of_trade_effect': (manuf_share - baseline_manuf_share) * 10
            })

        return pd.DataFrame(results)

    def simulate_alternative_integration(self,
                                        countries: List[str],
                                        integration_type: str = "fair_trade",
                                        years: int = 15) -> pd.DataFrame:
        """
        Simulate alternative forms of international integration.

        Integration types:
        - "fair_trade": Equitable trade agreements with labor/environment standards
        - "technology_commons": Open-source technology sharing
        - "debt_relief": Cancellation/reduction of external debt
        - "capital_controls": Restrictions on financial flows

        Args:
            countries: Countries in alternative integration scheme
            integration_type: Type of integration
            years: Simulation time horizon

        Returns:
            DataFrame with results
        """
        results = []

        baseline_debt_service = 50
        baseline_ip_payments = 30
        baseline_profit_repatriation = 40

        for year in range(years + 1):
            if year == 0:
                results.append({
                    'year': year,
                    'scenario': 'baseline',
                    'debt_service': baseline_debt_service,
                    'ip_payments': baseline_ip_payments,
                    'profit_repatriation': baseline_profit_repatriation,
                    'total_outflows': baseline_debt_service + baseline_ip_payments + baseline_profit_repatriation
                })

            # Alternative integration effects
            if integration_type == "fair_trade":
                debt_reduction = 0.02 * year
                ip_reduction = 0.01 * year  # Technology transfer provisions
                profit_reduction = 0.015 * year  # Labor standards raise wages
                productivity_gain = 0.015 * year
            elif integration_type == "technology_commons":
                debt_reduction = 0
                ip_reduction = 0.05 * year  # Major IP savings
                profit_reduction = 0.01 * year
                productivity_gain = 0.03 * year  # Rapid tech adoption
            elif integration_type == "debt_relief":
                debt_reduction = min(1.0, 0.10 * year)  # 10% per year
                ip_reduction = 0
                profit_reduction = 0.005 * year
                productivity_gain = 0.01 * year  # Resources freed for investment
            else:  # capital_controls
                debt_reduction = 0.015 * year
                ip_reduction = 0.01 * year
                profit_reduction = 0.04 * year  # Major reduction in profit outflows
                productivity_gain = 0.02 * year

            debt_service = baseline_debt_service * (1 - debt_reduction)
            ip_payments = baseline_ip_payments * (1 - ip_reduction)
            profit_repat = baseline_profit_repatriation * (1 - profit_reduction)

            total_outflows = debt_service + ip_payments + profit_repat
            outflow_reduction = (baseline_debt_service + baseline_ip_payments +
                                baseline_profit_repatriation) - total_outflows

            results.append({
                'year': year,
                'scenario': f'{integration_type}_integration',
                'debt_service': debt_service,
                'ip_payments': ip_payments,
                'profit_repatriation': profit_repat,
                'total_outflows': total_outflows,
                'outflow_reduction': outflow_reduction,
                'cumulative_savings': outflow_reduction * (year + 1),  # Simplified
                'productivity_gain_pct': productivity_gain * 100,
                'fiscal_space_created': outflow_reduction
            })

        return pd.DataFrame(results)

    def compare_scenarios(self, *scenario_results: pd.DataFrame) -> pd.DataFrame:
        """
        Compare multiple policy scenarios.

        Args:
            *scenario_results: Multiple scenario DataFrames

        Returns:
            Comparison DataFrame
        """
        # Concatenate all scenarios
        combined = pd.concat(scenario_results, ignore_index=True)

        return combined

    def _calculate_baseline_transfers(self, countries: List[str]) -> float:
        """Calculate baseline value transfers for countries"""
        # Placeholder - would use actual model
        return 100.0 * len(countries)

    def _calculate_avg_productivity(self, countries: List[str]) -> float:
        """Calculate average productivity"""
        # Placeholder
        return 50.0

    def _calculate_avg_wage(self, countries: List[str]) -> float:
        """Calculate average wage"""
        # Placeholder
        return 10.0

    def evaluate_policy_effectiveness(self, scenario_df: pd.DataFrame,
                                     objectives: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate policy scenario against objectives.

        Args:
            scenario_df: Scenario results
            objectives: Dictionary of objectives and target values
                       e.g., {'value_transfer_reduction': 50, 'gdp_growth': 0.05}

        Returns:
            Effectiveness scores
        """
        results = {}

        final_year = scenario_df[scenario_df['scenario'] != 'baseline'].iloc[-1]

        for objective, target in objectives.items():
            if objective in final_year:
                actual = final_year[objective]
                achievement = (actual / target * 100) if target != 0 else 0
                results[f'{objective}_achievement_pct'] = achievement
                results[f'{objective}_actual'] = actual
                results[f'{objective}_target'] = target

        return results
