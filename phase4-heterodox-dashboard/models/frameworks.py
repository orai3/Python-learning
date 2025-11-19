"""
Theoretical Frameworks for Heterodox Economic Analysis

This module implements analytical frameworks from different heterodox schools:
- Post-Keynesian Economics
- Marxian Political Economy
- Institutionalist Economics

Each framework provides specific indicators, relationships, and interpretations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod


class EconomicFramework(ABC):
    """Abstract base class for economic theoretical frameworks."""

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the framework."""
        pass

    @abstractmethod
    def get_key_indicators(self) -> List[str]:
        """Return list of key indicators for this framework."""
        pass

    @abstractmethod
    def analyze(self, data_model) -> Dict:
        """Perform analysis using this framework."""
        pass

    @abstractmethod
    def get_theoretical_notes(self) -> str:
        """Return theoretical background and key references."""
        pass


class PostKeynesianFramework(EconomicFramework):
    """
    Post-Keynesian Economics Framework

    Key features:
    - Effective demand determines output
    - Endogenous money creation
    - Fundamental uncertainty
    - Stock-Flow Consistent accounting
    - Class conflict and distribution

    References:
    - Lavoie, M. (2014). Post-Keynesian Economics: New Foundations. Edward Elgar.
    - Godley, W., & Lavoie, M. (2007). Monetary Economics. Palgrave Macmillan.
    - Kalecki, M. (1971). Selected Essays on the Dynamics of the Capitalist Economy.
    """

    def get_name(self) -> str:
        return "Post-Keynesian"

    def get_key_indicators(self) -> List[str]:
        return [
            "Wage Share",
            "Profit Share",
            "Capacity Utilization",
            "Government Balance",
            "Private Sector Balance",
            "Foreign Sector Balance",
            "Debt-to-GDP Ratio",
            "Credit Growth",
            "Investment Rate"
        ]

    def analyze(self, data_model) -> Dict:
        """
        Perform Post-Keynesian analysis.

        Returns:
            Dictionary with analysis results
        """
        results = {
            'framework': self.get_name(),
            'indicators': {},
            'sectoral_balances': {},
            'distributional_analysis': {},
            'financial_stability': {},
            'interpretation': []
        }

        try:
            # Load macro dataset
            if 'macro' not in data_model.get_available_datasets():
                data_model.load_dataset('macro')

            # 1. Distributional Analysis
            wage_share = data_model.calculate_wage_share('macro')
            profit_share = data_model.calculate_profit_share('macro')

            results['distributional_analysis'] = {
                'current_wage_share': wage_share.iloc[-1] if len(wage_share) > 0 else None,
                'current_profit_share': profit_share.iloc[-1] if len(profit_share) > 0 else None,
                'wage_share_trend': self._calculate_trend(wage_share),
                'historical_avg_wage_share': wage_share.mean()
            }

            # Interpret wage share
            if wage_share.iloc[-1] < wage_share.mean():
                results['interpretation'].append(
                    "Wage share below historical average suggests profit-led regime "
                    "(Kaleckian interpretation)"
                )

            # 2. Sectoral Balances (Godley's approach)
            if 'sfc' not in data_model.get_available_datasets():
                data_model.load_dataset('sfc')

            sfc_data = data_model.get_dataset('sfc')

            if sfc_data is not None and len(sfc_data) > 0:
                balance_cols = [col for col in sfc_data.columns if 'balance' in col.lower()]

                if balance_cols:
                    latest = sfc_data[balance_cols].iloc[-1]
                    results['sectoral_balances'] = latest.to_dict()

                    # Godley sectoral balances identity check
                    balance_sum = latest.sum()
                    results['sectoral_balances']['identity_check'] = abs(balance_sum) < 0.01

                    # Interpret sectoral positions
                    if 'government_balance' in sfc_data.columns:
                        gov_balance = sfc_data['government_balance'].iloc[-1]
                        if gov_balance < 0:
                            results['interpretation'].append(
                                "Government deficit supports private sector surplus "
                                "(functional finance perspective)"
                            )

            # 3. Financial Stability Indicators (Minsky)
            financial_data = data_model.get_financial_indicators('macro')

            if 'debt_gdp_ratio' in financial_data.columns:
                debt_gdp = financial_data['debt_gdp_ratio']
                results['financial_stability']['debt_gdp_ratio'] = debt_gdp.iloc[-1]
                results['financial_stability']['debt_gdp_trend'] = self._calculate_trend(debt_gdp)

                # Minsky fragility assessment
                if self._calculate_trend(debt_gdp) > 0.5:
                    results['interpretation'].append(
                        "Rising debt-to-GDP ratio suggests increasing financial fragility "
                        "(Minsky's Financial Instability Hypothesis)"
                    )

            # 4. Investment and Growth
            macro_df = data_model.get_dataset('macro')

            if 'investment' in macro_df.columns and 'gdp' in macro_df.columns:
                investment_rate = (macro_df['investment'] / macro_df['gdp'] * 100)
                results['indicators']['investment_rate'] = investment_rate.iloc[-1]

                # Kaleckian principle of effective demand
                if 'capacity_utilization' in macro_df.columns:
                    capacity_util = macro_df['capacity_utilization'].iloc[-1]
                    results['indicators']['capacity_utilization'] = capacity_util

                    if capacity_util < 80:
                        results['interpretation'].append(
                            "Low capacity utilization indicates demand constraint "
                            "(Keynes's principle of effective demand)"
                        )

        except Exception as e:
            results['error'] = str(e)

        return results

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend coefficient."""
        if len(series) < 2:
            return 0.0

        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0

        x = np.arange(len(clean_series))
        y = clean_series.values

        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]

    def get_theoretical_notes(self) -> str:
        return """
        POST-KEYNESIAN ECONOMICS

        Core Principles:
        1. Effective Demand: Output is determined by aggregate demand, not supply
        2. Endogenous Money: Money supply is credit-driven and demand-determined
        3. Fundamental Uncertainty: The future is unknowable, not just risky
        4. Historical Time: Decisions are irreversible, past matters
        5. Class Conflict: Distribution affects demand and growth

        Key Models:
        - Kalecki Profit Equation: Profits = Investment + Capitalist Consumption - Worker Saving
        - Godley Sectoral Balances: Private + Government + Foreign = 0
        - Minsky Financial Instability: Stability breeds instability through increasing leverage

        Policy Implications:
        - Active fiscal policy for full employment
        - Financial regulation to prevent fragility
        - Incomes policy to manage distribution
        - Central bank as lender of last resort

        Major Figures:
        - Michal Kalecki, Joan Robinson, Nicholas Kaldor
        - Hyman Minsky, Wynne Godley, Marc Lavoie
        """


class MarxianFramework(EconomicFramework):
    """
    Marxian Political Economy Framework

    Key features:
    - Labor theory of value
    - Exploitation and surplus value
    - Tendency of rate of profit to fall
    - Accumulation and crisis
    - Class struggle

    References:
    - Marx, K. (1867/1990). Capital, Volume I. Penguin Classics.
    - Shaikh, A. (2016). Capitalism: Competition, Conflict, Crises. Oxford University Press.
    - Foley, D. (1986). Understanding Capital. Harvard University Press.
    """

    def get_name(self) -> str:
        return "Marxian"

    def get_key_indicators(self) -> List[str]:
        return [
            "Rate of Profit",
            "Rate of Surplus Value",
            "Organic Composition of Capital",
            "Wage Share",
            "Unemployment Rate",
            "Labor Productivity",
            "Capital Accumulation Rate"
        ]

    def analyze(self, data_model) -> Dict:
        """
        Perform Marxian analysis.

        Returns:
            Dictionary with analysis results
        """
        results = {
            'framework': self.get_name(),
            'indicators': {},
            'exploitation': {},
            'accumulation': {},
            'crisis_tendencies': {},
            'interpretation': []
        }

        try:
            # Load macro dataset
            if 'macro' not in data_model.get_available_datasets():
                data_model.load_dataset('macro')

            macro_df = data_model.get_dataset('macro')

            # 1. Rate of Profit Analysis
            if 'profits' in macro_df.columns and 'capital_stock' in macro_df.columns:
                rate_of_profit = (macro_df['profits'] / macro_df['capital_stock'] * 100)
                results['indicators']['rate_of_profit'] = rate_of_profit.iloc[-1]
                results['indicators']['rop_trend'] = self._calculate_trend(rate_of_profit)

                # Tendency of rate of profit to fall
                if self._calculate_trend(rate_of_profit) < -0.1:
                    results['interpretation'].append(
                        "Falling rate of profit observed, consistent with Marx's law "
                        "of tendency of rate of profit to fall (TRPF)"
                    )

            # 2. Rate of Surplus Value (Exploitation)
            wage_share = data_model.calculate_wage_share('macro')
            profit_share = data_model.calculate_profit_share('macro')

            if len(wage_share) > 0 and len(profit_share) > 0:
                # Rate of surplus value = surplus value / variable capital
                # Approximation: profit share / wage share
                rate_of_surplus_value = (profit_share / wage_share * 100)
                results['exploitation']['rate_of_surplus_value'] = rate_of_surplus_value.iloc[-1]
                results['exploitation']['rsv_trend'] = self._calculate_trend(rate_of_surplus_value)

                if rate_of_surplus_value.iloc[-1] > rate_of_surplus_value.mean():
                    results['interpretation'].append(
                        "Rate of surplus value above average indicates increasing exploitation"
                    )

            # 3. Organic Composition of Capital
            if 'capital_stock' in macro_df.columns and 'employment' in macro_df.columns:
                # OCC = constant capital / variable capital
                # Approximation: capital per worker
                organic_composition = macro_df['capital_stock'] / macro_df['employment']
                results['indicators']['organic_composition'] = organic_composition.iloc[-1]
                results['indicators']['occ_trend'] = self._calculate_trend(organic_composition)

                # Rising OCC is central to Marx's crisis theory
                if self._calculate_trend(organic_composition) > 0:
                    results['interpretation'].append(
                        "Rising organic composition of capital reflects labor-saving "
                        "technical change (Marx's capital-biased technological change)"
                    )

            # 4. Reserve Army of Labor
            if 'unemployment_rate' in macro_df.columns:
                unemployment = macro_df['unemployment_rate']
                results['indicators']['unemployment_rate'] = unemployment.iloc[-1]

                # High unemployment = large reserve army
                if unemployment.iloc[-1] > 7:
                    results['interpretation'].append(
                        "High unemployment maintains reserve army of labor, "
                        "disciplining workers and suppressing wages"
                    )

            # 5. Accumulation Rate
            if 'investment' in macro_df.columns and 'capital_stock' in macro_df.columns:
                accumulation_rate = (macro_df['investment'] / macro_df['capital_stock'] * 100)
                results['accumulation']['rate'] = accumulation_rate.iloc[-1]
                results['accumulation']['trend'] = self._calculate_trend(accumulation_rate)

            # 6. Productivity and Real Wages
            if 'labor_productivity' in macro_df.columns and 'real_wage' in macro_df.columns:
                productivity = macro_df['labor_productivity']
                real_wage = macro_df['real_wage']

                prod_growth = productivity.pct_change().mean() * 100
                wage_growth = real_wage.pct_change().mean() * 100

                results['exploitation']['productivity_growth'] = prod_growth
                results['exploitation']['real_wage_growth'] = wage_growth

                if prod_growth > wage_growth:
                    results['interpretation'].append(
                        "Productivity growth exceeds wage growth: "
                        "increasing surplus value extraction"
                    )

            # 7. Crisis Indicators
            results['crisis_tendencies'] = {
                'falling_profit_rate': self._calculate_trend(rate_of_profit) < 0 if 'rate_of_profit' in results['indicators'] else None,
                'rising_unemployment': self._calculate_trend(unemployment) > 0 if 'unemployment_rate' in macro_df.columns else None,
                'stagnant_wages': wage_growth < 1 if 'wage_growth' in dir() else None
            }

        except Exception as e:
            results['error'] = str(e)

        return results

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend coefficient."""
        if len(series) < 2:
            return 0.0

        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0

        x = np.arange(len(clean_series))
        y = clean_series.values

        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]

    def get_theoretical_notes(self) -> str:
        return """
        MARXIAN POLITICAL ECONOMY

        Core Principles:
        1. Labor Theory of Value: Labor is the source of value
        2. Exploitation: Surplus value extracted from workers
        3. Accumulation: Drive to expand capital (M-C-M')
        4. Crisis: Inherent contradictions lead to periodic crises
        5. Class Struggle: Conflict between capital and labor

        Key Concepts:
        - Rate of Profit: r = s/(c+v) where s=surplus, c=constant capital, v=variable capital
        - Rate of Surplus Value: s/v (degree of exploitation)
        - Organic Composition of Capital: c/v (capital intensity)
        - Reserve Army of Labor: Unemployed workers discipline wage demands

        Crisis Tendencies:
        1. Tendency of Rate of Profit to Fall (rising OCC)
        2. Underconsumption (wages lag productivity)
        3. Disproportionality (sectoral imbalances)
        4. Financial speculation (fictitious capital)

        Policy Implications:
        - Worker organization and class struggle
        - Socialization of production
        - Democratic economic planning

        Major Figures:
        - Karl Marx, Rosa Luxemburg, Paul Sweezy
        - Anwar Shaikh, Duncan Foley, David Harvey
        """


class InstitutionalistFramework(EconomicFramework):
    """
    Institutionalist Economics Framework

    Key features:
    - Institutions shape economic behavior
    - Power relations and social provisioning
    - Cumulative causation
    - Technological change and evolution
    - Comparative systems analysis

    References:
    - Veblen, T. (1899). The Theory of the Leisure Class.
    - Galbraith, J.K. (1967). The New Industrial State.
    - Hodgson, G. (2015). Conceptualizing Capitalism. University of Chicago Press.
    """

    def get_name(self) -> str:
        return "Institutionalist"

    def get_key_indicators(self) -> List[str]:
        return [
            "Financialization Ratio",
            "Market Concentration",
            "Government Size",
            "Labor Union Density",
            "Inequality Measures",
            "Technological Change",
            "Institutional Quality"
        ]

    def analyze(self, data_model) -> Dict:
        """
        Perform Institutionalist analysis.

        Returns:
            Dictionary with analysis results
        """
        results = {
            'framework': self.get_name(),
            'indicators': {},
            'power_relations': {},
            'institutional_change': {},
            'comparative_analysis': {},
            'interpretation': []
        }

        try:
            # Load datasets
            if 'macro' not in data_model.get_available_datasets():
                data_model.load_dataset('macro')

            if 'inequality' not in data_model.get_available_datasets():
                data_model.load_dataset('inequality')

            macro_df = data_model.get_dataset('macro')
            ineq_df = data_model.get_dataset('inequality')

            # 1. Financialization Analysis (Veblen's "absentee ownership")
            if 'financial_sector_share' in macro_df.columns:
                fin_share = macro_df['financial_sector_share']
                results['indicators']['financialization'] = fin_share.iloc[-1]
                results['indicators']['fin_trend'] = self._calculate_trend(fin_share)

                if fin_share.iloc[-1] > 20:
                    results['interpretation'].append(
                        "High financial sector share indicates financialization, "
                        "consistent with Veblen's critique of absentee ownership"
                    )

            # 2. Power Relations and Distribution
            if ineq_df is not None and 'gini_coefficient' in ineq_df.columns:
                gini = ineq_df['gini_coefficient']
                results['power_relations']['gini'] = gini.iloc[-1]
                results['power_relations']['gini_trend'] = self._calculate_trend(gini)

                if gini.iloc[-1] > 0.4:
                    results['interpretation'].append(
                        "High inequality reflects power imbalances in social provisioning "
                        "(Institutionalist emphasis on power and distribution)"
                    )

            # 3. Government Role (Galbraith's countervailing power)
            if 'government_spending' in macro_df.columns and 'gdp' in macro_df.columns:
                gov_size = (macro_df['government_spending'] / macro_df['gdp'] * 100)
                results['indicators']['government_size'] = gov_size.iloc[-1]

                if gov_size.iloc[-1] > 30:
                    results['interpretation'].append(
                        "Substantial government role provides countervailing power "
                        "to corporate sector (Galbraith)"
                    )

            # 4. Technological Change (Veblen's technological determinism)
            if 'labor_productivity' in macro_df.columns:
                productivity = macro_df['labor_productivity']
                prod_growth = productivity.pct_change().mean() * 100
                results['indicators']['productivity_growth'] = prod_growth

                results['interpretation'].append(
                    f"Productivity growth of {prod_growth:.2f}% reflects technological "
                    "and institutional evolution (Veblen's technological determinism)"
                )

            # 5. Labor Market Institutions
            if 'union_density' in macro_df.columns:
                union_density = macro_df['union_density']
                results['power_relations']['union_density'] = union_density.iloc[-1]

                if self._calculate_trend(union_density) < 0:
                    results['interpretation'].append(
                        "Declining union density indicates shifting power relations "
                        "toward capital (institutional change)"
                    )

            # 6. Cumulative Causation (Myrdal)
            # Analyze whether inequality and growth are mutually reinforcing
            if 'gdp_growth' in macro_df.columns and len(ineq_df) > 0:
                if 'gini_coefficient' in ineq_df.columns:
                    # Check correlation between growth and inequality
                    correlation = macro_df['gdp_growth'].corr(ineq_df['gini_coefficient'])
                    results['institutional_change']['growth_inequality_correlation'] = correlation

                    if abs(correlation) > 0.5:
                        results['interpretation'].append(
                            "Strong growth-inequality correlation suggests cumulative causation "
                            "(Myrdal's circular and cumulative causation)"
                        )

            # 7. Comparative Systems
            if 'panel' in data_model.get_available_datasets():
                panel_df = data_model.get_dataset('panel')
                if panel_df is not None and 'country' in panel_df.columns:
                    # Identify different institutional configurations
                    if 'welfare_regime' in panel_df.columns:
                        regime_counts = panel_df['welfare_regime'].value_counts()
                        results['comparative_analysis']['regime_types'] = regime_counts.to_dict()

                        results['interpretation'].append(
                            "Multiple welfare regimes demonstrate variety of capitalisms "
                            "(Comparative Political Economy perspective)"
                        )

        except Exception as e:
            results['error'] = str(e)

        return results

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend coefficient."""
        if len(series) < 2:
            return 0.0

        clean_series = series.dropna()
        if len(clean_series) < 2:
            return 0.0

        x = np.arange(len(clean_series))
        y = clean_series.values

        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]

    def get_theoretical_notes(self) -> str:
        return """
        INSTITUTIONALIST ECONOMICS

        Core Principles:
        1. Institutions Matter: Rules, norms, and organizations shape economy
        2. Power and Conflict: Economic outcomes reflect power relations
        3. Social Provisioning: Economy serves social reproduction
        4. Evolutionary Change: Cumulative, path-dependent change
        5. Holism: Economy embedded in society and nature

        Key Concepts:
        - Ceremonial vs. Instrumental: Status-seeking vs. problem-solving (Veblen)
        - Countervailing Power: Offsetting market power (Galbraith)
        - Cumulative Causation: Self-reinforcing processes (Myrdal)
        - Technological Determinism: Technology drives institutional change (Veblen)

        Research Methods:
        - Case studies and historical analysis
        - Comparative institutional analysis
        - Pattern models (not regression)
        - Interdisciplinary approaches

        Policy Implications:
        - Institutional reform for social provisioning
        - Democratic economic governance
        - Regulation of corporate power
        - Environmental sustainability

        Major Figures:
        - Thorstein Veblen, John R. Commons, Wesley Mitchell
        - John Kenneth Galbraith, Gunnar Myrdal
        - Ha-Joon Chang, William Lazonick
        """


class FrameworkManager:
    """
    Manages multiple theoretical frameworks and comparative analysis.
    """

    def __init__(self):
        """Initialize with available frameworks."""
        self.frameworks = {
            'post_keynesian': PostKeynesianFramework(),
            'marxian': MarxianFramework(),
            'institutionalist': InstitutionalistFramework()
        }

    def get_framework(self, name: str) -> Optional[EconomicFramework]:
        """Get a framework by name."""
        return self.frameworks.get(name)

    def get_available_frameworks(self) -> List[str]:
        """Get list of available frameworks."""
        return list(self.frameworks.keys())

    def analyze_all(self, data_model) -> Dict:
        """
        Run analysis using all frameworks.

        Returns:
            Dictionary with results from each framework
        """
        results = {}

        for name, framework in self.frameworks.items():
            results[name] = framework.analyze(data_model)

        return results

    def compare_frameworks(self, data_model, indicator: str) -> pd.DataFrame:
        """
        Compare how different frameworks analyze the same indicator.

        Args:
            data_model: DataModel instance
            indicator: Indicator to compare across frameworks

        Returns:
            DataFrame with comparative analysis
        """
        comparison = []

        for name, framework in self.frameworks.items():
            analysis = framework.analyze(data_model)

            # Extract indicator from results
            value = None
            if indicator in analysis.get('indicators', {}):
                value = analysis['indicators'][indicator]
            elif indicator in analysis.get('power_relations', {}):
                value = analysis['power_relations'][indicator]

            comparison.append({
                'framework': framework.get_name(),
                'indicator': indicator,
                'value': value
            })

        return pd.DataFrame(comparison)
