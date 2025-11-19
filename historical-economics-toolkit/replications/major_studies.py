"""
Replication of Major Historical Economics Studies
=================================================

Implements analytical frameworks from major heterodox studies of capitalist development:

1. Robert Brenner - The Economics of Global Turbulence
   - International profit rate dynamics
   - Overcapacity and inter-capitalist competition
   - Long downturn thesis (1973-present)

2. Giovanni Arrighi - The Long Twentieth Century
   - Systemic cycles of accumulation
   - Material vs financial expansion phases
   - Hegemonic transitions

3. Gérard Duménil & Dominique Lévy - Capital Resurgent / The Crisis of Neoliberalism
   - Profit rate analysis
   - Class power dynamics
   - Managerial vs financial capitalism
   - Neoliberal counterrevolution

Each replication includes:
- Theoretical framework
- Data requirements
- Key calculations
- Stylized results
- Interpretation guide
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt


class BrennerAnalysis:
    """
    Replicate Robert Brenner's analysis of post-war capitalism.

    Key arguments:
    - International overcapacity in manufacturing (1965+)
    - Persistent downward pressure on profit rates
    - Failed attempts at restoration
    - Financialization as symptom of productive stagnation

    Reference:
    Brenner, R. (2006). The Economics of Global Turbulence. Verso.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize Brenner analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Historical data with profit rates, investment, etc.
        """
        self.data = data

    def calculate_profit_rate_trend(self,
                                   profit_rate_var: str = 'profit_rate',
                                   start_year: int = 1945) -> Dict:
        """
        Calculate long-run profit rate trend.

        Brenner identifies:
        - Rising profit rates: 1945-1965
        - Falling profit rates: 1965-1982
        - Stagnant/volatile: 1982-present

        Parameters
        ----------
        profit_rate_var : str
            Profit rate variable
        start_year : int
            Starting year for analysis

        Returns
        -------
        Dict
            Profit rate trends by period
        """
        df = self.data[self.data['year'] >= start_year].copy()
        df = df[df[profit_rate_var].notna()]

        # Define Brenner periods
        periods = [
            {'name': 'Golden Age (Profit Rate Rise)', 'start': 1945, 'end': 1965},
            {'name': 'Long Downturn Begin (Profit Rate Fall)', 'start': 1965, 'end': 1982},
            {'name': 'Neoliberal Era (Incomplete Restoration)', 'start': 1982, 'end': 2007},
            {'name': 'Great Recession Era', 'start': 2007, 'end': 2020}
        ]

        results = []

        for period in periods:
            period_data = df[
                (df['year'] >= period['start']) &
                (df['year'] <= period['end'])
            ]

            if len(period_data) == 0:
                continue

            # Calculate trend
            years = period_data['year'].values
            profit_rates = period_data[profit_rate_var].values

            if len(years) > 1:
                # Linear trend
                coeffs = np.polyfit(years - years[0], profit_rates, 1)
                slope = coeffs[0]
                avg_level = profit_rates.mean()

                # Annual change
                annual_change = (profit_rates[-1] - profit_rates[0]) / len(years)
            else:
                slope = 0
                avg_level = profit_rates[0]
                annual_change = 0

            results.append({
                'period': period['name'],
                'start_year': period['start'],
                'end_year': period['end'],
                'avg_profit_rate': avg_level,
                'trend_slope': slope,
                'annual_change': annual_change,
                'start_level': profit_rates[0],
                'end_level': profit_rates[-1],
                'total_change': profit_rates[-1] - profit_rates[0]
            })

        return {
            'periods': pd.DataFrame(results),
            'interpretation': self._interpret_brenner_results(results)
        }

    def analyze_investment_profit_relationship(self,
                                              profit_var: str = 'profit_rate',
                                              gdp_var: str = 'gdp_growth') -> Dict:
        """
        Analyze relationship between profitability and accumulation.

        Brenner argues falling profitability → weak investment → slow growth

        Parameters
        ----------
        profit_var : str
            Profitability measure
        gdp_var : str
            Growth measure

        Returns
        -------
        Dict
            Correlation analysis by period
        """
        df = self.data[[profit_var, gdp_var, 'year']].dropna()

        # Calculate correlations by period
        periods = [
            {'name': 'Golden Age', 'start': 1945, 'end': 1973},
            {'name': 'Long Downturn', 'start': 1973, 'end': 2007}
        ]

        results = []

        for period in periods:
            period_data = df[
                (df['year'] >= period['start']) &
                (df['year'] <= period['end'])
            ]

            if len(period_data) < 5:
                continue

            correlation = period_data[profit_var].corr(period_data[gdp_var])

            results.append({
                'period': period['name'],
                'correlation': correlation,
                'avg_profit_rate': period_data[profit_var].mean(),
                'avg_growth': period_data[gdp_var].mean()
            })

        return {
            'correlation_analysis': pd.DataFrame(results),
            'thesis': 'Brenner: Falling profits → weak accumulation → slow growth'
        }

    def _interpret_brenner_results(self, results: List[Dict]) -> str:
        """Generate interpretation of results."""
        interpretation = "Brenner's Long Downturn Thesis:\n\n"

        for result in results:
            direction = "rising" if result['total_change'] > 0 else "falling"
            interpretation += f"{result['period']}: Profit rate {direction} "
            interpretation += f"({result['total_change']:.3f} total change)\n"

        interpretation += "\nKey finding: Persistent pressure on profit rates post-1965, "
        interpretation += "with incomplete restoration under neoliberalism."

        return interpretation


class ArrighiAnalysis:
    """
    Replicate Giovanni Arrighi's systemic cycles of accumulation.

    Key concepts:
    - Each hegemonic cycle has material & financial expansion phases
    - Financial expansion = "signal crisis" (autumn of hegemony)
    - Hegemonic transitions through competition and conflict

    Reference:
    Arrighi, G. (1994). The Long Twentieth Century. Verso.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize Arrighi analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Historical data with financialization, hegemony indicators
        """
        self.data = data

    @staticmethod
    def get_systemic_cycles() -> pd.DataFrame:
        """
        Return Arrighi's systemic cycles of accumulation.

        Returns
        -------
        pd.DataFrame
            Historical cycles
        """
        cycles = [
            {
                'hegemon': 'Genoa',
                'material_start': 1340,
                'material_end': 1560,
                'financial_start': 1560,
                'financial_end': 1640,
                'duration': 300,
                'core_activity': 'Trade networks'
            },
            {
                'hegemon': 'Dutch Republic',
                'material_start': 1560,
                'material_end': 1740,
                'financial_start': 1740,
                'financial_end': 1815,
                'duration': 255,
                'core_activity': 'Commercial capitalism'
            },
            {
                'hegemon': 'Britain',
                'material_start': 1780,
                'material_end': 1870,
                'financial_start': 1870,
                'financial_end': 1945,
                'duration': 165,
                'core_activity': 'Industrial capitalism'
            },
            {
                'hegemon': 'United States',
                'material_start': 1870,
                'material_end': 1970,
                'financial_start': 1970,
                'financial_end': None,  # Ongoing?
                'duration': 150,  # So far
                'core_activity': 'Corporate capitalism'
            }
        ]

        return pd.DataFrame(cycles)

    def identify_accumulation_phases(self,
                                    financialization_var: str = 'financialization',
                                    threshold: float = 0.45) -> pd.DataFrame:
        """
        Classify years as material vs financial expansion.

        Material expansion: productive investment dominates
        Financial expansion: financial activity dominates (signal crisis)

        Parameters
        ----------
        financialization_var : str
            Financialization indicator
        threshold : float
            Threshold for financial expansion

        Returns
        -------
        pd.DataFrame
            Phase classification
        """
        df = self.data[['year', financialization_var]].dropna()

        df['phase'] = 'Material Expansion'
        df.loc[df[financialization_var] > threshold, 'phase'] = 'Financial Expansion'

        # Identify phase transitions
        df['phase_change'] = df['phase'] != df['phase'].shift()

        # Calculate phase durations
        phases = []
        current_phase = df.iloc[0]['phase']
        start_year = df.iloc[0]['year']

        for idx, row in df.iterrows():
            if row['phase_change'] and idx > 0:
                phases.append({
                    'phase': current_phase,
                    'start_year': int(start_year),
                    'end_year': int(df.loc[idx-1, 'year']),
                    'duration': int(df.loc[idx-1, 'year'] - start_year + 1)
                })
                current_phase = row['phase']
                start_year = row['year']

        # Add final phase
        phases.append({
            'phase': current_phase,
            'start_year': int(start_year),
            'end_year': int(df.iloc[-1]['year']),
            'duration': int(df.iloc[-1]['year'] - start_year + 1)
        })

        return {
            'phase_classification': df,
            'phase_periods': pd.DataFrame(phases),
            'interpretation': self._interpret_arrighi_phases(phases)
        }

    def analyze_us_cycle(self,
                        start_year: int = 1945,
                        financialization_var: str = 'financialization') -> Dict:
        """
        Analyze US hegemonic cycle in detail.

        Parameters
        ----------
        start_year : int
            Start of analysis (post-WWII)
        financialization_var : str
            Financialization indicator

        Returns
        -------
        Dict
            US cycle analysis
        """
        df = self.data[self.data['year'] >= start_year].copy()

        # Identify material expansion peak (around 1970)
        material_period = df[df['year'] <= 1973]
        if len(material_period) > 0 and financialization_var in material_period.columns:
            material_avg_finc = material_period[financialization_var].mean()
        else:
            material_avg_finc = None

        # Financial expansion (1970s onward)
        financial_period = df[df['year'] >= 1973]
        if len(financial_period) > 0 and financialization_var in financial_period.columns:
            financial_avg_finc = financial_period[financialization_var].mean()
            finc_growth_rate = (financial_period[financialization_var].iloc[-1] -
                              financial_period[financialization_var].iloc[0]) / len(financial_period)
        else:
            financial_avg_finc = None
            finc_growth_rate = None

        return {
            'hegemon': 'United States',
            'material_expansion': '1945-1970',
            'financial_expansion': '1970-present',
            'material_avg_financialization': material_avg_finc,
            'financial_avg_financialization': financial_avg_finc,
            'financialization_growth_rate': finc_growth_rate,
            'interpretation': 'US entered financial expansion phase ~1970, signaling hegemonic decline'
        }

    def _interpret_arrighi_phases(self, phases: List[Dict]) -> str:
        """Interpret phase results."""
        interpretation = "Arrighi's Systemic Cycles Framework:\n\n"

        for phase in phases:
            interpretation += f"{phase['phase']}: {phase['start_year']}-{phase['end_year']} "
            interpretation += f"({phase['duration']} years)\n"

        interpretation += "\nKey insight: Financial expansion phases represent 'autumn' of hegemony, "
        interpretation += "as capital shifts from productive to financial circuits."

        return interpretation


class DumenilLevyAnalysis:
    """
    Replicate Duménil & Lévy's analysis of neoliberalism and class power.

    Key arguments:
    - Neoliberalism as restoration of capitalist/financial class power
    - Profit rate analysis with class decomposition
    - Managerial vs financial capitalism
    - First structural crisis (1970s) → neoliberal response → second crisis (2008)

    References:
    Duménil, G., & Lévy, D. (2004). Capital Resurgent. Harvard UP.
    Duménil, G., & Lévy, D. (2011). The Crisis of Neoliberalism. Harvard UP.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize Duménil-Lévy analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Historical data
        """
        self.data = data

    def identify_structural_crises(self) -> Dict:
        """
        Identify Duménil-Lévy's structural crises of capitalism.

        First structural crisis: Late 1960s-1970s
        - Falling profit rates
        - Labor militancy
        - Inflation

        Second structural crisis: 2008-present
        - Financial crisis
        - Limits of neoliberal model
        - Stagnation

        Returns
        -------
        Dict
            Structural crises and characteristics
        """
        crises = [
            {
                'crisis': 'First Structural Crisis',
                'period': '1967-1982',
                'characteristics': [
                    'Falling profit rates',
                    'Rising labor power',
                    'Stagflation',
                    'End of Bretton Woods'
                ],
                'response': 'Neoliberal counterrevolution',
                'class_dynamic': 'Capitalist class loses power → mobilizes for restoration'
            },
            {
                'crisis': 'Second Structural Crisis',
                'period': '2007-present',
                'characteristics': [
                    'Financial crisis',
                    'Great Recession',
                    'Secular stagnation',
                    'Inequality crisis'
                ],
                'response': 'Uncertain (austerity, QE, rising populism)',
                'class_dynamic': 'Limits of financialization, popular discontent'
            }
        ]

        return {
            'structural_crises': pd.DataFrame(crises),
            'framework': 'Duménil-Lévy: Structural crises arise from contradictions, '
                        'trigger class struggle over institutional restructuring'
        }

    def analyze_neoliberal_restoration(self,
                                      profit_var: str = 'profit_rate',
                                      wage_var: str = 'wage_share',
                                      finc_var: str = 'financialization') -> Dict:
        """
        Analyze neoliberal restoration of capitalist power (1980s onward).

        Key metrics:
        - Profit rate recovery?
        - Wage share decline (labor's weakening)
        - Financialization rise
        - Inequality increase

        Parameters
        ----------
        profit_var : str
            Profit rate variable
        wage_var : str
            Wage share variable
        finc_var : str
            Financialization variable

        Returns
        -------
        Dict
            Neoliberal period analysis
        """
        # Pre-neoliberal (Golden Age)
        golden_age = self.data[
            (self.data['year'] >= 1945) &
            (self.data['year'] <= 1973)
        ]

        # Neoliberal era
        neoliberal = self.data[
            (self.data['year'] >= 1980) &
            (self.data['year'] <= 2007)
        ]

        # Calculate changes
        results = {}

        for var, name in [(profit_var, 'profit_rate'),
                         (wage_var, 'wage_share'),
                         (finc_var, 'financialization')]:

            if var in golden_age.columns and var in neoliberal.columns:
                ga_mean = golden_age[var].mean()
                neo_mean = neoliberal[var].mean()
                change = neo_mean - ga_mean

                results[name] = {
                    'golden_age_avg': ga_mean,
                    'neoliberal_avg': neo_mean,
                    'change': change,
                    'percent_change': (change / ga_mean * 100) if ga_mean != 0 else None
                }

        interpretation = self._interpret_dumenil_levy_results(results)

        return {
            'comparison': results,
            'interpretation': interpretation
        }

    def calculate_class_power_index(self,
                                   wage_var: str = 'wage_share',
                                   labor_var: str = 'labor_militancy') -> pd.DataFrame:
        """
        Construct index of labor vs capital power.

        Higher values = greater labor power
        Lower values = greater capital power

        Parameters
        ----------
        wage_var : str
            Wage share (labor's income share)
        labor_var : str
            Labor militancy index

        Returns
        -------
        pd.DataFrame
            Class power index over time
        """
        vars_available = [v for v in [wage_var, labor_var] if v in self.data.columns]

        if len(vars_available) == 0:
            return pd.DataFrame()

        df = self.data[['year'] + vars_available].dropna()

        # Normalize to 0-1 scale
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        df['labor_power_index'] = scaler.fit_transform(
            df[vars_available].values
        ).mean(axis=1)

        df['capital_power_index'] = 1 - df['labor_power_index']

        return df[['year', 'labor_power_index', 'capital_power_index']]

    def _interpret_dumenil_levy_results(self, results: Dict) -> str:
        """Interpret results."""
        interpretation = "Duménil-Lévy Framework: Neoliberal Restoration\n\n"

        interpretation += "Comparison: Golden Age (1945-1973) vs Neoliberal Era (1980-2007)\n\n"

        for metric, values in results.items():
            interpretation += f"{metric.replace('_', ' ').title()}:\n"
            interpretation += f"  Golden Age: {values['golden_age_avg']:.3f}\n"
            interpretation += f"  Neoliberal: {values['neoliberal_avg']:.3f}\n"
            interpretation += f"  Change: {values['change']:.3f}\n\n"

        interpretation += "Key finding: Neoliberalism represented successful (though incomplete) "
        interpretation += "restoration of capitalist class power after crisis of 1970s."

        return interpretation


def compare_all_frameworks(data: pd.DataFrame) -> Dict:
    """
    Compare all three frameworks on same data.

    Shows complementarity and differences in periodization and emphasis.

    Parameters
    ----------
    data : pd.DataFrame
        Historical data

    Returns
    -------
    Dict
        Comparative analysis
    """
    brenner = BrennerAnalysis(data)
    arrighi = ArrighiAnalysis(data)
    dumenil_levy = DumenilLevyAnalysis(data)

    # Get key periodizations
    brenner_periods = brenner.calculate_profit_rate_trend()
    arrighi_phases = arrighi.identify_accumulation_phases()
    dl_crises = dumenil_levy.identify_structural_crises()

    comparison = {
        'Brenner': {
            'focus': 'International competition and overcapacity',
            'key_variable': 'Profit rate',
            'turning_point': '1965 (onset of long downturn)',
            'periodization': brenner_periods['periods']
        },
        'Arrighi': {
            'focus': 'Hegemonic cycles and financialization',
            'key_variable': 'Financialization index',
            'turning_point': '1970 (shift to financial expansion)',
            'periodization': arrighi_phases['phase_periods']
        },
        'Duménil-Lévy': {
            'focus': 'Class power and institutional change',
            'key_variable': 'Wage share / class power',
            'turning_point': '1970s (first structural crisis)',
            'periodization': dl_crises['structural_crises']
        }
    }

    return {
        'frameworks': comparison,
        'synthesis': 'All three identify 1965-1975 as critical turning point. '
                    'Brenner emphasizes profitability crisis, Arrighi financialization, '
                    'Duménil-Lévy class conflict. Complementary rather than contradictory.'
    }


if __name__ == '__main__':
    print("Replication Studies module loaded successfully.")
    print("\nAvailable classes:")
    print("- BrennerAnalysis: Profit rates and long downturn")
    print("- ArrighiAnalysis: Systemic cycles of accumulation")
    print("- DumenilLevyAnalysis: Class power and neoliberalism")
    print("\nAvailable functions:")
    print("- compare_all_frameworks: Comparative framework analysis")
