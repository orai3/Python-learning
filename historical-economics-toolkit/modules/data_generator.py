"""
Historical Economic Data Generator
===================================

Generates synthetic historical economic data (150+ years) incorporating:
- Structural breaks reflecting regime changes
- Long waves (Kondratiev cycles ~50-60 years)
- Financial crises with clustering
- Distributional dynamics (wage/profit shares)
- Institutional change
- Technology revolutions
- Multiple countries for comparative analysis

Theoretical foundations:
- Regulation School periodization (extensive/intensive accumulation)
- Long wave theory (Kondratiev, Schumpeter, Mandel)
- Historical materialism (crisis tendencies)
- Social Structure of Accumulation (SSA) theory
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HistoricalEconomicDataGenerator:
    """
    Generate synthetic historical economic data with realistic long-run dynamics.

    Implements stylized facts from historical capitalism:
    - Trend growth with declining rate over very long run
    - Cyclical dynamics at multiple frequencies
    - Structural breaks corresponding to regime changes
    - Crisis clustering in certain periods
    - Distributional shifts correlated with institutional change
    """

    def __init__(self,
                 start_year: int = 1870,
                 end_year: int = 2020,
                 frequency: str = 'A',  # A=annual, Q=quarterly
                 seed: int = 42):
        """
        Initialize data generator.

        Parameters
        ----------
        start_year : int
            Starting year for data generation
        end_year : int
            Ending year
        frequency : str
            'A' for annual, 'Q' for quarterly
        seed : int
            Random seed for reproducibility
        """
        self.start_year = start_year
        self.end_year = end_year
        self.frequency = frequency
        self.seed = seed
        np.random.seed(seed)

        # Create time index
        self.years = end_year - start_year + 1
        if frequency == 'A':
            self.periods = self.years
            self.time_index = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='Y'
            )
        else:  # quarterly
            self.periods = self.years * 4
            self.time_index = pd.date_range(
                start=f'{start_year}-01-01',
                end=f'{end_year}-12-31',
                freq='Q'
            )

        # Define historical regimes based on Regulation School periodization
        self.regimes = self._define_regimes()

        # Define crisis dates
        self.crisis_dates = self._define_crisis_dates()

        # Define technology revolutions
        self.tech_revolutions = self._define_tech_revolutions()

    def _define_regimes(self) -> List[Dict]:
        """
        Define historical regimes of accumulation.

        Based on Regulation School periodization:
        1. Competitive capitalism (1870-1914)
        2. Crisis period (1914-1945)
        3. Fordist/Golden Age (1945-1973)
        4. Neoliberal/Financialized (1973-2008)
        5. Post-crisis uncertain (2008-2020)
        """
        regimes = [
            {
                'name': 'Competitive Capitalism',
                'start': 1870,
                'end': 1914,
                'growth_trend': 0.025,  # 2.5% average growth
                'volatility': 0.04,
                'wage_share': 0.55,
                'financialization': 0.3,
                'inequality_trend': 'rising',
                'crisis_probability': 0.05
            },
            {
                'name': 'Crisis Period',
                'start': 1914,
                'end': 1945,
                'growth_trend': 0.01,  # Low growth
                'volatility': 0.08,  # High volatility
                'wage_share': 0.58,
                'financialization': 0.25,
                'inequality_trend': 'falling',
                'crisis_probability': 0.15
            },
            {
                'name': 'Fordist Golden Age',
                'start': 1945,
                'end': 1973,
                'growth_trend': 0.045,  # High growth
                'volatility': 0.025,  # Low volatility
                'wage_share': 0.65,  # High wage share
                'financialization': 0.2,
                'inequality_trend': 'falling',
                'crisis_probability': 0.02
            },
            {
                'name': 'Neoliberal Period',
                'start': 1973,
                'end': 2008,
                'growth_trend': 0.028,  # Moderate growth
                'volatility': 0.035,
                'wage_share': 0.58,  # Falling wage share
                'financialization': 0.6,  # Rising finance
                'inequality_trend': 'rising',
                'crisis_probability': 0.06
            },
            {
                'name': 'Post-Crisis',
                'start': 2008,
                'end': 2020,
                'growth_trend': 0.018,  # Low growth
                'volatility': 0.04,
                'wage_share': 0.56,
                'financialization': 0.65,
                'inequality_trend': 'stable_high',
                'crisis_probability': 0.08
            }
        ]
        return regimes

    def _define_crisis_dates(self) -> List[Dict]:
        """
        Define major financial/economic crises.

        Based on historical record with realistic features:
        - Clustering in certain periods
        - Varying severity
        - Recovery times
        """
        crises = [
            {'year': 1873, 'name': 'Long Depression', 'severity': 0.15, 'duration': 6},
            {'year': 1893, 'name': 'Panic of 1893', 'severity': 0.12, 'duration': 4},
            {'year': 1907, 'name': 'Panic of 1907', 'severity': 0.10, 'duration': 2},
            {'year': 1920, 'name': 'Post-WWI Depression', 'severity': 0.13, 'duration': 3},
            {'year': 1929, 'name': 'Great Depression', 'severity': 0.30, 'duration': 10},
            {'year': 1973, 'name': 'Oil Crisis/Stagflation', 'severity': 0.08, 'duration': 3},
            {'year': 1979, 'name': 'Volcker Shock', 'severity': 0.10, 'duration': 3},
            {'year': 1987, 'name': 'Black Monday', 'severity': 0.05, 'duration': 1},
            {'year': 1997, 'name': 'Asian Financial Crisis', 'severity': 0.07, 'duration': 2},
            {'year': 2000, 'name': 'Dot-com Crash', 'severity': 0.06, 'duration': 2},
            {'year': 2007, 'name': 'Global Financial Crisis', 'severity': 0.18, 'duration': 5},
            {'year': 2020, 'name': 'COVID-19 Recession', 'severity': 0.12, 'duration': 2},
        ]
        return crises

    def _define_tech_revolutions(self) -> List[Dict]:
        """
        Define major technology revolutions (Perez, Freeman).

        Each revolution has:
        - Installation phase (new tech, financial bubble)
        - Turning point (crisis/regulation)
        - Deployment phase (mature growth)
        """
        tech_revs = [
            {'name': 'Second Industrial Revolution', 'start': 1875, 'peak': 1900, 'end': 1920,
             'productivity_boost': 0.015},
            {'name': 'Mass Production', 'start': 1920, 'peak': 1945, 'end': 1970,
             'productivity_boost': 0.025},
            {'name': 'ICT Revolution', 'start': 1970, 'peak': 1995, 'end': 2020,
             'productivity_boost': 0.020},
        ]
        return tech_revs

    def _get_regime_at_year(self, year: int) -> Dict:
        """Get the regime parameters for a given year."""
        for regime in self.regimes:
            if regime['start'] <= year <= regime['end']:
                return regime
        return self.regimes[-1]  # Default to last regime

    def _add_long_waves(self, t: np.ndarray) -> np.ndarray:
        """
        Add Kondratiev long waves (~50-60 year cycles).

        Based on long wave theory (Kondratiev, Schumpeter, Mandel):
        - Period: 50-60 years
        - Amplitude varies by regime
        - Phase shifts at major structural breaks
        """
        # Primary long wave: 54-year cycle
        wave1 = 0.08 * np.sin(2 * np.pi * t / 54 + np.pi/4)

        # Secondary wave: 48-year cycle (interaction)
        wave2 = 0.04 * np.sin(2 * np.pi * t / 48)

        return wave1 + wave2

    def _add_medium_cycles(self, t: np.ndarray) -> np.ndarray:
        """
        Add medium-term business cycles (Juglar cycles, ~8-10 years).
        """
        # Primary business cycle: 9-year
        cycle1 = 0.05 * np.sin(2 * np.pi * t / 9)

        # Investment cycle: 7-year
        cycle2 = 0.03 * np.sin(2 * np.pi * t / 7 + np.pi/3)

        return cycle1 + cycle2

    def _add_short_cycles(self, t: np.ndarray) -> np.ndarray:
        """
        Add short-term cycles (Kitchin inventory cycles, ~3-4 years).
        """
        return 0.02 * np.sin(2 * np.pi * t / 3.5)

    def _generate_crisis_impacts(self) -> np.ndarray:
        """
        Generate crisis impact series.

        Crises cause temporary negative shocks with recovery periods.
        """
        impacts = np.zeros(self.periods)

        for crisis in self.crisis_dates:
            if crisis['year'] < self.start_year or crisis['year'] > self.end_year:
                continue

            crisis_idx = crisis['year'] - self.start_year
            severity = crisis['severity']
            duration = crisis['duration']

            # Crisis impact profile: sharp drop, gradual recovery
            for i in range(duration):
                if crisis_idx + i < self.periods:
                    recovery_factor = (i / duration) ** 0.7  # Nonlinear recovery
                    impacts[crisis_idx + i] -= severity * (1 - recovery_factor)

        return impacts

    def generate_gdp(self, country: str = 'Country1') -> pd.Series:
        """
        Generate GDP series with realistic long-run dynamics.

        Incorporates:
        - Regime-specific trend growth
        - Long waves
        - Business cycles
        - Crisis shocks
        - Stochastic variation
        """
        t = np.arange(self.periods)

        # Initialize log GDP
        log_gdp = np.zeros(self.periods)
        base_level = 10.0  # Log scale starting point

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            regime = self._get_regime_at_year(year)

            # Trend growth (with slight decline over very long run)
            trend = base_level + regime['growth_trend'] * i

            # Add long waves
            long_wave = self._add_long_waves(t)[i]

            # Add medium cycles
            medium_cycle = self._add_medium_cycles(t)[i]

            # Add short cycles
            short_cycle = self._add_short_cycles(t)[i]

            # Stochastic component
            noise = np.random.normal(0, regime['volatility'])

            log_gdp[i] = trend + long_wave + medium_cycle + short_cycle + noise

        # Add crisis impacts
        crisis_impacts = self._generate_crisis_impacts()
        log_gdp += crisis_impacts

        # Convert to levels
        gdp = np.exp(log_gdp)

        return pd.Series(gdp, index=self.time_index, name=f'GDP_{country}')

    def generate_wage_profit_shares(self, gdp: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Generate wage and profit shares with distributional dynamics.

        Features:
        - Regime-specific baseline shares
        - Kalecki-Goodwin cyclical dynamics
        - Structural breaks at regime transitions
        - Crisis impacts on distribution
        """
        t = np.arange(self.periods)
        wage_share = np.zeros(self.periods)

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            regime = self._get_regime_at_year(year)

            # Base wage share for regime
            base_ws = regime['wage_share']

            # Goodwin cycle: counter-cyclical wage share dynamics
            # Wage share tends to rise in booms (worker bargaining power)
            # but effect reversed in neoliberal period
            if year < 1973:
                cyclical = 0.03 * np.sin(2 * np.pi * t[i] / 9 + np.pi/2)
            else:
                cyclical = -0.02 * np.sin(2 * np.pi * t[i] / 9 + np.pi/2)

            # Trend based on regime inequality dynamics
            if regime['inequality_trend'] == 'rising':
                trend = -0.0003 * (i % 100)  # Gradual decline
            elif regime['inequality_trend'] == 'falling':
                trend = 0.0003 * (i % 100)  # Gradual rise
            else:
                trend = 0

            # Crisis impacts: varies by period
            # Early crises hurt labor, post-1945 crises hurt capital
            crisis_impact = 0
            for crisis in self.crisis_dates:
                if crisis['year'] == year:
                    if year < 1945:
                        crisis_impact = -0.02
                    else:
                        crisis_impact = 0.01

            wage_share[i] = base_ws + cyclical + trend + crisis_impact
            wage_share[i] = np.clip(wage_share[i], 0.50, 0.70)  # Realistic bounds

        profit_share = 1 - wage_share

        ws_series = pd.Series(wage_share, index=self.time_index, name='wage_share')
        ps_series = pd.Series(profit_share, index=self.time_index, name='profit_share')

        return ws_series, ps_series

    def generate_inequality(self) -> pd.Series:
        """
        Generate inequality measure (Gini coefficient).

        Features:
        - High inequality in early period
        - Great Compression (1914-1973)
        - Rising inequality in neoliberal period
        - Correlated with wage/profit shares but distinct dynamics
        """
        gini = np.zeros(self.periods)

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            regime = self._get_regime_at_year(year)

            # Base inequality by period
            if year < 1914:
                base = 0.48
            elif year < 1945:
                base = 0.43 - (year - 1914) * 0.0003  # Falling
            elif year < 1973:
                base = 0.35  # Low inequality
            elif year < 2008:
                base = 0.35 + (year - 1973) * 0.0004  # Rising
            else:
                base = 0.49  # High and stable

            # Add noise
            gini[i] = base + np.random.normal(0, 0.01)
            gini[i] = np.clip(gini[i], 0.25, 0.60)

        return pd.Series(gini, index=self.time_index, name='gini')

    def generate_financialization(self) -> pd.Series:
        """
        Generate financialization index (0-1 scale).

        Measures:
        - Financial sector share of GDP
        - Credit to GDP ratio
        - Financial profits / total profits

        Historical pattern:
        - High in pre-1914 (first globalization)
        - Collapse 1914-1945
        - Repression 1945-1973
        - Rapid rise 1973-2007
        - High plateau post-2008
        """
        finc = np.zeros(self.periods)

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            regime = self._get_regime_at_year(year)
            base = regime['financialization']

            # Add trend within regime
            if 1973 <= year < 2008:
                # Strong upward trend in neoliberal period
                base += (year - 1973) * 0.008

            # Crisis impacts: temporary deleveraging
            crisis_impact = 0
            for crisis in self.crisis_dates:
                if crisis['year'] == year:
                    crisis_impact = -0.05
                elif year - crisis['year'] <= 2:  # Recovery period
                    crisis_impact = -0.02

            finc[i] = base + crisis_impact + np.random.normal(0, 0.02)
            finc[i] = np.clip(finc[i], 0.15, 0.75)

        return pd.Series(finc, index=self.time_index, name='financialization')

    def generate_labor_militancy(self) -> pd.Series:
        """
        Generate labor militancy index (strike frequency/intensity).

        Historical pattern:
        - Rising late 19th century
        - Peak WWI era and 1930s
        - Moderate post-WWII
        - Collapse in neoliberal period
        """
        militancy = np.zeros(self.periods)

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            # Base level by period
            if year < 1890:
                base = 0.3
            elif year < 1920:
                base = 0.3 + (year - 1890) * 0.015  # Rising
            elif year < 1945:
                base = 0.6  # High
            elif year < 1970:
                base = 0.5  # Moderate
            elif year < 2000:
                base = 0.5 - (year - 1970) * 0.012  # Declining
            else:
                base = 0.15  # Very low

            # Cyclical component: rises in booms
            cycle = 0.1 * np.sin(2 * np.pi * i / 9)

            # Crisis spikes
            crisis_spike = 0
            for crisis in self.crisis_dates:
                if crisis['year'] == year and year < 1980:
                    crisis_spike = 0.15

            militancy[i] = base + cycle + crisis_spike + np.random.normal(0, 0.05)
            militancy[i] = np.clip(militancy[i], 0.05, 0.85)

        return pd.Series(militancy, index=self.time_index, name='labor_militancy')

    def generate_institutional_index(self) -> pd.Series:
        """
        Generate institutional coordination index.

        Measures degree of institutional coordination/regulation:
        - Low: laissez-faire, weak regulation
        - High: strong state intervention, coordinated capitalism

        Pattern:
        - Low pre-1914
        - Rising 1914-1945
        - High 1945-1973
        - Falling 1973-2008
        - Uncertain post-2008
        """
        inst = np.zeros(self.periods)

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            if year < 1914:
                base = 0.25
            elif year < 1945:
                base = 0.25 + (year - 1914) * 0.015
            elif year < 1973:
                base = 0.75
            elif year < 2008:
                base = 0.75 - (year - 1973) * 0.010
            else:
                base = 0.40

            inst[i] = base + np.random.normal(0, 0.03)
            inst[i] = np.clip(inst[i], 0.15, 0.85)

        return pd.Series(inst, index=self.time_index, name='institutional_coordination')

    def generate_profit_rate(self, gdp: pd.Series) -> pd.Series:
        """
        Generate rate of profit (Marxian measure).

        r = Π / K where Π = profits, K = capital stock

        Features:
        - Tendential fall over long run (Marx)
        - Counter-tendencies in certain periods
        - Cyclical dynamics
        - Restoration attempts via neoliberalism
        """
        t = np.arange(self.periods)
        profit_rate = np.zeros(self.periods)

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            regime = self._get_regime_at_year(year)

            # Base profit rate by regime
            if year < 1914:
                base = 0.12 - i * 0.00008  # Gradual fall
            elif year < 1945:
                base = 0.08  # Crisis-depressed
            elif year < 1973:
                base = 0.15  # Golden Age restoration
            elif year < 2008:
                # Neoliberal attempt to restore, partial success
                base = 0.10 + (year - 1973) * 0.0003
            else:
                base = 0.095  # Post-crisis low

            # Cyclical component
            cycle = 0.02 * np.sin(2 * np.pi * t[i] / 9)

            # Long wave component
            long_wave = 0.015 * np.sin(2 * np.pi * t[i] / 54)

            profit_rate[i] = base + cycle + long_wave + np.random.normal(0, 0.01)
            profit_rate[i] = np.clip(profit_rate[i], 0.03, 0.20)

        return pd.Series(profit_rate, index=self.time_index, name='profit_rate')

    def generate_crisis_indicator(self) -> pd.Series:
        """
        Generate binary crisis indicator.

        1 = crisis year, 0 = normal year
        """
        crisis_ind = np.zeros(self.periods)

        for crisis in self.crisis_dates:
            if self.start_year <= crisis['year'] <= self.end_year:
                idx = crisis['year'] - self.start_year
                for j in range(crisis['duration']):
                    if idx + j < self.periods:
                        crisis_ind[idx + j] = 1

        return pd.Series(crisis_ind, index=self.time_index, name='crisis')

    def generate_hegemony_index(self) -> pd.Series:
        """
        Generate hegemonic power index (Arrighi-style).

        Measures relative power of hegemonic state in world system.
        - High = strong hegemony
        - Low = hegemonic transition/crisis

        Pattern:
        - British hegemony declining 1870-1914
        - Interregnum 1914-1945
        - US hegemony rising then stable 1945-1973
        - US hegemony declining 1973-2020
        """
        hegemony = np.zeros(self.periods)

        for i, year in enumerate(range(self.start_year, self.end_year + 1)):
            if year < 1914:
                base = 0.70 - (year - 1870) * 0.005  # British decline
            elif year < 1945:
                base = 0.35  # Interregnum
            elif year < 1973:
                base = 0.85  # US hegemony peak
            elif year < 2008:
                base = 0.85 - (year - 1973) * 0.006  # US decline
            else:
                base = 0.50  # Uncertain transition

            hegemony[i] = base + np.random.normal(0, 0.03)
            hegemony[i] = np.clip(hegemony[i], 0.25, 0.95)

        return pd.Series(hegemony, index=self.time_index, name='hegemony')

    def generate_complete_dataset(self,
                                   countries: List[str] = None) -> pd.DataFrame:
        """
        Generate complete historical dataset with all variables.

        Parameters
        ----------
        countries : List[str], optional
            List of country names. If None, generates single country.

        Returns
        -------
        pd.DataFrame
            Complete dataset with all variables
        """
        if countries is None:
            countries = ['Country1']

        all_data = []

        for country in countries:
            # Set different seed for each country
            np.random.seed(self.seed + hash(country) % 1000)

            # Generate core series
            gdp = self.generate_gdp(country)
            wage_share, profit_share = self.generate_wage_profit_shares(gdp)

            # Create country dataframe
            df = pd.DataFrame(index=self.time_index)
            df['country'] = country
            df['year'] = self.time_index.year
            df['gdp'] = gdp.values
            df['wage_share'] = wage_share.values
            df['profit_share'] = profit_share.values
            df['gini'] = self.generate_inequality().values
            df['financialization'] = self.generate_financialization().values
            df['labor_militancy'] = self.generate_labor_militancy().values
            df['institutional_coordination'] = self.generate_institutional_index().values
            df['profit_rate'] = self.generate_profit_rate(gdp).values
            df['crisis'] = self.generate_crisis_indicator().values
            df['hegemony'] = self.generate_hegemony_index().values

            # Derived variables
            df['gdp_growth'] = df['gdp'].pct_change()
            df['log_gdp'] = np.log(df['gdp'])

            # Wages and profits (absolute)
            df['total_wages'] = df['gdp'] * df['wage_share']
            df['total_profits'] = df['gdp'] * df['profit_share']

            all_data.append(df)

        # Combine all countries
        full_df = pd.concat(all_data, ignore_index=False)
        full_df = full_df.reset_index()
        full_df = full_df.rename(columns={'index': 'date'})

        return full_df


def generate_multi_country_dataset(start_year: int = 1870,
                                   end_year: int = 2020,
                                   n_countries: int = 5) -> pd.DataFrame:
    """
    Convenience function to generate multi-country dataset.

    Parameters
    ----------
    start_year : int
        Start year
    end_year : int
        End year
    n_countries : int
        Number of countries to generate

    Returns
    -------
    pd.DataFrame
        Multi-country panel dataset
    """
    countries = [f'Country{i+1}' for i in range(n_countries)]

    generator = HistoricalEconomicDataGenerator(
        start_year=start_year,
        end_year=end_year,
        frequency='A'
    )

    df = generator.generate_complete_dataset(countries=countries)

    return df


if __name__ == '__main__':
    # Example: Generate dataset
    print("Generating historical economic dataset (1870-2020)...")

    generator = HistoricalEconomicDataGenerator(
        start_year=1870,
        end_year=2020,
        frequency='A'
    )

    # Generate for 5 countries
    countries = ['USA', 'UK', 'Germany', 'France', 'Japan']
    df = generator.generate_complete_dataset(countries=countries)

    print(f"\nDataset shape: {df.shape}")
    print(f"\nVariables: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nSummary statistics:\n{df.describe()}")

    # Save
    output_path = '../data/historical_economic_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
