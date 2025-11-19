"""
Crisis Analysis and Hegemonic Transitions
=========================================

Analyzes economic crises and hegemonic transitions in world capitalism.

Crisis Analysis:
- Crisis identification and dating
- Severity measurement
- Duration analysis
- Clustering patterns
- Recovery dynamics
- Leading indicators

Hegemonic Transitions:
- Arrighi's systemic cycles of accumulation
- Hegemonic rise and decline
- Transition periods (interregna)
- Financial vs productive hegemonies
- Material vs financial expansion phases

Theoretical foundations:
- Arrighi: The Long Twentieth Century
- Kindleberger-Minsky financial instability
- World-systems theory (Wallerstein)
- Regulation School crisis theory
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CrisisAnalyzer:
    """
    Identify and analyze economic/financial crises.

    Methods:
    - Algorithmic crisis detection
    - Severity measurement
    - Duration analysis
    - Recovery patterns
    - Crisis clustering
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize crisis analyzer.

        Parameters
        ----------
        data : pd.DataFrame
            Historical economic data
        """
        self.data = data
        self.crises = []

    def detect_crises(self,
                     variable: str = 'gdp_growth',
                     threshold: float = -0.02,
                     min_duration: int = 1) -> pd.DataFrame:
        """
        Detect crisis periods using growth threshold.

        Parameters
        ----------
        variable : str
            Variable to use for detection (typically GDP growth)
        threshold : float
            Threshold for crisis (e.g., -0.02 = -2% growth)
        min_duration : int
            Minimum duration in years to count as crisis

        Returns
        -------
        pd.DataFrame
            Detected crises with characteristics
        """
        df = self.data[self.data[variable].notna()].copy()
        df = df.sort_values('year')

        # Identify crisis years
        df['is_crisis'] = df[variable] < threshold

        # Identify crisis episodes (consecutive years)
        df['crisis_start'] = df['is_crisis'] & ~df['is_crisis'].shift(1, fill_value=False)
        df['crisis_end'] = df['is_crisis'].shift(-1, fill_value=False) & ~df['is_crisis'].shift(-1, fill_value=True)

        crises = []
        crisis_id = 1

        for idx, row in df[df['crisis_start']].iterrows():
            start_year = row['year']

            # Find end of this crisis
            crisis_period = df[(df['year'] >= start_year) & df['is_crisis']]
            if len(crisis_period) == 0:
                continue

            # Find when crisis ends
            end_idx = crisis_period.index[-1]
            end_year = df.loc[end_idx, 'year']

            duration = int(end_year - start_year + 1)

            if duration >= min_duration:
                # Calculate severity metrics
                crisis_data = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

                severity = abs(crisis_data[variable].min())
                cumulative_loss = abs(crisis_data[variable].sum())
                avg_growth = crisis_data[variable].mean()

                # Recovery time (years to return to pre-crisis level)
                pre_crisis_gdp = df[df['year'] == start_year - 1]['gdp'].values
                if len(pre_crisis_gdp) > 0:
                    pre_crisis_level = pre_crisis_gdp[0]

                    post_crisis = df[df['year'] > end_year]
                    recovery = post_crisis[post_crisis['gdp'] >= pre_crisis_level]

                    if len(recovery) > 0:
                        recovery_year = recovery.iloc[0]['year']
                        recovery_time = int(recovery_year - end_year)
                    else:
                        recovery_time = None
                else:
                    recovery_time = None

                crises.append({
                    'crisis_id': crisis_id,
                    'start_year': int(start_year),
                    'end_year': int(end_year),
                    'duration': duration,
                    'severity': severity,
                    'cumulative_loss': cumulative_loss,
                    'average_growth': avg_growth,
                    'recovery_time': recovery_time
                })

                crisis_id += 1

        return pd.DataFrame(crises)

    def measure_crisis_severity(self, crisis: Dict, variable: str = 'gdp') -> Dict:
        """
        Calculate multiple severity measures for a crisis.

        Parameters
        ----------
        crisis : Dict
            Crisis information (start_year, end_year)
        variable : str
            Variable to measure

        Returns
        -------
        Dict
            Multiple severity metrics
        """
        start = crisis['start_year']
        end = crisis['end_year']

        crisis_data = self.data[
            (self.data['year'] >= start) &
            (self.data['year'] <= end)
        ]

        if len(crisis_data) == 0:
            return {}

        # Peak-to-trough decline
        if 'gdp' in crisis_data.columns:
            peak = crisis_data['gdp'].max()
            trough = crisis_data['gdp'].min()
            peak_to_trough = (trough - peak) / peak
        else:
            peak_to_trough = None

        # Duration
        duration = end - start + 1

        # Output loss (cumulative growth shortfall)
        if 'gdp_growth' in crisis_data.columns:
            # Assume potential growth of 2.5%
            potential_growth = 0.025
            output_loss = sum(potential_growth - crisis_data['gdp_growth'])
        else:
            output_loss = None

        # Distributional impact
        if 'wage_share' in crisis_data.columns:
            pre_crisis = self.data[self.data['year'] == start - 1]
            if len(pre_crisis) > 0:
                wage_share_change = crisis_data['wage_share'].mean() - pre_crisis['wage_share'].values[0]
            else:
                wage_share_change = None
        else:
            wage_share_change = None

        # Financial impact
        if 'financialization' in crisis_data.columns:
            financial_stress = crisis_data['financialization'].std()
        else:
            financial_stress = None

        return {
            'start_year': start,
            'end_year': end,
            'duration': duration,
            'peak_to_trough_decline': peak_to_trough,
            'cumulative_output_loss': output_loss,
            'wage_share_impact': wage_share_change,
            'financial_stress': financial_stress
        }

    def analyze_crisis_clustering(self, crises: pd.DataFrame) -> Dict:
        """
        Analyze temporal clustering of crises.

        Tests whether crises cluster in certain periods or are randomly distributed.

        Parameters
        ----------
        crises : pd.DataFrame
            Detected crises

        Returns
        -------
        Dict
            Clustering statistics
        """
        if len(crises) < 2:
            return {'n_crises': len(crises), 'clustering': 'Insufficient data'}

        # Calculate inter-crisis intervals
        crises = crises.sort_values('start_year')
        intervals = []

        for i in range(len(crises) - 1):
            interval = crises.iloc[i+1]['start_year'] - crises.iloc[i]['end_year']
            intervals.append(interval)

        intervals = np.array(intervals)

        # Test for clustering using coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else 0

        # High CV suggests clustering (some short, some long intervals)
        # Low CV suggests regular spacing

        # Identify clusters (intervals < mean)
        short_intervals = intervals < mean_interval
        n_clusters = np.sum(short_intervals)

        return {
            'n_crises': len(crises),
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'coefficient_of_variation': cv,
            'clustering_interpretation': 'High clustering' if cv > 0.5 else 'Low clustering',
            'n_short_intervals': n_clusters,
            'intervals': intervals.tolist()
        }

    def crisis_frequency_by_period(self, crises: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate crisis frequency by historical period.

        Parameters
        ----------
        crises : pd.DataFrame
            Detected crises

        Returns
        -------
        pd.DataFrame
            Crisis frequency by period
        """
        # Define periods
        periods = [
            {'name': 'Pre-WWI', 'start': 1870, 'end': 1914},
            {'name': 'Interwar', 'start': 1914, 'end': 1945},
            {'name': 'Golden Age', 'start': 1945, 'end': 1973},
            {'name': 'Neoliberal', 'start': 1973, 'end': 2008},
            {'name': 'Post-GFC', 'start': 2008, 'end': 2020}
        ]

        results = []

        for period in periods:
            period_crises = crises[
                (crises['start_year'] >= period['start']) &
                (crises['start_year'] <= period['end'])
            ]

            period_years = period['end'] - period['start'] + 1
            n_crises = len(period_crises)
            frequency = n_crises / period_years

            if n_crises > 0:
                avg_severity = period_crises['severity'].mean()
                avg_duration = period_crises['duration'].mean()
            else:
                avg_severity = 0
                avg_duration = 0

            results.append({
                'period': period['name'],
                'start_year': period['start'],
                'end_year': period['end'],
                'years': period_years,
                'n_crises': n_crises,
                'frequency': frequency,
                'avg_severity': avg_severity,
                'avg_duration': avg_duration
            })

        return pd.DataFrame(results)

    def identify_systemic_crises(self, crises: pd.DataFrame) -> List[Dict]:
        """
        Identify systemic crises (major structural crises).

        Systemic crises are severe, long-lasting, and lead to institutional change.

        Parameters
        ----------
        crises : pd.DataFrame
            All detected crises

        Returns
        -------
        List[Dict]
            Systemic crises
        """
        if len(crises) == 0:
            return []

        # Criteria for systemic crisis:
        # 1. Duration > 3 years OR
        # 2. Severity > 75th percentile AND duration > 2 years

        severity_threshold = crises['severity'].quantile(0.75)

        systemic = []

        for _, crisis in crises.iterrows():
            is_systemic = False

            if crisis['duration'] > 3:
                is_systemic = True
                reason = 'Long duration'
            elif crisis['severity'] > severity_threshold and crisis['duration'] > 2:
                is_systemic = True
                reason = 'High severity and moderate duration'

            if is_systemic:
                systemic.append({
                    'start_year': crisis['start_year'],
                    'end_year': crisis['end_year'],
                    'duration': crisis['duration'],
                    'severity': crisis['severity'],
                    'systemic_reason': reason
                })

        return systemic


class HegemonyAnalyzer:
    """
    Analyze hegemonic cycles and transitions.

    Based on Giovanni Arrighi's framework:
    - Systemic cycles of accumulation
    - Hegemonic rise, maturity, decline
    - Material expansion vs financial expansion phases
    - Hegemonic transitions and interregna
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize hegemony analyzer.

        Parameters
        ----------
        data : pd.DataFrame
            Historical data with hegemony indicator
        """
        self.data = data

    @staticmethod
    def get_arrighi_cycles() -> List[Dict]:
        """
        Return Arrighi's systemic cycles of accumulation.

        Each cycle has:
        - Hegemonic power
        - Material expansion phase
        - Financial expansion phase (signal crisis)
        - Terminal crisis

        Returns
        -------
        List[Dict]
            Historical hegemonic cycles
        """
        cycles = [
            {
                'hegemon': 'Genoa',
                'material_expansion': (1340, 1560),
                'financial_expansion': (1560, 1640),
                'terminal_crisis': '1627-1654',
                'key_features': 'Trade-based, Mediterranean centered'
            },
            {
                'hegemon': 'Dutch Republic',
                'material_expansion': (1560, 1740),
                'financial_expansion': (1740, 1815),
                'terminal_crisis': '1780-1815',
                'key_features': 'Commercial capitalism, global trade networks'
            },
            {
                'hegemon': 'Britain',
                'material_expansion': (1780, 1870),
                'financial_expansion': (1870, 1945),
                'terminal_crisis': '1914-1945',
                'key_features': 'Industrial capitalism, colonial empire'
            },
            {
                'hegemon': 'United States',
                'material_expansion': (1870, 1970),
                'financial_expansion': (1970, None),  # Ongoing?
                'terminal_crisis': 'TBD',
                'key_features': 'Corporate capitalism, global military presence'
            }
        ]

        return cycles

    def detect_hegemonic_transitions(self,
                                     hegemony_var: str = 'hegemony',
                                     threshold: float = 0.1) -> List[Dict]:
        """
        Detect periods of hegemonic transition.

        Transitions are periods when hegemonic power index declines significantly.

        Parameters
        ----------
        hegemony_var : str
            Hegemony indicator variable
        threshold : float
            Decline threshold for transition detection

        Returns
        -------
        List[Dict]
            Detected transition periods
        """
        df = self.data[self.data[hegemony_var].notna()].copy()
        df = df.sort_values('year')

        # Calculate rolling change in hegemony
        window = 5
        df['hegemony_change'] = df[hegemony_var].rolling(window=window).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
        )

        # Identify periods of sustained decline
        df['declining'] = df['hegemony_change'] < -threshold

        # Find transition periods (consecutive declining years)
        transitions = []
        in_transition = False
        start_year = None

        for idx, row in df.iterrows():
            if row['declining'] and not in_transition:
                # Start of transition
                in_transition = True
                start_year = row['year']
            elif not row['declining'] and in_transition:
                # End of transition
                in_transition = False
                end_year = df.loc[idx - 1, 'year']

                # Get hegemony levels
                start_level = df[df['year'] == start_year][hegemony_var].values[0]
                end_level = df[df['year'] == end_year][hegemony_var].values[0]

                transitions.append({
                    'start_year': int(start_year),
                    'end_year': int(end_year),
                    'duration': int(end_year - start_year + 1),
                    'hegemony_start': start_level,
                    'hegemony_end': end_level,
                    'decline_rate': (end_level - start_level) / (end_year - start_year + 1)
                })

        return transitions

    def classify_accumulation_phase(self,
                                    financialization_var: str = 'financialization',
                                    threshold: float = 0.45) -> pd.DataFrame:
        """
        Classify periods as material expansion vs financial expansion.

        In Arrighi's framework:
        - Material expansion: productive investment dominates
        - Financial expansion: financial accumulation dominates (signal crisis)

        Parameters
        ----------
        financialization_var : str
            Financialization indicator
        threshold : float
            Threshold for financial expansion

        Returns
        -------
        pd.DataFrame
            Phase classification by year
        """
        df = self.data[['year', financialization_var]].dropna().copy()

        df['accumulation_phase'] = 'Material Expansion'
        df.loc[df[financialization_var] > threshold, 'accumulation_phase'] = 'Financial Expansion'

        # Identify phase transitions
        df['phase_change'] = df['accumulation_phase'] != df['accumulation_phase'].shift()

        return df

    def measure_hegemonic_strength(self,
                                   variables: List[str] = None) -> pd.DataFrame:
        """
        Construct composite hegemonic strength index.

        Components:
        - Economic dominance (GDP share)
        - Financial power (reserve currency status)
        - Military capacity
        - Institutional control

        Parameters
        ----------
        variables : List[str], optional
            Variables to include in index

        Returns
        -------
        pd.DataFrame
            Hegemonic strength index over time
        """
        if variables is None:
            # Use available proxy variables
            variables = ['hegemony']  # Use existing hegemony indicator

        df = self.data[['year'] + variables].dropna().copy()

        # Normalize variables to 0-1 scale
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        df['hegemonic_strength'] = scaler.fit_transform(
            df[variables].values
        ).mean(axis=1)

        return df[['year', 'hegemonic_strength']]

    def identify_interregnum_periods(self, transitions: List[Dict]) -> List[Dict]:
        """
        Identify interregnum periods (between hegemonies).

        Interregnum: period of hegemonic crisis with no clear successor.

        Parameters
        ----------
        transitions : List[Dict]
            Detected hegemonic transitions

        Returns
        -------
        List[Dict]
            Interregnum periods with characteristics
        """
        # In historical record, interregna are:
        # - 1914-1945: British decline, not yet full US hegemony
        # - Potentially ongoing since 2008?

        historical_interregna = [
            {
                'period': '1914-1945',
                'declining_hegemon': 'Britain',
                'rising_power': 'USA',
                'characteristics': [
                    'World wars',
                    'Great Depression',
                    'Competing economic systems',
                    'Institutional breakdown'
                ]
            },
            {
                'period': '2008-present?',
                'declining_hegemon': 'USA',
                'rising_power': 'China?',
                'characteristics': [
                    'Financial crisis',
                    'Declining US relative power',
                    'Rising multipolarity',
                    'Institutional strain'
                ]
            }
        ]

        return historical_interregna


def analyze_crisis_hegemony_relationship(data: pd.DataFrame) -> Dict:
    """
    Analyze relationship between crises and hegemonic cycles.

    Hypothesis: crises cluster during hegemonic transitions and
    financial expansion phases.

    Parameters
    ----------
    data : pd.DataFrame
        Historical data

    Returns
    -------
    Dict
        Analysis results
    """
    crisis_analyzer = CrisisAnalyzer(data)
    hegemony_analyzer = HegemonyAnalyzer(data)

    # Detect crises
    crises = crisis_analyzer.detect_crises()

    # Classify accumulation phases
    phases = hegemony_analyzer.classify_accumulation_phase()

    # Count crises by phase
    crisis_counts = {'Material Expansion': 0, 'Financial Expansion': 0}

    for _, crisis in crises.iterrows():
        crisis_year = crisis['start_year']

        phase_data = phases[phases['year'] == crisis_year]
        if len(phase_data) > 0:
            phase = phase_data.iloc[0]['accumulation_phase']
            crisis_counts[phase] = crisis_counts.get(phase, 0) + 1

    # Calculate crisis rates
    material_years = len(phases[phases['accumulation_phase'] == 'Material Expansion'])
    financial_years = len(phases[phases['accumulation_phase'] == 'Financial Expansion'])

    material_rate = crisis_counts['Material Expansion'] / material_years if material_years > 0 else 0
    financial_rate = crisis_counts['Financial Expansion'] / financial_years if financial_years > 0 else 0

    return {
        'total_crises': len(crises),
        'crises_in_material_expansion': crisis_counts['Material Expansion'],
        'crises_in_financial_expansion': crisis_counts['Financial Expansion'],
        'material_expansion_years': material_years,
        'financial_expansion_years': financial_years,
        'crisis_rate_material': material_rate,
        'crisis_rate_financial': financial_rate,
        'ratio_financial_to_material': financial_rate / material_rate if material_rate > 0 else None
    }


if __name__ == '__main__':
    print("Crisis and Hegemony Analysis module loaded successfully.")
    print("\nAvailable classes:")
    print("- CrisisAnalyzer: Detect and analyze economic crises")
    print("- HegemonyAnalyzer: Analyze hegemonic cycles and transitions")
    print("\nAvailable functions:")
    print("- analyze_crisis_hegemony_relationship: Test crisis-hegemony linkage")
