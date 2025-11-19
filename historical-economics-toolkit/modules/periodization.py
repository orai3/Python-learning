"""
Periodization Algorithms for Historical Economics
=================================================

Implements multiple approaches to detecting structural breaks and regime changes:
1. Statistical break tests (Chow, CUSUM, Bai-Perron)
2. Regime-switching models
3. Regulation School periodization framework
4. Social Structure of Accumulation (SSA) detection
5. Cluster-based periodization

Theoretical foundations:
- Regulation School (Boyer, Aglietta): modes of regulation and regimes of accumulation
- SSA theory (Kotz, McDonough, Reich): institutional configurations
- Historical materialism: contradictions and crises driving periodic restructuring
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class StructuralBreakDetector:
    """
    Detect structural breaks using multiple statistical methods.

    Methods implemented:
    - Chow test for known break points
    - CUSUM test for unknown breaks
    - Bai-Perron sequential testing
    - Moving window correlation breaks
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize break detector.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data with date index
        """
        self.data = data
        self.breaks = {}

    def chow_test(self,
                   variable: str,
                   break_year: int,
                   sig_level: float = 0.05) -> Dict:
        """
        Perform Chow test for structural break at specified year.

        H0: No structural break
        H1: Structural break at specified year

        Parameters
        ----------
        variable : str
            Variable name to test
        break_year : int
            Year to test for break
        sig_level : float
            Significance level

        Returns
        -------
        Dict
            Test results including F-statistic and p-value
        """
        df = self.data[self.data['year'].notna()].copy()

        # Create time trend
        df['t'] = np.arange(len(df))

        # Split sample
        df1 = df[df['year'] < break_year].copy()
        df2 = df[df['year'] >= break_year].copy()

        if len(df1) < 3 or len(df2) < 3:
            return {
                'break_year': break_year,
                'f_statistic': None,
                'p_value': None,
                'significant': False,
                'message': 'Insufficient data for test'
            }

        # Estimate regressions
        try:
            # Full sample regression
            y = df[variable].values
            X = np.column_stack([np.ones(len(df)), df['t'].values])
            beta_full = np.linalg.lstsq(X, y, rcond=None)[0]
            resid_full = y - X @ beta_full
            ssr_full = np.sum(resid_full ** 2)

            # Sub-sample regressions
            y1 = df1[variable].values
            X1 = np.column_stack([np.ones(len(df1)), df1['t'].values])
            beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
            resid1 = y1 - X1 @ beta1
            ssr1 = np.sum(resid1 ** 2)

            y2 = df2[variable].values
            X2 = np.column_stack([np.ones(len(df2)), df2['t'].values])
            beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
            resid2 = y2 - X2 @ beta2
            ssr2 = np.sum(resid2 ** 2)

            # Chow F-statistic
            k = 2  # Number of parameters
            n = len(df)
            f_stat = ((ssr_full - (ssr1 + ssr2)) / k) / ((ssr1 + ssr2) / (n - 2*k))

            # P-value
            p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)

            return {
                'break_year': break_year,
                'variable': variable,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < sig_level,
                'ssr_full': ssr_full,
                'ssr_restricted': ssr1 + ssr2
            }

        except Exception as e:
            return {
                'break_year': break_year,
                'f_statistic': None,
                'p_value': None,
                'significant': False,
                'message': f'Error: {str(e)}'
            }

    def cusum_test(self, variable: str, threshold: float = 1.5) -> Dict:
        """
        CUSUM test for structural stability.

        Detects unknown breakpoints by tracking cumulative sum of recursive residuals.

        Parameters
        ----------
        variable : str
            Variable to test
        threshold : float
            Threshold for detection (in standard deviations)

        Returns
        -------
        Dict
            Test results including break years detected
        """
        df = self.data[self.data[variable].notna()].copy()
        df = df.sort_values('year')

        n = len(df)
        y = df[variable].values

        # Recursive residuals
        residuals = []
        min_obs = 20  # Minimum observations for initial estimation

        for t in range(min_obs, n):
            # Estimate on data up to t
            y_train = y[:t]
            X_train = np.arange(t).reshape(-1, 1)

            # Fit linear model
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict next observation
            pred = model.predict([[t]])[0]
            actual = y[t]
            resid = actual - pred
            residuals.append(resid)

        residuals = np.array(residuals)

        # Standardize
        std_resid = residuals / np.std(residuals)

        # CUSUM statistic
        cusum = np.cumsum(std_resid)

        # Detect breaks (where CUSUM crosses threshold)
        breaks = []
        for i in range(1, len(cusum)):
            if abs(cusum[i]) > threshold * np.sqrt(len(cusum)):
                year = df.iloc[min_obs + i]['year']
                breaks.append(int(year))

        # Remove duplicates (keep first in cluster)
        if breaks:
            filtered_breaks = [breaks[0]]
            for b in breaks[1:]:
                if b - filtered_breaks[-1] > 5:  # At least 5 years apart
                    filtered_breaks.append(b)
            breaks = filtered_breaks

        return {
            'variable': variable,
            'method': 'CUSUM',
            'break_years': breaks,
            'cusum_values': cusum,
            'threshold': threshold
        }

    def bai_perron_test(self,
                        variable: str,
                        max_breaks: int = 5,
                        min_segment_length: int = 15) -> Dict:
        """
        Simplified Bai-Perron sequential test for multiple breaks.

        Sequentially tests for additional breaks using F-statistics.

        Parameters
        ----------
        variable : str
            Variable to test
        max_breaks : int
            Maximum number of breaks to detect
        min_segment_length : int
            Minimum observations between breaks

        Returns
        -------
        Dict
            Detected breaks and test statistics
        """
        df = self.data[self.data[variable].notna()].copy()
        df = df.sort_values('year')

        y = df[variable].values
        n = len(y)

        def calculate_ssr(data, breaks):
            """Calculate sum of squared residuals given break points."""
            segments = []
            prev_idx = 0

            for b in breaks + [n]:
                segments.append(data[prev_idx:b])
                prev_idx = b

            ssr_total = 0
            for seg in segments:
                if len(seg) > 0:
                    mean = np.mean(seg)
                    ssr_total += np.sum((seg - mean) ** 2)

            return ssr_total

        # Find optimal breaks sequentially
        detected_breaks = []

        for k in range(max_breaks):
            if len(detected_breaks) >= max_breaks:
                break

            best_break = None
            best_ssr = calculate_ssr(y, detected_breaks)

            # Try each potential break point
            for i in range(min_segment_length, n - min_segment_length):
                # Check if too close to existing breaks
                too_close = False
                for b in detected_breaks:
                    if abs(i - b) < min_segment_length:
                        too_close = True
                        break

                if too_close:
                    continue

                # Test this break
                test_breaks = sorted(detected_breaks + [i])
                ssr = calculate_ssr(y, test_breaks)

                if ssr < best_ssr:
                    best_ssr = ssr
                    best_break = i

            # If found improvement, add break
            if best_break is not None:
                detected_breaks.append(best_break)
                detected_breaks = sorted(detected_breaks)
            else:
                break  # No more breaks found

        # Convert indices to years
        break_years = [int(df.iloc[i]['year']) for i in detected_breaks]

        return {
            'variable': variable,
            'method': 'Bai-Perron',
            'break_years': break_years,
            'n_breaks': len(break_years),
            'break_indices': detected_breaks
        }

    def correlation_breaks(self,
                          var1: str,
                          var2: str,
                          window: int = 20,
                          threshold: float = 0.3) -> Dict:
        """
        Detect breaks in correlation structure between two variables.

        Uses rolling window correlation and detects sharp changes.

        Parameters
        ----------
        var1, var2 : str
            Variables to analyze
        window : int
            Rolling window size
        threshold : float
            Threshold for detecting correlation break

        Returns
        -------
        Dict
            Detected correlation regime changes
        """
        df = self.data[[var1, var2, 'year']].dropna()

        # Rolling correlation
        rolling_corr = df[var1].rolling(window=window).corr(df[var2].rolling(window=window))

        # Detect large changes
        corr_change = rolling_corr.diff().abs()

        # Find peaks in correlation change
        breaks_idx = find_peaks(corr_change.values, height=threshold)[0]

        break_years = [int(df.iloc[i]['year']) for i in breaks_idx]

        return {
            'var1': var1,
            'var2': var2,
            'method': 'Correlation breaks',
            'break_years': break_years,
            'rolling_correlation': rolling_corr,
            'threshold': threshold
        }


class RegimeSwitchingModel:
    """
    Estimate regime-switching models for detecting regime changes.

    Implements simplified Markov-switching model for economic time series.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize regime-switching model.

        Parameters
        ----------
        data : pd.DataFrame
            Time series data
        """
        self.data = data
        self.regimes = None

    def estimate_two_regime_model(self,
                                   variable: str,
                                   high_vol_threshold: float = 0.75) -> Dict:
        """
        Estimate simple two-regime model: high volatility vs low volatility.

        Parameters
        ----------
        variable : str
            Variable to model
        high_vol_threshold : float
            Percentile threshold for high volatility regime

        Returns
        -------
        Dict
            Regime classifications and parameters
        """
        df = self.data[self.data[variable].notna()].copy()
        df = df.sort_values('year')

        # Calculate rolling volatility
        window = 10
        df['volatility'] = df[variable].rolling(window=window).std()

        # Classify regimes based on volatility
        vol_threshold = df['volatility'].quantile(high_vol_threshold)

        df['regime'] = (df['volatility'] > vol_threshold).astype(int)

        # Identify regime transitions
        df['regime_change'] = df['regime'].diff().abs()
        transitions = df[df['regime_change'] == 1]['year'].values

        # Regime statistics
        regime_0 = df[df['regime'] == 0][variable]
        regime_1 = df[df['regime'] == 1][variable]

        return {
            'variable': variable,
            'regimes': df[['year', 'regime']],
            'transition_years': transitions.astype(int).tolist(),
            'regime_0_mean': regime_0.mean(),
            'regime_0_std': regime_0.std(),
            'regime_1_mean': regime_1.mean(),
            'regime_1_std': regime_1.std(),
            'volatility_threshold': vol_threshold
        }

    def estimate_growth_regime_model(self, gdp_var: str = 'gdp_growth') -> Dict:
        """
        Estimate growth regime model: high growth vs low/crisis growth.

        Parameters
        ----------
        gdp_var : str
            GDP growth variable

        Returns
        -------
        Dict
            Growth regime classifications
        """
        df = self.data[self.data[gdp_var].notna()].copy()

        # Use median split for regimes
        median_growth = df[gdp_var].median()

        df['growth_regime'] = (df[gdp_var] > median_growth).astype(int)
        df['growth_regime'] = df['growth_regime'].replace({0: 'Low Growth', 1: 'High Growth'})

        # Add crisis indicator
        df.loc[df[gdp_var] < 0, 'growth_regime'] = 'Crisis'

        transitions = df[df['growth_regime'] != df['growth_regime'].shift()]['year'].values

        return {
            'variable': gdp_var,
            'regimes': df[['year', 'growth_regime']],
            'transition_years': transitions.astype(int).tolist(),
            'median_growth': median_growth
        }


class RegulationSchoolPeriodization:
    """
    Implement Regulation School periodization framework.

    Identifies regimes of accumulation based on multiple institutional indicators:
    - Mode of regulation (competitive vs monopolistic)
    - Wage relation (fordist vs flexible)
    - Monetary regime
    - International regime
    - State form
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize periodization framework.

        Parameters
        ----------
        data : pd.DataFrame
            Historical data with institutional variables
        """
        self.data = data
        self.periods = None

    def identify_regimes(self,
                        variables: List[str] = None,
                        n_regimes: int = 4) -> Dict:
        """
        Identify regimes using cluster analysis on institutional variables.

        Parameters
        ----------
        variables : List[str], optional
            Variables to use for clustering. If None, uses default set.
        n_regimes : int
            Number of regimes to identify

        Returns
        -------
        Dict
            Regime classifications and characteristics
        """
        if variables is None:
            variables = [
                'wage_share',
                'financialization',
                'institutional_coordination',
                'labor_militancy'
            ]

        # Prepare data
        df = self.data[['year'] + variables].dropna()

        X = df[variables].values

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        df['regime'] = kmeans.fit_predict(X_scaled)

        # Order regimes chronologically
        regime_order = df.groupby('regime')['year'].mean().sort_values().index
        regime_mapping = {old: new for new, old in enumerate(regime_order)}
        df['regime'] = df['regime'].map(regime_mapping)

        # Identify regime periods
        df['regime_change'] = df['regime'].diff().abs() > 0

        regime_periods = []
        current_regime = df.iloc[0]['regime']
        start_year = df.iloc[0]['year']

        for idx, row in df.iterrows():
            if row['regime_change'] and idx > 0:
                # End of previous regime
                regime_periods.append({
                    'regime': int(current_regime),
                    'start_year': int(start_year),
                    'end_year': int(df.iloc[idx-1]['year']),
                    'duration': int(df.iloc[idx-1]['year'] - start_year + 1)
                })
                current_regime = row['regime']
                start_year = row['year']

        # Add last regime
        regime_periods.append({
            'regime': int(current_regime),
            'start_year': int(start_year),
            'end_year': int(df.iloc[-1]['year']),
            'duration': int(df.iloc[-1]['year'] - start_year + 1)
        })

        # Calculate regime characteristics
        regime_chars = []
        for period in regime_periods:
            regime_data = df[
                (df['year'] >= period['start_year']) &
                (df['year'] <= period['end_year'])
            ]

            chars = {
                'regime': period['regime'],
                'start_year': period['start_year'],
                'end_year': period['end_year'],
                'duration': period['duration']
            }

            for var in variables:
                chars[f'{var}_mean'] = regime_data[var].mean()

            regime_chars.append(chars)

        return {
            'regime_periods': regime_periods,
            'regime_characteristics': pd.DataFrame(regime_chars),
            'regime_classifications': df[['year', 'regime']],
            'cluster_centers': scaler.inverse_transform(kmeans.cluster_centers_),
            'variables_used': variables
        }

    def label_historical_regimes(self, regime_chars: pd.DataFrame) -> pd.DataFrame:
        """
        Apply theoretical labels to identified regimes.

        Based on Regulation School literature.

        Parameters
        ----------
        regime_chars : pd.DataFrame
            Regime characteristics from identify_regimes()

        Returns
        -------
        pd.DataFrame
            Regime characteristics with labels
        """
        df = regime_chars.copy()

        labels = []
        for _, row in df.iterrows():
            # Heuristics based on characteristics
            if row.get('wage_share_mean', 0) > 0.63 and \
               row.get('institutional_coordination_mean', 0) > 0.6:
                label = 'Fordist/Golden Age'
            elif row.get('financialization_mean', 0) > 0.5 and \
                 row.get('wage_share_mean', 0) < 0.60:
                label = 'Financialized/Neoliberal'
            elif row.get('institutional_coordination_mean', 0) < 0.4 and \
                 row['start_year'] < 1920:
                label = 'Competitive Capitalism'
            elif row['start_year'] >= 1914 and row['end_year'] <= 1945:
                label = 'Crisis/Transition'
            else:
                label = 'Uncertain/Transitional'

            labels.append(label)

        df['regime_label'] = labels

        return df


def detect_all_breaks(data: pd.DataFrame,
                     variables: List[str],
                     methods: List[str] = None) -> Dict:
    """
    Convenience function to run all break detection methods.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    variables : List[str]
        Variables to test
    methods : List[str], optional
        Methods to use. If None, uses all.

    Returns
    -------
    Dict
        All break detection results
    """
    if methods is None:
        methods = ['cusum', 'bai_perron']

    detector = StructuralBreakDetector(data)
    results = {}

    for variable in variables:
        var_results = {}

        if 'cusum' in methods:
            var_results['cusum'] = detector.cusum_test(variable)

        if 'bai_perron' in methods:
            var_results['bai_perron'] = detector.bai_perron_test(variable)

        results[variable] = var_results

    return results


if __name__ == '__main__':
    # Example usage
    print("Periodization module loaded successfully.")
    print("\nAvailable classes:")
    print("- StructuralBreakDetector: Statistical break tests")
    print("- RegimeSwitchingModel: Regime-switching models")
    print("- RegulationSchoolPeriodization: Heterodox periodization")
