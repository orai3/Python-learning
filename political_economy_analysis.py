"""
Institutional Political Economy Analysis Module

Core analytical tools for comparative political economy research:
- Principal Component Analysis for institutional dimensions
- Cluster analysis for regime identification
- Complementarity testing between institutions
- Historical trajectory analysis (path dependence)
- Regime transition detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster import hierarchy
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class PoliticalEconomyAnalyzer:
    """
    Main analysis class for institutional political economy research
    """

    def __init__(self, data_path=None, df=None):
        """
        Initialize analyzer with dataset

        Args:
            data_path: Path to CSV file
            df: Or directly provide DataFrame
        """
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either data_path or df")

        self.scaler = StandardScaler()
        self.pca_model = None
        self.cluster_labels = None

        # Define indicator groups for analysis
        self.indicator_groups = {
            'voc': [
                'labor_market_coordination',
                'union_density',
                'employment_protection',
                'vocational_training',
                'stakeholder_governance',
                'interfirm_cooperation'
            ],
            'power_resources': [
                'left_cabinet_share',
                'welfare_generosity',
                'strike_days_per_1000',
                'decommodification_index',
                'union_density'
            ],
            'financialization': [
                'financial_sector_gdp_share',
                'household_debt_to_income',
                'corporate_debt_to_gdp',
                'stock_market_cap_to_gdp',
                'nfc_financial_income_ratio',
                'financial_deregulation',
                'financialization_index'
            ],
            'neoliberalism': [
                'trade_openness',
                'capital_account_openness',
                'privatization_index',
                'labor_market_flexibility',
                'top_tax_rate',
                'social_spending_gdp',
                'neoliberalism_index'
            ],
            'institutional': [
                'wage_bargaining_level',
                'almp_spending_gdp',
                'product_market_regulation',
                'wage_share_gdp',
                'policy_orthodoxy'
            ]
        }

        # All quantitative indicators for PCA
        self.all_indicators = list(set([
            ind for group in self.indicator_groups.values() for ind in group
        ]))

        print(f"Loaded dataset: {len(self.df)} observations")
        print(f"Countries: {self.df['country'].nunique()}")
        print(f"Time period: {self.df['year'].min()}-{self.df['year'].max()}")
        print(f"Indicators: {len(self.all_indicators)}")

    def perform_pca(self, indicators=None, n_components=None, year=None):
        """
        Perform Principal Component Analysis

        Args:
            indicators: List of indicators to include (default: all)
            n_components: Number of components (default: auto based on variance)
            year: Specific year for cross-sectional analysis (default: all years)

        Returns:
            Dictionary with PCA results
        """
        if indicators is None:
            indicators = self.all_indicators

        # Filter data
        if year:
            data = self.df[self.df['year'] == year].copy()
        else:
            data = self.df.copy()

        # Prepare data matrix
        X = data[indicators].values
        country_year = data[['country', 'year']].values

        # Handle missing values
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Determine optimal number of components
        if n_components is None:
            # Use components explaining 90% variance
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.90) + 1
            n_components = max(n_components, 2)  # At least 2

        # Perform PCA
        self.pca_model = PCA(n_components=n_components)
        principal_components = self.pca_model.fit_transform(X_scaled)

        # Create results dataframe
        pc_cols = [f'PC{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(
            principal_components,
            columns=pc_cols
        )
        pca_df['country'] = country_year[:, 0]
        pca_df['year'] = country_year[:, 1].astype(int)
        pca_df = pd.concat([pca_df, data[['regime_type']].reset_index(drop=True)], axis=1)

        # Component loadings
        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            columns=pc_cols,
            index=indicators
        )

        results = {
            'pca_scores': pca_df,
            'loadings': loadings,
            'explained_variance': self.pca_model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca_model.explained_variance_ratio_),
            'n_components': n_components,
            'indicators': indicators
        }

        return results

    def cluster_analysis(self, data_source='pca', n_clusters=None, method='kmeans',
                        year=None, indicators=None):
        """
        Perform cluster analysis for regime identification

        Args:
            data_source: 'pca' (use PCA scores) or 'raw' (use raw indicators)
            n_clusters: Number of clusters (auto-detect if None)
            method: 'kmeans', 'hierarchical', or 'dbscan'
            year: Specific year for analysis
            indicators: List of indicators if data_source='raw'

        Returns:
            Dictionary with clustering results
        """
        if data_source == 'pca':
            # Use PCA scores
            if self.pca_model is None:
                pca_results = self.perform_pca(year=year)
            else:
                pca_results = self.perform_pca(year=year)

            X = pca_results['pca_scores'][[c for c in pca_results['pca_scores'].columns if c.startswith('PC')]].values
            metadata = pca_results['pca_scores'][['country', 'year', 'regime_type']]

        else:  # raw indicators
            if indicators is None:
                indicators = self.all_indicators

            if year:
                data = self.df[self.df['year'] == year].copy()
            else:
                data = self.df.copy()

            X = data[indicators].values
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
            X = self.scaler.fit_transform(X)
            metadata = data[['country', 'year', 'regime_type']]

        # Determine optimal number of clusters
        if n_clusters is None and method != 'dbscan':
            n_clusters = self._optimal_clusters(X, max_k=10)

        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            labels = clusterer.fit_predict(X)
            centers = clusterer.cluster_centers_

        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            labels = clusterer.fit_predict(X)
            centers = None  # Computed separately for hierarchical

        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(X)
            centers = None
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Cluster quality metrics
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
        else:
            silhouette = None
            calinski = None

        # Create results dataframe
        results_df = metadata.copy()
        results_df['cluster'] = labels

        # Cluster profiles (mean values)
        if data_source == 'pca':
            profile_data = pca_results['pca_scores'].copy()
            profile_data['cluster'] = labels
            cluster_profiles = profile_data.groupby('cluster')[[c for c in profile_data.columns if c.startswith('PC')]].mean()
        else:
            profile_data = pd.DataFrame(X, columns=indicators)
            profile_data['cluster'] = labels
            cluster_profiles = profile_data.groupby('cluster').mean()

        results = {
            'labels': labels,
            'n_clusters': n_clusters,
            'method': method,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'cluster_assignments': results_df,
            'cluster_profiles': cluster_profiles,
            'centers': centers
        }

        return results

    def _optimal_clusters(self, X, max_k=10):
        """
        Determine optimal number of clusters using elbow method and silhouette

        Args:
            X: Data matrix
            max_k: Maximum clusters to test

        Returns:
            Optimal number of clusters
        """
        inertias = []
        silhouettes = []

        for k in range(2, min(max_k + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))

        # Use silhouette score
        optimal_k = np.argmax(silhouettes) + 2

        return optimal_k

    def test_complementarities(self, institution_pairs=None, year=None):
        """
        Test for institutional complementarities (Hall & Soskice)

        Complementarity exists when institutions co-vary and enhance each other's effectiveness

        Args:
            institution_pairs: List of tuples of institution pairs to test
            year: Specific year (default: pooled across all years)

        Returns:
            DataFrame with correlation results
        """
        if institution_pairs is None:
            # Test key VoC complementarities
            institution_pairs = [
                ('labor_market_coordination', 'vocational_training'),
                ('labor_market_coordination', 'stakeholder_governance'),
                ('vocational_training', 'stakeholder_governance'),
                ('stakeholder_governance', 'interfirm_cooperation'),
                ('labor_market_flexibility', 'stock_market_cap_to_gdp'),
                ('union_density', 'welfare_generosity'),
                ('wage_bargaining_level', 'employment_protection'),
            ]

        if year:
            data = self.df[self.df['year'] == year].copy()
        else:
            data = self.df.copy()

        results = []

        for inst1, inst2 in institution_pairs:
            # Overall correlation
            r, p = pearsonr(data[inst1].dropna(), data[inst2].dropna())

            # By regime type
            regime_corrs = {}
            for regime in data['regime_type'].unique():
                regime_data = data[data['regime_type'] == regime]
                if len(regime_data) > 3:
                    r_regime, _ = pearsonr(
                        regime_data[inst1].dropna(),
                        regime_data[inst2].dropna()
                    )
                    regime_corrs[regime] = r_regime

            # Test if correlation differs by regime (complementarity strength varies)
            complementarity_strength = np.std(list(regime_corrs.values()))

            results.append({
                'institution_1': inst1,
                'institution_2': inst2,
                'overall_correlation': r,
                'p_value': p,
                'complementarity_strength': complementarity_strength,
                **{f'corr_{regime}': regime_corrs.get(regime, np.nan)
                   for regime in ['LME', 'CME', 'MME', 'Transition', 'EAsia', 'LatAm']}
            })

        return pd.DataFrame(results)

    def trajectory_analysis(self, country, indicators=None):
        """
        Analyze historical trajectory for a country (path dependence)

        Args:
            country: Country name
            indicators: List of indicators to analyze

        Returns:
            Dictionary with trajectory analysis
        """
        if indicators is None:
            indicators = ['neoliberalism_index', 'financialization_index',
                         'power_resources_index', 'wage_share_gdp']

        country_data = self.df[self.df['country'] == country].sort_values('year')

        if len(country_data) == 0:
            raise ValueError(f"Country '{country}' not found in dataset")

        trajectories = {}
        trends = {}
        structural_breaks = {}

        for indicator in indicators:
            values = country_data[indicator].values
            years = country_data['year'].values

            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)

            # Test for structural break (Chow test approximation)
            # Test break at midpoint
            mid_point = len(values) // 2
            pre_slope, pre_int, *_ = stats.linregress(years[:mid_point], values[:mid_point])
            post_slope, post_int, *_ = stats.linregress(years[mid_point:], values[mid_point:])

            # Significant break if slopes differ substantially
            slope_change = abs(post_slope - pre_slope)
            structural_break = slope_change > abs(slope) * 0.5

            trajectories[indicator] = {
                'values': values,
                'years': years
            }

            trends[indicator] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'strength': abs(r_value)
            }

            structural_breaks[indicator] = {
                'detected': structural_break,
                'pre_slope': pre_slope,
                'post_slope': post_slope,
                'change': slope_change
            }

        # Path dependence measure: autocorrelation
        path_dependence = {}
        for indicator in indicators:
            values = country_data[indicator].values
            if len(values) > 1:
                # Lag-1 autocorrelation
                autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
                path_dependence[indicator] = autocorr

        results = {
            'country': country,
            'regime': country_data['regime_type'].iloc[0],
            'trajectories': trajectories,
            'trends': trends,
            'structural_breaks': structural_breaks,
            'path_dependence': path_dependence,
            'data': country_data
        }

        return results

    def detect_regime_transitions(self, country=None, threshold=0.15):
        """
        Detect regime transitions based on significant institutional changes

        A transition is detected when multiple key indicators change significantly

        Args:
            country: Specific country (or None for all countries)
            threshold: Minimum change in composite indices to count as transition

        Returns:
            DataFrame with detected transitions
        """
        if country:
            data = self.df[self.df['country'] == country].copy()
        else:
            data = self.df.copy()

        transitions = []

        for ctry in data['country'].unique():
            country_data = data[data['country'] == ctry].sort_values('year')

            # Track key composite indices
            indices = ['neoliberalism_index', 'financialization_index',
                      'power_resources_index']

            for i in range(1, len(country_data)):
                changes = {}
                for idx in indices:
                    change = country_data.iloc[i][idx] - country_data.iloc[i-1][idx]
                    changes[idx] = change

                # Check if significant change occurred
                avg_change = np.mean([abs(c) for c in changes.values()])

                if avg_change > threshold:
                    transitions.append({
                        'country': ctry,
                        'year': country_data.iloc[i]['year'],
                        'regime_type': country_data.iloc[i]['regime_type'],
                        'neoliberalism_change': changes['neoliberalism_index'],
                        'financialization_change': changes['financialization_index'],
                        'power_resources_change': changes['power_resources_index'],
                        'magnitude': avg_change,
                        'direction': self._classify_transition(changes)
                    })

        return pd.DataFrame(transitions)

    def _classify_transition(self, changes):
        """
        Classify type of regime transition based on changes

        Args:
            changes: Dictionary of index changes

        Returns:
            Transition type string
        """
        neol = changes['neoliberalism_index']
        fin = changes['financialization_index']
        power = changes['power_resources_index']

        if neol > 0 and fin > 0 and power < 0:
            return 'neoliberal_turn'
        elif neol < 0 and power > 0:
            return 'social_democratic_shift'
        elif fin > 0:
            return 'financialization_wave'
        elif neol < 0 and fin < 0:
            return 're_regulation'
        else:
            return 'mixed'

    def regime_coherence_score(self, year=None):
        """
        Calculate coherence score for each country's institutional configuration

        High coherence = institutions align with regime type ideal-type
        Low coherence = hybrid/incoherent configuration

        Args:
            year: Specific year for analysis (default: latest year)

        Returns:
            DataFrame with coherence scores
        """
        if year is None:
            year = self.df['year'].max()

        data = self.df[self.df['year'] == year].copy()

        coherence_scores = []

        for idx, row in data.iterrows():
            regime = row['regime_type']
            country = row['country']

            # Calculate distance from regime ideal-type
            if regime == 'LME':
                # LME ideal: high flexibility, low coordination, high financialization
                ideal_score = (
                    row['labor_market_flexibility'] * 0.25 +
                    (1 - row['labor_market_coordination']) * 0.25 +
                    row['financialization_index'] * 0.25 +
                    np.clip(row['stock_market_cap_to_gdp'] / 2, 0, 1) * 0.25
                )
            elif regime == 'CME':
                # CME ideal: high coordination, high vocational training, high stakeholder gov
                ideal_score = (
                    row['labor_market_coordination'] * 0.30 +
                    row['vocational_training'] * 0.25 +
                    row['stakeholder_governance'] * 0.25 +
                    row['union_density'] * 0.20
                )
            else:
                # For other regimes, use general coherence measure
                # Consistency between related institutions
                ideal_score = row['cme_complementarity'] * 0.5 + row['lme_complementarity'] * 0.5

            coherence_scores.append({
                'country': country,
                'regime_type': regime,
                'coherence_score': ideal_score,
                'year': year
            })

        return pd.DataFrame(coherence_scores).sort_values('coherence_score', ascending=False)


class RegulationSchoolAnalyzer:
    """
    Specialized analyzer for Regulation School periodization
    (Fordism, Post-Fordism, Neoliberalism)
    """

    def __init__(self, df):
        self.df = df

    def identify_period(self, year):
        """
        Classify year into regulation school period

        Args:
            year: Year to classify

        Returns:
            Period label
        """
        if year < 1973:
            return 'Fordism/Golden Age'
        elif 1973 <= year < 1980:
            return 'Crisis of Fordism'
        elif 1980 <= year < 2008:
            return 'Neoliberal/Finance-led'
        else:
            return 'Post-Crisis'

    def regime_of_accumulation_scores(self, year=None):
        """
        Calculate regime of accumulation characteristics

        Args:
            year: Specific year (default: all years)

        Returns:
            DataFrame with regime scores
        """
        if year:
            data = self.df[self.df['year'] == year].copy()
        else:
            data = self.df.copy()

        data['period'] = data['year'].apply(self.identify_period)

        # Fordist regime characteristics
        data['fordist_score'] = (
            data['wage_share_gdp'] * 0.4 +  # High wage share
            data['union_density'] * 0.3 +  # Strong unions
            (1 - data['financialization_index']) * 0.3  # Low financialization
        )

        # Finance-led regime
        data['finance_led_score'] = (
            data['financialization_index'] * 0.5 +
            (1 - data['wage_share_gdp']) * 0.25 +
            data['household_debt_to_income'] / 2 * 0.25
        ).clip(0, 1)

        # Export-led regime
        data['export_led_score'] = (
            data['trade_openness'] * 0.5 +
            (data['wage_share_gdp'] < 0.60).astype(float) * 0.25 +  # Wage suppression
            data['stakeholder_governance'] * 0.25
        )

        return data[['country', 'year', 'period', 'fordist_score',
                    'finance_led_score', 'export_led_score', 'regime_type']]


if __name__ == "__main__":
    print("=" * 80)
    print("POLITICAL ECONOMY ANALYSIS MODULE")
    print("=" * 80)
    print("\nLoading dataset...")

    analyzer = PoliticalEconomyAnalyzer(
        data_path='/home/user/Python-learning/political_economy_dataset.csv'
    )

    print("\n" + "=" * 80)
    print("EXAMPLE ANALYSES")
    print("=" * 80)

    # 1. PCA Analysis
    print("\n1. PRINCIPAL COMPONENT ANALYSIS (2023)")
    print("-" * 80)
    pca_results = analyzer.perform_pca(year=2023)
    print(f"Number of components: {pca_results['n_components']}")
    print(f"Explained variance: {pca_results['explained_variance'][:3]}")
    print(f"Cumulative variance: {pca_results['cumulative_variance'][:3]}")
    print("\nTop loadings on PC1:")
    print(pca_results['loadings']['PC1'].abs().sort_values(ascending=False).head(5))

    # 2. Cluster Analysis
    print("\n2. CLUSTER ANALYSIS (K-means, 2023)")
    print("-" * 80)
    cluster_results = analyzer.cluster_analysis(year=2023, method='kmeans')
    print(f"Number of clusters: {cluster_results['n_clusters']}")
    print(f"Silhouette score: {cluster_results['silhouette_score']:.3f}")
    print("\nCluster sizes:")
    print(cluster_results['cluster_assignments']['cluster'].value_counts().sort_index())

    # 3. Complementarity Testing
    print("\n3. INSTITUTIONAL COMPLEMENTARITIES")
    print("-" * 80)
    comp_results = analyzer.test_complementarities(year=2023)
    print("Strongest complementarities:")
    print(comp_results.nlargest(3, 'overall_correlation')[
        ['institution_1', 'institution_2', 'overall_correlation', 'complementarity_strength']
    ])

    # 4. Trajectory Analysis
    print("\n4. HISTORICAL TRAJECTORY ANALYSIS (USA)")
    print("-" * 80)
    traj_usa = analyzer.trajectory_analysis('USA')
    print("Trends:")
    for ind, trend in traj_usa['trends'].items():
        print(f"  {ind}: {trend['direction']} (R²={trend['r_squared']:.3f})")

    # 5. Regime Transitions
    print("\n5. REGIME TRANSITIONS (threshold=0.15)")
    print("-" * 80)
    transitions = analyzer.detect_regime_transitions(threshold=0.15)
    print(f"Total transitions detected: {len(transitions)}")
    if len(transitions) > 0:
        print("\nTransitions by type:")
        print(transitions['direction'].value_counts())

    # 6. Regime Coherence
    print("\n6. REGIME COHERENCE SCORES (2023)")
    print("-" * 80)
    coherence = analyzer.regime_coherence_score(year=2023)
    print("Most coherent regimes:")
    print(coherence.head(5)[['country', 'regime_type', 'coherence_score']])

    # 7. Regulation School
    print("\n7. REGULATION SCHOOL ANALYSIS")
    print("-" * 80)
    reg_analyzer = RegulationSchoolAnalyzer(analyzer.df)
    reg_scores = reg_analyzer.regime_of_accumulation_scores(year=2023)
    print("Regime of accumulation scores (sample):")
    print(reg_scores[reg_scores['country'].isin(['USA', 'Germany', 'China'])][
        ['country', 'fordist_score', 'finance_led_score', 'export_led_score']
    ])

    print("\n" + "=" * 80)
    print("Analysis module ready for use!")
    print("=" * 80)
