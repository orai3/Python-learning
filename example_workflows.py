"""
Example Analysis Workflows
Institutional Political Economy Research

This script demonstrates common research workflows using the
Political Economy Analysis System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from political_economy_analysis import (
    PoliticalEconomyAnalyzer,
    RegulationSchoolAnalyzer
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("INSTITUTIONAL POLITICAL ECONOMY ANALYSIS")
print("Example Research Workflows")
print("=" * 80)

# Load analyzer
analyzer = PoliticalEconomyAnalyzer(
    data_path='/home/user/Python-learning/political_economy_dataset.csv'
)
df = analyzer.df
reg_analyzer = RegulationSchoolAnalyzer(df)

print("\n" + "=" * 80)
print("WORKFLOW 1: REGIME TYPOLOGY VALIDATION")
print("Research Question: Do empirically-derived clusters match theoretical regimes?")
print("=" * 80)

# Run PCA
print("\n1.1 Running PCA on 2023 data...")
pca_2023 = analyzer.perform_pca(year=2023, n_components=3)

print(f"Explained variance: {pca_2023['explained_variance'][:3]}")
print(f"Cumulative: {pca_2023['cumulative_variance'][2]:.1%}")

print("\nTop 5 loadings on PC1:")
print(pca_2023['loadings']['PC1'].abs().sort_values(ascending=False).head(5))

# Run cluster analysis
print("\n1.2 Running K-means clustering (k=4)...")
cluster_2023 = analyzer.cluster_analysis(year=2023, method='kmeans', n_clusters=4)

print(f"Silhouette score: {cluster_2023['silhouette_score']:.3f}")
print(f"Calinski-Harabasz score: {cluster_2023['calinski_harabasz_score']:.1f}")

# Compare clusters to theoretical regimes
print("\n1.3 Cluster composition by theoretical regime:")
comparison = pd.crosstab(
    cluster_2023['cluster_assignments']['regime_type'],
    cluster_2023['cluster_assignments']['cluster']
)
print(comparison)

print("\n1.4 Countries by cluster:")
for cluster_id in sorted(cluster_2023['cluster_assignments']['cluster'].unique()):
    countries = cluster_2023['cluster_assignments'][
        cluster_2023['cluster_assignments']['cluster'] == cluster_id
    ]['country'].tolist()
    print(f"\nCluster {cluster_id} ({len(countries)} countries):")
    print("  " + ", ".join(countries))

print("\n" + "=" * 80)
print("WORKFLOW 2: NEOLIBERAL CONVERGENCE TEST")
print("Research Question: Are all regimes converging toward neoliberalism?")
print("=" * 80)

print("\n2.1 Neoliberalism by regime over time...")

# Calculate variance over time
yearly_variance = []
years_sample = range(1974, 2024, 10)  # Every 10 years

for year in years_sample:
    year_data = df[df['year'] == year]
    variance = year_data['neoliberalism_index'].var()
    yearly_variance.append(variance)
    print(f"{year}: Variance = {variance:.4f}")

if yearly_variance[0] > yearly_variance[-1]:
    print("\n→ CONVERGENCE detected (variance decreasing)")
else:
    print("\n→ DIVERGENCE or stability (variance not decreasing)")

# Regime-specific trends
print("\n2.2 Average neoliberalism by regime:")
regime_trends = df.groupby(['year', 'regime_type'])['neoliberalism_index'].mean().unstack()

print("\n1980 vs 2023 comparison:")
for regime in ['LME', 'CME', 'MME']:
    if regime in regime_trends.columns:
        change = regime_trends.loc[2023, regime] - regime_trends.loc[1980, regime]
        print(f"{regime}: {regime_trends.loc[1980, regime]:.3f} → "
              f"{regime_trends.loc[2023, regime]:.3f} (Δ = {change:+.3f})")

print("\n" + "=" * 80)
print("WORKFLOW 3: INSTITUTIONAL COMPLEMENTARITIES")
print("Research Question: Are VoC-predicted complementarities empirically robust?")
print("=" * 80)

print("\n3.1 Testing key complementarities (2023)...")
comp_results = analyzer.test_complementarities(year=2023)

print("\nStrongest complementarities (overall correlation):")
top_comp = comp_results.nlargest(5, 'overall_correlation')
for idx, row in top_comp.iterrows():
    print(f"  {row['institution_1']} ↔ {row['institution_2']}: "
          f"r = {row['overall_correlation']:.3f} (p = {row['p_value']:.4f})")

print("\n3.2 Regime-specific complementarities:")
print("\nLabor coordination ↔ Vocational training by regime:")
labor_voc = comp_results[
    (comp_results['institution_1'] == 'labor_market_coordination') &
    (comp_results['institution_2'] == 'vocational_training')
].iloc[0]

for regime in ['LME', 'CME', 'MME']:
    col_name = f'corr_{regime}'
    if col_name in labor_voc.index:
        print(f"  {regime}: r = {labor_voc[col_name]:.3f}")

print("\n→ CME should show strongest complementarity (Hall & Soskice prediction)")

print("\n" + "=" * 80)
print("WORKFLOW 4: FINANCIALIZATION AND LABOR SHARE")
print("Research Question: Does financialization reduce labor's share?")
print("=" * 80)

print("\n4.1 Correlation analysis...")

# Overall correlation
corr, pval = np.corrcoef(
    df['financialization_index'],
    df['wage_share_gdp']
)[0, 1], 0.0  # Simplified

from scipy.stats import pearsonr
corr, pval = pearsonr(df['financialization_index'].dropna(),
                      df['wage_share_gdp'].dropna())

print(f"Overall correlation: r = {corr:.3f} (p < {pval:.4f})")

# By period
print("\n4.2 Correlation by period:")
for period_name, years in [
    ('Pre-neoliberal (1974-1979)', (1974, 1979)),
    ('Neoliberal era (1980-2007)', (1980, 2007)),
    ('Post-crisis (2008-2023)', (2008, 2023))
]:
    period_data = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]
    r, p = pearsonr(period_data['financialization_index'].dropna(),
                   period_data['wage_share_gdp'].dropna())
    print(f"{period_name}: r = {r:.3f}")

# By regime type
print("\n4.3 Correlation by regime type:")
for regime in ['LME', 'CME', 'EAsia']:
    regime_data = df[df['regime_type'] == regime]
    if len(regime_data) > 10:
        r, p = pearsonr(regime_data['financialization_index'].dropna(),
                       regime_data['wage_share_gdp'].dropna())
        print(f"{regime}: r = {r:.3f}")

print("\n→ Expect negative correlation: higher financialization → lower wage share")

print("\n" + "=" * 80)
print("WORKFLOW 5: POWER RESOURCES AND WELFARE")
print("Research Question: Does left power predict welfare in neoliberal era?")
print("=" * 80)

print("\n5.1 Correlation in different periods...")

for period_name, years in [
    ('Golden Age (1974-1979)', (1974, 1979)),
    ('Neoliberal (1980-2007)', (1980, 2007)),
    ('Post-crisis (2008-2023)', (2008, 2023))
]:
    period_data = df[(df['year'] >= years[0]) & (df['year'] <= years[1])]
    r, p = pearsonr(period_data['power_resources_index'].dropna(),
                   period_data['welfare_generosity'].dropna())
    print(f"{period_name}: r = {r:.3f} (p = {p:.4f})")

print("\n→ If r declining: Power resources theory weakening in neoliberal era")
print("→ If r stable: Power resources still matter despite neoliberalism")

print("\n5.2 Mean welfare generosity by power resources quartile (2023):")
data_2023 = df[df['year'] == 2023].copy()
data_2023['pr_quartile'] = pd.qcut(data_2023['power_resources_index'],
                                    q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
welfare_by_pr = data_2023.groupby('pr_quartile')['welfare_generosity'].mean()
print(welfare_by_pr)

print("\n" + "=" * 80)
print("WORKFLOW 6: CRITICAL JUNCTURES")
print("Research Question: Identify and characterize regime transitions")
print("=" * 80)

print("\n6.1 Detecting transitions (threshold = 15%)...")
transitions = analyzer.detect_regime_transitions(threshold=0.15)

if len(transitions) > 0:
    print(f"\nDetected {len(transitions)} regime transitions\n")

    print("Transitions by type:")
    print(transitions['direction'].value_counts())

    print("\nSample transitions:")
    sample = transitions.nlargest(5, 'magnitude')
    for idx, row in sample.iterrows():
        print(f"\n{row['country']} ({row['year']}):")
        print(f"  Type: {row['direction']}")
        print(f"  Magnitude: {row['magnitude']:.3f}")
        print(f"  Neoliberalism change: {row['neoliberalism_change']:+.3f}")
else:
    print("\nNo major transitions detected with current threshold")
    print("(Synthetic data designed for institutional stability)")

print("\n6.2 Case study: USA trajectory")
usa_traj = analyzer.trajectory_analysis(
    'USA',
    indicators=['neoliberalism_index', 'financialization_index',
               'power_resources_index', 'wage_share_gdp']
)

print("\nUSA institutional trends (1974-2023):")
for indicator, trend in usa_traj['trends'].items():
    print(f"\n{indicator}:")
    print(f"  Direction: {trend['direction']}")
    print(f"  R²: {trend['r_squared']:.3f}")
    print(f"  Strength: {trend['strength']:.3f}")

print("\n→ Expect: ↑neoliberalism, ↑financialization, ↓power resources, ↓wage share")

print("\n" + "=" * 80)
print("WORKFLOW 7: EXPORT-LED vs DEBT-LED GROWTH")
print("Research Question: Which post-Fordist model is more stable/egalitarian?")
print("=" * 80)

print("\n7.1 Classifying growth models (2023)...")
reg_scores = reg_analyzer.regime_of_accumulation_scores(year=2023)

# Classify
reg_scores['dominant_model'] = reg_scores[
    ['fordist_score', 'finance_led_score', 'export_led_score']
].idxmax(axis=1)

print("\nGrowth model distribution:")
print(reg_scores['dominant_model'].value_counts())

# Compare characteristics
print("\n7.2 Comparing growth models:")

for model in ['fordist_score', 'finance_led_score', 'export_led_score']:
    model_name = model.replace('_score', '').title()

    # Top 5 countries in this model
    top5 = reg_scores.nlargest(5, model)

    print(f"\n{model_name} - Top 5 countries:")
    for idx, row in top5.iterrows():
        print(f"  {row['country']}: {row[model]:.3f}")

# Compare wage share and household debt
print("\n7.3 Model characteristics (mean values, 2023):")
data_2023_full = df[df['year'] == 2023].copy()

# Merge with growth model classification
merged = data_2023_full.merge(
    reg_scores[['country', 'dominant_model']],
    on='country'
)

characteristics = merged.groupby('dominant_model').agg({
    'wage_share_gdp': 'mean',
    'household_debt_to_income': 'mean',
    'neoliberalism_index': 'mean',
    'welfare_generosity': 'mean'
}).round(3)

print(characteristics)

print("\n→ Expected patterns:")
print("  - Export-led: Higher wage share, lower household debt")
print("  - Finance-led: Lower wage share, higher household debt")
print("  - Fordist: Highest wage share, lowest debt")

print("\n" + "=" * 80)
print("WORKFLOW 8: REGIME COHERENCE ANALYSIS")
print("Research Question: Which countries have coherent institutional configurations?")
print("=" * 80)

print("\n8.1 Calculating coherence scores (2023)...")
coherence = analyzer.regime_coherence_score(year=2023)

print("\nMost coherent regimes:")
print(coherence.head(10)[['country', 'regime_type', 'coherence_score']])

print("\nLeast coherent regimes (hybrids):")
print(coherence.tail(10)[['country', 'regime_type', 'coherence_score']])

print("\n8.2 Coherence by regime type:")
coherence_by_type = coherence.groupby('regime_type')['coherence_score'].agg(['mean', 'std'])
print(coherence_by_type.sort_values('mean', ascending=False))

print("\n→ High coherence = 'pure' ideal-type")
print("→ Low coherence = hybrid/transitional configuration")
print("→ CME and LME should show highest coherence")

print("\n" + "=" * 80)
print("WORKFLOW 9: PATH DEPENDENCE ANALYSIS")
print("Research Question: How sticky are institutional configurations?")
print("=" * 80)

print("\n9.1 Measuring path dependence (autocorrelation)...")

# Sample countries from different regimes
sample_countries = ['USA', 'Germany', 'France', 'Poland', 'South Korea']

path_dep_results = []

for country in sample_countries:
    traj = analyzer.trajectory_analysis(
        country,
        indicators=['neoliberalism_index', 'cme_complementarity', 'power_resources_index']
    )

    country_result = {
        'country': country,
        'regime': traj['regime']
    }

    for indicator, autocorr in traj['path_dependence'].items():
        country_result[indicator] = autocorr

    path_dep_results.append(country_result)

path_dep_df = pd.DataFrame(path_dep_results)
print(path_dep_df.round(3))

print("\n→ High autocorrelation (>0.9) = strong path dependence")
print("→ Low autocorrelation (<0.7) = institutional volatility")
print("→ CME expected to show highest path dependence")

print("\n" + "=" * 80)
print("WORKFLOW 10: CROSS-NATIONAL COMPARISON")
print("Research Question: Compare institutional profiles of key countries")
print("=" * 80)

print("\n10.1 Institutional profiles (2023)...")

comparison_countries = ['USA', 'Germany', 'Sweden', 'United Kingdom',
                       'France', 'Japan', 'China']

indicators_to_compare = [
    'labor_market_coordination',
    'union_density',
    'welfare_generosity',
    'financialization_index',
    'neoliberalism_index',
    'power_resources_index',
    'wage_share_gdp'
]

comparison_data = df[
    (df['year'] == 2023) &
    (df['country'].isin(comparison_countries))
][['country', 'regime_type'] + indicators_to_compare]

comparison_data = comparison_data.set_index('country')
print(comparison_data.round(3))

print("\n10.2 Normalized comparison (Z-scores):")
# Standardize for easier comparison
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
comparison_scaled = comparison_data[indicators_to_compare].copy()
comparison_scaled[indicators_to_compare] = scaler.fit_transform(
    comparison_scaled[indicators_to_compare]
)
comparison_scaled['regime_type'] = comparison_data['regime_type']

print(comparison_scaled.round(2))

print("\n→ Positive Z-score = above average")
print("→ Negative Z-score = below average")
print("→ Shows relative institutional positions")

print("\n" + "=" * 80)
print("ANALYSIS WORKFLOWS COMPLETE")
print("=" * 80)

print("\nNext steps for academic research:")
print("1. Visualize key findings (use GUI or matplotlib)")
print("2. Run regression analyses for causal inference")
print("3. Compare to real-world data sources (OECD, ICTWSS)")
print("4. Develop theoretical interpretations")
print("5. Write up findings for publication")

print("\nSuggested publication outlets:")
print("- Comparative Political Studies")
print("- Socio-Economic Review")
print("- Review of International Political Economy")
print("- New Political Economy")
print("- Politics & Society")
print("- British Journal of Political Science")

print("\n" + "=" * 80)
