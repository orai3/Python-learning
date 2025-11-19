# Institutional Political Economy Quantitative Analysis System

## Documentation

**Author:** Heterodox Economics Research Tools
**Version:** 1.0
**Date:** 2025

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Foundations](#theoretical-foundations)
3. [Measurement Frameworks](#measurement-frameworks)
4. [Statistical Methods](#statistical-methods)
5. [Data Structure](#data-structure)
6. [Usage Guide](#usage-guide)
7. [Academic Applications](#academic-applications)
8. [References](#references)

---

## 1. Introduction

This system provides comprehensive quantitative tools for institutional political economy research, integrating multiple theoretical traditions:

- **Varieties of Capitalism (VoC)** - Hall & Soskice (2001)
- **Power Resources Theory (PRT)** - Korpi (1983), Esping-Andersen (1990)
- **Financialization Theory** - Epstein (2005), Krippner (2011)
- **Neoliberalism Studies** - Harvey (2005), Crouch (2011)
- **Regulation School** - Boyer & Saillard (2002), Aglietta (1979)

### Key Features

- **40 countries** across 8 regime types
- **50 years** of institutional data (1974-2023)
- **35+ indicators** covering labor, finance, welfare, trade
- **Advanced statistical methods**: PCA, clustering, time series
- **Interactive PyQt6 GUI** for academic research
- **Theory-grounded synthetic data** with realistic dynamics

### Research Applications

- Cross-national regime comparison
- Historical trajectory analysis
- Institutional complementarity testing
- Regime transition detection
- Periodization (Regulation School)
- Quantitative typologies

---

## 2. Theoretical Foundations

### 2.1 Varieties of Capitalism (VoC)

**Core Argument:**
Developed economies organize into distinct institutional configurations based on how firms coordinate with:
- Labor (wage bargaining, skills)
- Finance (corporate governance)
- Other firms (supply chains, technology transfer)

**Ideal Types:**

**Liberal Market Economies (LME)**
- Decentralized wage bargaining
- Flexible labor markets
- Shareholder-oriented corporate governance
- Competitive inter-firm relations
- General skills via market-based education
- **Examples:** USA, UK, Canada, Australia

**Coordinated Market Economies (CME)**
- Centralized/sectoral wage bargaining
- Employment protection
- Stakeholder corporate governance
- Cooperative inter-firm relations
- Industry-specific vocational training
- **Examples:** Germany, Sweden, Denmark, Austria

**Key Hypothesis: Institutional Complementarities**
Institutions within each regime type reinforce each other. For example:
- CME: Vocational training ↔ Long-term employment ↔ Patient capital ↔ Incremental innovation
- LME: General education ↔ Flexible employment ↔ Stock market finance ↔ Radical innovation

### 2.2 Power Resources Theory (PRT)

**Core Argument:**
The distribution of power resources (especially labor organization and left political parties) determines:
- Welfare state generosity
- Labor market regulation
- Income distribution
- Decommodification (independence from market)

**Key Indicators:**
- Union density and coverage
- Left party cabinet share
- Strike activity
- Welfare spending and generosity
- Labor market decommodification

**Causal Mechanism:**
Working class organization → Political representation → Redistributive policies → Decommodification

**Welfare Regimes (Esping-Andersen):**
- **Social Democratic:** High decommodification (Nordic countries)
- **Conservative:** Moderate, status-preserving (Continental Europe)
- **Liberal:** Minimal, means-tested (Anglophone countries)

### 2.3 Financialization

**Definition:**
"Increasing role of financial motives, financial markets, financial actors, and financial institutions in the operation of domestic and international economies" (Epstein, 2005)

**Dimensions:**

1. **Financial Sector Growth**
   - Finance as % of GDP
   - Employment in finance
   - Profits accruing to finance

2. **Household Financialization**
   - Debt-to-income ratios
   - Asset-based welfare
   - Housing financialization

3. **Non-Financial Corporate (NFC) Financialization**
   - "Shareholder value orientation"
   - Financial income of NFCs
   - Short-termism

4. **Regulatory Environment**
   - Financial deregulation
   - Capital account liberalization
   - Derivatives markets

**Consequences:**
- Rising inequality
- Wage stagnation despite productivity growth
- Financial instability
- Investment decline in NFCs

### 2.4 Neoliberalism

**Core Features (Harvey, 2005):**
- Market fundamentalism
- Deregulation (labor, product, financial markets)
- Privatization
- Trade and capital account liberalization
- Retrenchment of welfare state
- Tax cuts (especially top marginal rates)
- Weakening of labor power

**Measurement:**
We construct a composite index capturing:
- Trade openness
- Capital account openness
- Privatization
- Labor market flexibility
- Tax progressivity (inverse)
- Social spending (inverse)

**Historical Periodization:**
- **Pre-1980:** Embedded liberalism / Fordism
- **1980-2008:** Neoliberal ascendancy
- **Post-2008:** Crisis and contested re-regulation

### 2.5 Regulation School

**Core Concepts:**

**Regime of Accumulation:**
Stable macroeconomic pattern of production and consumption ensuring growth

**Mode of Regulation:**
Institutional framework (labor relations, finance, state) supporting accumulation regime

**Historical Regimes:**

1. **Fordism (1945-1973)**
   - Mass production + mass consumption
   - Collective bargaining + wage growth
   - Keynesian macroeconomic management
   - High wage share, low financialization

2. **Crisis of Fordism (1973-1980)**
   - Profit squeeze
   - Stagflation
   - Institutional breakdown

3. **Finance-Led Accumulation (1980-2008)**
   - Consumption financed by debt, not wages
   - Asset price inflation
   - Wage suppression
   - Shareholder value

4. **Post-Crisis (2008+)**
   - Unresolved contradictions
   - Austerity vs stimulus debates
   - Rising inequality

**Alternative Models:**
- **Export-Led:** Wage suppression + trade surplus (Germany, Japan)
- **Debt-Led:** Rising household debt + consumption (USA, UK, Spain)

---

## 3. Measurement Frameworks

### 3.1 Varieties of Capitalism Indicators

| Indicator | Description | Range | VoC Prediction |
|-----------|-------------|-------|----------------|
| `labor_market_coordination` | Centralization of wage bargaining | 0-1 | CME high, LME low |
| `union_density` | % workforce unionized | 0-1 | CME high, LME low |
| `employment_protection` | Strictness of dismissal regulations | 0-1 | CME high, LME low |
| `vocational_training` | Strength of industry-specific training | 0-1 | CME high, LME low |
| `stakeholder_governance` | Stakeholder vs shareholder orientation | 0-1 | CME high, LME low |
| `interfirm_cooperation` | Cooperative vs competitive relations | 0-1 | CME high, LME low |

**Data Sources (Real-world equivalents):**
- OECD Employment Database
- ICTWSS Database (Amsterdam Institute)
- Comparative Political Data Set (Armingeon et al.)

### 3.2 Power Resources Theory Metrics

| Indicator | Description | Theoretical Role |
|-----------|-------------|------------------|
| `left_cabinet_share` | % cabinet seats held by left parties | Political power |
| `union_density` | Union membership rate | Organizational power |
| `strike_days_per_1000` | Days lost to strikes per 1000 workers | Mobilization capacity |
| `welfare_generosity` | Replacement rates, coverage | Policy outcome |
| `decommodification_index` | Independence from market (Esping-Andersen) | Ultimate outcome |

**Decommodification Calculation:**
```
decommodification = 0.4 × welfare_generosity +
                   0.3 × union_density +
                   0.3 × employment_protection
```

### 3.3 Financialization Indices

| Indicator | Description | Significance |
|-----------|-------------|--------------|
| `financial_sector_gdp_share` | Finance value-added / GDP | Sectoral shift |
| `household_debt_to_income` | Household debt / disposable income | Household finance |
| `corporate_debt_to_gdp` | Non-financial corporate debt / GDP | Corporate leverage |
| `stock_market_cap_to_gdp` | Market capitalization / GDP | Capital markets |
| `nfc_financial_income_ratio` | Financial income / operating surplus | NFC financialization |
| `financial_deregulation` | Index of regulatory restrictions | Policy environment |

**Composite Financialization Index:**
```
financialization = 0.25 × (fin_sector_gdp / 0.15) +
                  0.20 × (hh_debt / 1.5) +
                  0.20 × (stock_market / 1.0) +
                  0.20 × (nfc_fin_income / 0.2) +
                  0.15 × fin_deregulation
```

### 3.4 Neoliberalism Scores

| Indicator | Description | Neoliberal Direction |
|-----------|-------------|---------------------|
| `trade_openness` | (Exports + Imports) / GDP | High = neoliberal |
| `capital_account_openness` | Chinn-Ito index | Open = neoliberal |
| `privatization_index` | Cumulative privatization | High = neoliberal |
| `labor_market_flexibility` | 1 - (protection + coordination) | High = neoliberal |
| `top_tax_rate` | Top marginal income tax rate | Low = neoliberal |
| `social_spending_gdp` | Social expenditure % GDP | Low = neoliberal |

### 3.5 Regulation School Scores

**Fordist Score:**
```
fordist = 0.4 × wage_share +
         0.3 × union_density +
         0.3 × (1 - financialization)
```

**Finance-Led Score:**
```
finance_led = 0.5 × financialization +
             0.25 × (1 - wage_share) +
             0.25 × household_debt
```

**Export-Led Score:**
```
export_led = 0.5 × trade_openness +
            0.25 × wage_suppression +
            0.25 × stakeholder_governance
```

---

## 4. Statistical Methods

### 4.1 Principal Component Analysis (PCA)

**Purpose:**
Reduce dimensionality of institutional indicators to identify underlying patterns

**Method:**
1. Standardize all indicators (mean=0, SD=1)
2. Compute covariance matrix
3. Extract eigenvectors (principal components)
4. Retain components explaining 90% variance

**Interpretation:**
- **PC1** typically captures overall "coordination" vs "marketization"
- **PC2** often captures "welfare generosity" dimension
- **Loadings** show which institutions contribute to each component

**Academic Use:**
- Visualize institutional space
- Identify hybrid regimes
- Track movement over time

**Implementation:**
```python
analyzer = PoliticalEconomyAnalyzer(data_path='dataset.csv')
pca_results = analyzer.perform_pca(year=2023, n_components=3)

# Biplot: countries in PC space + institutional loadings
loadings = pca_results['loadings']
scores = pca_results['pca_scores']
```

### 4.2 Cluster Analysis

**Purpose:**
Empirically identify regime types without theoretical priors

**Methods:**

**K-Means Clustering:**
- Partitional method
- Minimizes within-cluster variance
- Requires specifying K (number of clusters)

**Hierarchical Clustering:**
- Agglomerative (bottom-up)
- Ward linkage (minimize variance)
- Produces dendrogram

**Optimal K Selection:**
- Silhouette score (cohesion vs separation)
- Elbow method (within-cluster sum of squares)
- Theory-guided (e.g., VoC predicts 2-3 clusters)

**Validation:**
- Compare to theoretical regimes (VoC, Esping-Andersen)
- Check stability across years
- Examine cluster profiles

**Implementation:**
```python
cluster_results = analyzer.cluster_analysis(
    year=2023,
    method='kmeans',
    n_clusters=4,
    data_source='pca'  # Cluster on PCA scores
)

# Cluster profiles show mean values for each group
profiles = cluster_results['cluster_profiles']
```

### 4.3 Institutional Complementarity Testing

**Theory (Hall & Soskice):**
Institutions are complementary when they:
1. Co-vary empirically
2. Enhance each other's effectiveness
3. Differ in strength across regime types

**Method:**
1. Compute pairwise correlations
2. Test if correlation differs by regime
3. High variance in regime-specific correlations → strong complementarity

**Example Complementarities:**
- Labor coordination ↔ Vocational training (CME)
- Labor flexibility ↔ Stock market finance (LME)
- Union density ↔ Welfare generosity (Social Democratic)

**Statistical Test:**
```python
comp_results = analyzer.test_complementarities(year=2023)

# Look for:
# - High overall correlation
# - High complementarity_strength (variance across regimes)
# - Theoretical plausibility
```

**Academic Interpretation:**
- Supports VoC institutional complementarity hypothesis
- Identifies which institutions form coherent bundles
- Explains path dependence and resistance to reform

### 4.4 Trajectory Analysis (Path Dependence)

**Purpose:**
Analyze historical evolution to test path dependence claims

**Metrics:**

**1. Linear Trend**
```
Y = β₀ + β₁(time) + ε
```
- β₁ = rate of change
- R² = strength of trend
- Direction: increasing/decreasing

**2. Structural Break Detection**
```
Pre-break:  Y = α₀ + α₁(time) + ε
Post-break: Y = β₀ + β₁(time) + ε
Test: |β₁ - α₁| > threshold
```
Common breaks:
- 1980 (neoliberal turn)
- 2008 (financial crisis)

**3. Path Dependence (Autocorrelation)**
```
corr(Yₜ, Yₜ₋₁)
```
High autocorrelation → institutional stickiness

**Implementation:**
```python
traj = analyzer.trajectory_analysis(
    country='USA',
    indicators=['neoliberalism_index', 'wage_share_gdp']
)

# Examine:
# - trends['slope']: rate of change
# - structural_breaks['detected']: critical junctures
# - path_dependence: institutional persistence
```

### 4.5 Regime Transition Detection

**Definition:**
Significant, rapid change in institutional configuration

**Detection Algorithm:**
1. Calculate year-to-year changes in composite indices
2. Flag transitions when average change > threshold (e.g., 0.15)
3. Classify transition type based on directional changes

**Transition Types:**
- **Neoliberal turn:** ↑neoliberalism, ↑financialization, ↓power resources
- **Social democratic shift:** ↓neoliberalism, ↑power resources
- **Financialization wave:** ↑financialization
- **Re-regulation:** ↓neoliberalism, ↓financialization

**Implementation:**
```python
transitions = analyzer.detect_regime_transitions(
    country='United Kingdom',
    threshold=0.15
)

# Typical findings:
# - UK: Neoliberal turn ~1979-1982 (Thatcher)
# - Nordic: Smaller shifts due to institutional stability
# - Transition economies: Large shifts post-1990
```

### 4.6 Regime Coherence Score

**Concept:**
How well does a country's institutional configuration match its regime type ideal?

**Calculation:**

For **LME:**
```
coherence = 0.25 × flexibility +
           0.25 × (1 - coordination) +
           0.25 × financialization +
           0.25 × stock_market
```

For **CME:**
```
coherence = 0.30 × coordination +
           0.25 × vocational_training +
           0.25 × stakeholder_governance +
           0.20 × union_density
```

**Interpretation:**
- **High coherence:** "Pure" ideal-type
- **Low coherence:** Hybrid or transitional
- **Use:** Identify incoherent configurations under reform pressure

---

## 5. Data Structure

### Dataset Schema

**File:** `political_economy_dataset.csv`
**Observations:** 2,000 (40 countries × 50 years)
**Period:** 1974-2023 (annual)

### Column Structure

**Identifiers (5 columns):**
- `country`: Country name
- `year`: Year (1974-2023)
- `regime_type`: VoC/CPE regime (LME, CME, MME, etc.)
- `region`: Geographic region
- `income_level`: World Bank classification

**VoC Indicators (6 columns):**
- `labor_market_coordination`
- `union_density`
- `employment_protection`
- `vocational_training`
- `stakeholder_governance`
- `interfirm_cooperation`

**Power Resources (5 columns):**
- `left_cabinet_share`
- `welfare_generosity`
- `strike_days_per_1000`
- `decommodification_index`

**Financialization (7 columns):**
- `financial_sector_gdp_share`
- `household_debt_to_income`
- `corporate_debt_to_gdp`
- `stock_market_cap_to_gdp`
- `nfc_financial_income_ratio`
- `financial_deregulation`
- `financialization_index`

**Neoliberalism (8 columns):**
- `trade_openness`
- `capital_account_openness`
- `privatization_index`
- `labor_market_flexibility`
- `top_tax_rate`
- `social_spending_gdp`
- `neoliberalism_index`

**Additional Institutional (5 columns):**
- `wage_bargaining_level`
- `almp_spending_gdp`
- `product_market_regulation`
- `wage_share_gdp`
- `policy_orthodoxy`

**Composite Indices (7 columns):**
- `cme_complementarity`
- `lme_complementarity`
- `power_resources_index`
- `golden_age_score`
- `neoliberal_regime_score`

### Regime Type Distribution

| Regime | N Countries | Examples |
|--------|-------------|----------|
| **LME** | 6 | USA, UK, Canada, Australia, Ireland, New Zealand |
| **CME** | 9 | Germany, Sweden, Denmark, Norway, Finland, Netherlands, Austria, Belgium, Switzerland |
| **MME** | 4 | Italy, Spain, Portugal, Greece |
| **Transition** | 5 | Poland, Czech Republic, Hungary, Estonia, Slovenia |
| **EAsia** | 4 | Japan, South Korea, Taiwan, Singapore |
| **LatAm** | 5 | Brazil, Argentina, Chile, Mexico, Colombia |
| **Statist** | 2 | France, China |
| **Developing** | 5 | India, South Africa, Turkey, Indonesia, Thailand |

---

## 6. Usage Guide

### 6.1 Installation

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib scikit-learn PyQt6 seaborn statsmodels

# Files needed:
# - political_economy_data_generator.py
# - political_economy_analysis.py
# - political_economy_app.py
```

### 6.2 Generate Dataset

```python
python political_economy_data_generator.py
# Output: political_economy_dataset.csv
```

### 6.3 Command-Line Analysis

```python
from political_economy_analysis import PoliticalEconomyAnalyzer

# Load data
analyzer = PoliticalEconomyAnalyzer(
    data_path='political_economy_dataset.csv'
)

# PCA
pca_results = analyzer.perform_pca(year=2023, n_components=3)
print(pca_results['explained_variance'])

# Clustering
cluster_results = analyzer.cluster_analysis(
    year=2023,
    method='kmeans',
    n_clusters=4
)
print(cluster_results['cluster_assignments'])

# Complementarities
comp = analyzer.test_complementarities(year=2023)
print(comp.nlargest(5, 'overall_correlation'))

# Trajectories
traj = analyzer.trajectory_analysis(
    country='Germany',
    indicators=['neoliberalism_index', 'power_resources_index']
)
print(traj['trends'])

# Transitions
trans = analyzer.detect_regime_transitions(threshold=0.15)
print(trans[['country', 'year', 'direction']])
```

### 6.4 GUI Application

```bash
python political_economy_app.py
```

**Tabs:**
1. **Overview:** Cross-sectional and time series plots
2. **PCA Analysis:** Biplots, scree plots, loadings
3. **Cluster Analysis:** Scatter plots, dendrograms, profiles
4. **Complementarities:** Correlation matrices, test results
5. **Historical Trajectories:** Time series, trend analysis
6. **Regime Transitions:** Detection and classification
7. **Regulation School:** Accumulation regime scores

### 6.5 Exporting Results

All visualizations in the GUI can be saved via the matplotlib toolbar.

For programmatic export:

```python
# Save PCA results
pca_results['pca_scores'].to_csv('pca_scores_2023.csv')
pca_results['loadings'].to_csv('pca_loadings.csv')

# Save cluster assignments
cluster_results['cluster_assignments'].to_csv('clusters_2023.csv')

# Save complementarity tests
comp.to_csv('complementarity_results.csv')
```

---

## 7. Academic Applications

### 7.1 Typology Construction

**Research Question:**
Do empirically-derived clusters match theoretical regimes?

**Method:**
1. Run cluster analysis on 2023 data
2. Compare cluster membership to regime_type
3. Examine cluster profiles

**Expected Findings:**
- 3-4 clusters emerge
- Roughly correspond to LME/CME/MME
- Some hybrids (France, Japan) ambiguous

**Publication Outlet:**
*Comparative Political Studies*, *Socio-Economic Review*

### 7.2 Neoliberal Convergence Hypothesis

**Research Question:**
Are all regimes converging toward neoliberal model?

**Method:**
1. Plot `neoliberalism_index` over time by regime
2. Test for convergence: σ²(2023) < σ²(1980)?
3. Trajectory analysis for each country

**Competing Hypotheses:**
- **Convergence:** Falling variance, upward trends for all
- **Divergence/Stability:** Persistent differences

**Publication Outlet:**
*World Politics*, *British Journal of Political Science*

### 7.3 Institutional Complementarities

**Research Question:**
Are VoC-predicted complementarities empirically robust?

**Method:**
1. Test correlations between:
   - Labor coordination ↔ Vocational training
   - Flexibility ↔ Stock market finance
2. Compare strength across regimes
3. Time series: Are complementarities stable or weakening?

**Expected Findings:**
- Strong complementarities in "pure" LME/CME
- Weaker in hybrids
- Possible weakening over time (neoliberal pressure)

**Publication Outlet:**
*Socio-Economic Review*, *Organization Studies*

### 7.4 Financialization and Inequality

**Research Question:**
Does financialization reduce labor's share?

**Method:**
1. Panel regression: `wage_share ~ financialization_index + controls`
2. Trajectory analysis: Co-movement
3. Test if effect varies by regime type

**Theoretical Basis:**
Post-Keynesian/Marxian: Financialization empowers capital, weakens labor

**Publication Outlet:**
*Review of Political Economy*, *Cambridge Journal of Economics*

### 7.5 Power Resources and Welfare

**Research Question:**
Does left power still predict welfare generosity in neoliberal era?

**Method:**
1. Compare pre-1980 vs post-1980
2. Regression: `welfare_generosity ~ power_resources_index + period`
3. Interaction term: Has effect weakened?

**Expected:**
Weakening but still significant relationship

**Publication Outlet:**
*European Journal of Political Research*, *Politics & Society*

### 7.6 Critical Junctures

**Research Question:**
Identify and classify regime transitions

**Method:**
1. Run transition detection
2. Case studies of major transitions (UK 1979, Chile 1973, Germany 1990s)
3. Classify: Exogenous shock vs endogenous crisis?

**Theoretical Framework:**
Historical institutionalism (Pierson, Mahoney)

**Publication Outlet:**
*Comparative Politics*, *Studies in Comparative International Development*

### 7.7 Export-Led vs Debt-Led Growth Models

**Research Question:**
Which post-Fordist model is more stable/egalitarian?

**Method:**
1. Classify countries using Regulation School scores
2. Compare: wage share, household debt, inequality
3. Stability: Crisis susceptibility (2008)

**Expected:**
- Export-led: More stable, but low wage growth
- Debt-led: Unstable, crisis-prone, high inequality

**Publication Outlet:**
*New Political Economy*, *Competition & Change*

---

## 8. References

### Varieties of Capitalism

- Hall, P. A., & Soskice, D. (2001). *Varieties of Capitalism: The Institutional Foundations of Comparative Advantage*. Oxford University Press.
- Hancké, B., Rhodes, M., & Thatcher, M. (2007). *Beyond Varieties of Capitalism: Conflict, Contradictions, and Complementarities*. Oxford University Press.
- Amable, B. (2003). *The Diversity of Modern Capitalism*. Oxford University Press.

### Power Resources Theory

- Korpi, W. (1983). *The Democratic Class Struggle*. Routledge.
- Esping-Andersen, G. (1990). *The Three Worlds of Welfare Capitalism*. Princeton University Press.
- Huber, E., & Stephens, J. D. (2001). *Development and Crisis of the Welfare State*. University of Chicago Press.

### Financialization

- Epstein, G. A. (Ed.). (2005). *Financialization and the World Economy*. Edward Elgar.
- Krippner, G. R. (2011). *Capitalizing on Crisis: The Political Origins of the Rise of Finance*. Harvard University Press.
- Lazonick, W., & O'Sullivan, M. (2000). Maximizing shareholder value: a new ideology for corporate governance. *Economy and Society*, 29(1), 13-35.
- Stockhammer, E. (2004). Financialisation and the slowdown of accumulation. *Cambridge Journal of Economics*, 28(5), 719-741.

### Neoliberalism

- Harvey, D. (2005). *A Brief History of Neoliberalism*. Oxford University Press.
- Crouch, C. (2011). *The Strange Non-Death of Neoliberalism*. Polity.
- Mirowski, P., & Plehwe, D. (Eds.). (2009). *The Road from Mont Pèlerin: The Making of the Neoliberal Thought Collective*. Harvard University Press.

### Regulation School

- Aglietta, M. (1979). *A Theory of Capitalist Regulation: The US Experience*. Verso.
- Boyer, R., & Saillard, Y. (Eds.). (2002). *Régulation Theory: The State of the Art*. Routledge.
- Boyer, R. (2000). Is a finance-led growth regime a viable alternative to Fordism? A preliminary analysis. *Economy and Society*, 29(1), 111-145.

### Comparative Political Economy (General)

- Streeck, W. (2009). *Re-Forming Capitalism: Institutional Change in the German Political Economy*. Oxford University Press.
- Thelen, K. (2014). *Varieties of Liberalization and the New Politics of Social Solidarity*. Cambridge University Press.
- Baccaro, L., & Pontusson, J. (2016). Rethinking comparative political economy: the growth model perspective. *Politics & Society*, 44(2), 175-207.

### Historical Institutionalism

- Pierson, P. (2004). *Politics in Time: History, Institutions, and Social Analysis*. Princeton University Press.
- Mahoney, J., & Thelen, K. (Eds.). (2010). *Explaining Institutional Change: Ambiguity, Agency, and Power*. Cambridge University Press.

### Data Sources (Real-world)

- OECD Employment Database
- ICTWSS Database (Amsterdam Institute for Advanced Labour Studies)
- Comparative Political Data Set (Armingeon et al.)
- Chinn-Ito Financial Openness Index
- World Bank World Development Indicators
- IMF International Financial Statistics
- European Social Survey
- Luxembourg Income Study (LIS)

---

## Appendix A: Critical Junctures in Dataset

The synthetic data incorporates realistic historical dynamics including:

### 1. Neoliberal Turn (~1980)

**Empirical manifestations:**
- Decline in union density (especially LME)
- Increase in labor market flexibility
- Financial deregulation begins
- Tax cuts on top earners
- Welfare retrenchment (except Nordic)

**Countries most affected:** USA, UK, Chile

### 2. Financialization Acceleration (mid-1990s)

**Empirical manifestations:**
- Rapid growth of financial sector
- Rising household debt
- Stock market boom
- NFC financialization
- Shift to shareholder value

**Countries most affected:** USA, UK, Spain, Ireland

### 3. Financial Crisis (2008)

**Empirical manifestations:**
- Stock market crash
- Deleveraging (household & corporate debt decline)
- Social spending spike (automatic stabilizers)
- Some re-regulation
- But persistent neoliberal orientation

**Countries most affected:** USA, UK, Ireland, Spain, Greece

---

## Appendix B: Institutional Complementarities Matrix

| Institution 1 | Institution 2 | Regime | Expected Correlation |
|---------------|---------------|--------|---------------------|
| Labor coordination | Vocational training | CME | Positive (strong) |
| Stakeholder governance | Interfirm cooperation | CME | Positive (strong) |
| Labor flexibility | Stock market finance | LME | Positive (strong) |
| Union density | Welfare generosity | All | Positive (moderate) |
| Left cabinet | Decommodification | All | Positive (moderate) |
| Financialization | Wage share | All | Negative (strong) |
| Neoliberalism | Power resources | All | Negative (strong) |

---

## Appendix C: Variable Coding and Ranges

| Variable | Type | Range | Interpretation |
|----------|------|-------|----------------|
| Indices (most) | Continuous | 0-1 | 0 = minimal, 1 = maximal |
| `wage_share_gdp` | Continuous | 0.45-0.75 | Labor share of GDP |
| `social_spending_gdp` | Continuous | 5-35 | % of GDP |
| `strike_days_per_1000` | Count | 0-150+ | Annual labor disruption |
| `household_debt_to_income` | Ratio | 0.2-3.0 | Multiples of income |
| `trade_openness` | Ratio | 0.2-2.0 | Multiples of GDP |
| `top_tax_rate` | Proportion | 0.20-0.90 | Marginal rate |

---

## Contact and Citation

**For academic use, please cite:**

[Your Name]. (2025). *Institutional Political Economy Quantitative Analysis System: Measurement Frameworks for Varieties of Capitalism, Power Resources Theory, and Financialization* [Software].

**Contact:** [Your email]

**License:** MIT License - Free for academic and educational use

**Contributions:** Issues and pull requests welcome at [repository URL]

---

**End of Documentation**
