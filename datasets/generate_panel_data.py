"""
Synthetic Cross-Country Panel Dataset Generator
================================================
Generates panel data for 30 countries over 50 years (1970-2024)
with quarterly frequency where appropriate.

Focus on comparative political economy:
- Different varieties of capitalism (LME, CME, MME)
- Different development levels
- Institutional differences
- Policy regime variations

Ideal for heterodox cross-country comparative analysis.
"""

import numpy as np
import pandas as pd

np.random.seed(46)

# Configuration
N_COUNTRIES = 30
START_YEAR = 1970
END_YEAR = 2024
YEARS = END_YEAR - START_YEAR + 1

# ============================================================================
# COUNTRY DEFINITIONS
# ============================================================================

countries = [
    # Advanced Liberal Market Economies (LME)
    {'code': 'USA', 'name': 'United States', 'type': 'LME', 'region': 'North America', 'dev_level': 'Advanced'},
    {'code': 'GBR', 'name': 'United Kingdom', 'type': 'LME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'CAN', 'name': 'Canada', 'type': 'LME', 'region': 'North America', 'dev_level': 'Advanced'},
    {'code': 'AUS', 'name': 'Australia', 'type': 'LME', 'region': 'Oceania', 'dev_level': 'Advanced'},

    # Coordinated Market Economies (CME)
    {'code': 'DEU', 'name': 'Germany', 'type': 'CME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'SWE', 'name': 'Sweden', 'type': 'CME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'NLD', 'name': 'Netherlands', 'type': 'CME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'AUT', 'name': 'Austria', 'type': 'CME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'DNK', 'name': 'Denmark', 'type': 'CME', 'region': 'Europe', 'dev_level': 'Advanced'},

    # Mediterranean/Mixed Market Economies (MME)
    {'code': 'FRA', 'name': 'France', 'type': 'MME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'ITA', 'name': 'Italy', 'type': 'MME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'ESP', 'name': 'Spain', 'type': 'MME', 'region': 'Europe', 'dev_level': 'Advanced'},
    {'code': 'GRC', 'name': 'Greece', 'type': 'MME', 'region': 'Europe', 'dev_level': 'Advanced'},

    # East Asian Developmental States
    {'code': 'JPN', 'name': 'Japan', 'type': 'Developmental', 'region': 'Asia', 'dev_level': 'Advanced'},
    {'code': 'KOR', 'name': 'South Korea', 'type': 'Developmental', 'region': 'Asia', 'dev_level': 'Advanced'},
    {'code': 'TWN', 'name': 'Taiwan', 'type': 'Developmental', 'region': 'Asia', 'dev_level': 'Emerging'},

    # Emerging Markets - Large
    {'code': 'CHN', 'name': 'China', 'type': 'State-led', 'region': 'Asia', 'dev_level': 'Emerging'},
    {'code': 'IND', 'name': 'India', 'type': 'Mixed', 'region': 'Asia', 'dev_level': 'Emerging'},
    {'code': 'BRA', 'name': 'Brazil', 'type': 'Mixed', 'region': 'Latin America', 'dev_level': 'Emerging'},
    {'code': 'MEX', 'name': 'Mexico', 'type': 'LME', 'region': 'Latin America', 'dev_level': 'Emerging'},
    {'code': 'RUS', 'name': 'Russia', 'type': 'State-led', 'region': 'Europe', 'dev_level': 'Emerging'},
    {'code': 'TUR', 'name': 'Turkey', 'type': 'Mixed', 'region': 'Middle East', 'dev_level': 'Emerging'},

    # Emerging Markets - Smaller
    {'code': 'POL', 'name': 'Poland', 'type': 'Transition', 'region': 'Europe', 'dev_level': 'Emerging'},
    {'code': 'ZAF', 'name': 'South Africa', 'type': 'Mixed', 'region': 'Africa', 'dev_level': 'Emerging'},
    {'code': 'IDN', 'name': 'Indonesia', 'type': 'Mixed', 'region': 'Asia', 'dev_level': 'Emerging'},
    {'code': 'THA', 'name': 'Thailand', 'type': 'Developmental', 'region': 'Asia', 'dev_level': 'Emerging'},

    # Developing
    {'code': 'VNM', 'name': 'Vietnam', 'type': 'State-led', 'region': 'Asia', 'dev_level': 'Developing'},
    {'code': 'NGA', 'name': 'Nigeria', 'type': 'Mixed', 'region': 'Africa', 'dev_level': 'Developing'},
    {'code': 'BGD', 'name': 'Bangladesh', 'type': 'Mixed', 'region': 'Asia', 'dev_level': 'Developing'},
    {'code': 'ARG', 'name': 'Argentina', 'type': 'Mixed', 'region': 'Latin America', 'dev_level': 'Emerging'},
]

# ============================================================================
# COUNTRY CHARACTERISTICS (time-invariant or slow-moving)
# ============================================================================

country_chars = {}
for i, country in enumerate(countries):
    code = country['code']

    # Growth parameters (differ by development level and type)
    if country['dev_level'] == 'Advanced':
        base_growth = 0.006  # 0.6% quarterly
        volatility = 0.01
    elif country['dev_level'] == 'Emerging':
        base_growth = 0.012  # 1.2% quarterly (catch-up)
        volatility = 0.02
    else:  # Developing
        base_growth = 0.015
        volatility = 0.03

    # Institutional quality (affects volatility and crisis probability)
    if country['type'] == 'CME':
        inst_quality = 0.9
        crisis_prob = 0.02
    elif country['type'] == 'LME':
        inst_quality = 0.85
        crisis_prob = 0.03
    elif country['type'] == 'Developmental':
        inst_quality = 0.80
        crisis_prob = 0.025
    else:
        inst_quality = 0.65
        crisis_prob = 0.05

    # Inequality (structural, varies by type)
    if country['type'] == 'CME':
        base_gini = 0.28
    elif country['type'] == 'LME':
        base_gini = 0.38
    elif country['type'] == 'Developmental':
        base_gini = 0.35
    else:
        base_gini = 0.45

    # Wage share (varies by institutional type)
    if country['type'] == 'CME':
        base_wage_share = 0.68
    elif country['type'] == 'LME':
        base_wage_share = 0.60
    else:
        base_wage_share = 0.55

    # Size
    if code in ['USA', 'CHN', 'IND', 'JPN', 'DEU', 'GBR', 'FRA']:
        size = 'Large'
        base_gdp = np.random.uniform(1000, 5000)
    elif code in ['BRA', 'RUS', 'ITA', 'CAN', 'KOR', 'ESP', 'AUS', 'MEX']:
        size = 'Medium'
        base_gdp = np.random.uniform(500, 1200)
    else:
        size = 'Small'
        base_gdp = np.random.uniform(100, 600)

    country_chars[code] = {
        'base_growth': base_growth,
        'volatility': volatility,
        'inst_quality': inst_quality,
        'crisis_prob': crisis_prob,
        'base_gini': base_gini,
        'base_wage_share': base_wage_share,
        'size': size,
        'base_gdp': base_gdp,
    }

# ============================================================================
# GENERATE TIME SERIES FOR EACH COUNTRY
# ============================================================================

all_data = []

for country in countries:
    code = country['code']
    chars = country_chars[code]

    # Time dimension (annual for simplicity, quarterly would be 4x larger)
    years = np.arange(START_YEAR, END_YEAR + 1)
    n_periods = len(years)

    # GDP growth with trend and cycles
    growth_trend = chars['base_growth'] * 4  # Annualized
    growth_cycle = chars['volatility'] * np.sin(2 * np.pi * np.arange(n_periods) / 8)
    growth_shock = np.random.normal(0, chars['volatility'], n_periods)

    # Crisis events (country-specific)
    crisis = np.random.binomial(1, chars['crisis_prob'], n_periods)
    crisis_effect = -0.05 * crisis

    gdp_growth = growth_trend + growth_cycle + growth_shock + crisis_effect

    # GDP level
    gdp = np.zeros(n_periods)
    gdp[0] = chars['base_gdp']
    for t in range(1, n_periods):
        gdp[t] = gdp[t-1] * (1 + gdp_growth[t])

    # GDP per capita (population grows slower)
    pop_growth = 0.01 - 0.005 * (country['dev_level'] == 'Advanced')
    population = chars['base_gdp'] * 0.5 * (1 + pop_growth) ** np.arange(n_periods)
    gdp_per_capita = gdp / population

    # Unemployment (counter-cyclical with Okun's law)
    natural_rate = 5 + np.random.uniform(-2, 3)
    unemployment = natural_rate - 0.3 * (gdp_growth - growth_trend) * 100 + np.random.normal(0, 0.5, n_periods)
    unemployment = np.clip(unemployment, 2, 25)

    # Inflation (varies by regime and development level)
    if country['dev_level'] == 'Advanced':
        base_inflation = 0.025
        inflation_vol = 0.01
    else:
        base_inflation = 0.06
        inflation_vol = 0.03

    inflation = base_inflation + 0.5 * growth_cycle + np.random.normal(0, inflation_vol, n_periods)
    inflation = np.clip(inflation, -0.02, 0.25)

    # Wage share (declining trend in neoliberal era)
    neoliberal_effect = -0.05 * (1 - np.exp(-np.arange(n_periods) / 20))
    wage_share = chars['base_wage_share'] + neoliberal_effect + \
                 0.02 * np.sin(2 * np.pi * np.arange(n_periods) / 8) + \
                 np.random.normal(0, 0.01, n_periods)
    wage_share = np.clip(wage_share, 0.45, 0.75)

    # Gini coefficient (rising inequality trend)
    gini_trend = 0.05 * (1 - np.exp(-np.arange(n_periods) / 25))
    gini = chars['base_gini'] + gini_trend + np.random.normal(0, 0.01, n_periods)
    gini = np.clip(gini, 0.23, 0.65)

    # Government spending (% of GDP)
    if country['type'] == 'CME':
        gov_spending_gdp = 45 + np.random.normal(0, 2, n_periods)
    elif country['type'] == 'LME':
        gov_spending_gdp = 35 + np.random.normal(0, 2, n_periods)
    else:
        gov_spending_gdp = 30 + np.random.normal(0, 3, n_periods)
    gov_spending_gdp = np.clip(gov_spending_gdp, 20, 60)

    # Government debt (% of GDP)
    gov_debt_gdp = np.zeros(n_periods)
    gov_debt_gdp[0] = 40 + np.random.uniform(-20, 30)
    for t in range(1, n_periods):
        # Debt grows with deficits and crises
        deficit = 2 + crisis[t] * 5 - 0.5 * gdp_growth[t] * 100
        gov_debt_gdp[t] = gov_debt_gdp[t-1] + deficit - gdp_growth[t] * gov_debt_gdp[t-1]
    gov_debt_gdp = np.clip(gov_debt_gdp, 10, 250)

    # Private debt (% of GDP, rising with financialization)
    private_debt_trend = 80 + 60 * (1 - np.exp(-np.arange(n_periods) / 25))
    private_debt_gdp = private_debt_trend + 10 * np.sin(2 * np.pi * np.arange(n_periods) / 12) + \
                      np.random.normal(0, 5, n_periods)
    private_debt_gdp = np.clip(private_debt_gdp, 40, 250)

    # Trade openness (exports + imports) / GDP
    if chars['size'] == 'Small':
        trade_openness = 80 + np.random.normal(0, 5, n_periods)
    elif chars['size'] == 'Medium':
        trade_openness = 50 + np.random.normal(0, 5, n_periods)
    else:
        trade_openness = 30 + np.random.normal(0, 5, n_periods)

    # Current account balance (% of GDP)
    if country['type'] == 'CME' or code in ['CHN', 'JPN', 'KOR', 'DEU']:
        # Export-oriented economies
        current_account_gdp = 3 + np.random.normal(0, 2, n_periods)
    else:
        current_account_gdp = -2 + np.random.normal(0, 3, n_periods)

    # FDI inflows (% of GDP)
    if country['dev_level'] == 'Developing' or country['dev_level'] == 'Emerging':
        fdi_inflows_gdp = 3 + np.random.normal(0, 1.5, n_periods)
    else:
        fdi_inflows_gdp = 1.5 + np.random.normal(0, 1, n_periods)
    fdi_inflows_gdp = np.clip(fdi_inflows_gdp, -5, 15)

    # Financial development index (0-1, rising over time)
    fin_dev = 0.3 + 0.4 * (1 - np.exp(-np.arange(n_periods) / 30)) + \
              0.2 * (country['dev_level'] == 'Advanced') + \
              np.random.normal(0, 0.05, n_periods)
    fin_dev = np.clip(fin_dev, 0, 1)

    # Union density (% of workforce)
    if country['type'] == 'CME':
        base_union = 60
        decline_rate = 10
    elif country['type'] == 'LME':
        base_union = 35
        decline_rate = 20
    else:
        base_union = 25
        decline_rate = 15

    union_density = base_union - decline_rate * (1 - np.exp(-np.arange(n_periods) / 25)) + \
                   np.random.normal(0, 2, n_periods)
    union_density = np.clip(union_density, 5, 80)

    # Employment protection index (0-6 OECD scale)
    if country['type'] == 'CME' or country['type'] == 'MME':
        emp_protection = 2.5 + np.random.normal(0, 0.3, n_periods)
    elif country['type'] == 'LME':
        emp_protection = 1.0 + np.random.normal(0, 0.2, n_periods)
    else:
        emp_protection = 2.0 + np.random.normal(0, 0.4, n_periods)
    emp_protection = np.clip(emp_protection, 0, 6)

    # Create dataframe for this country
    country_data = pd.DataFrame({
        'country_code': code,
        'country_name': country['name'],
        'year': years,
        'capitalism_type': country['type'],
        'region': country['region'],
        'development_level': country['dev_level'],
        'size_category': chars['size'],

        # Macroeconomic variables
        'gdp': gdp.round(2),
        'gdp_growth': (gdp_growth * 100).round(2),
        'gdp_per_capita': gdp_per_capita.round(2),
        'unemployment_rate': unemployment.round(2),
        'inflation_rate': (inflation * 100).round(2),
        'population': population.round(2),

        # Distribution
        'wage_share': (wage_share * 100).round(2),
        'gini_coefficient': gini.round(3),

        # Government
        'gov_spending_gdp': gov_spending_gdp.round(2),
        'gov_debt_gdp': gov_debt_gdp.round(2),

        # Finance
        'private_debt_gdp': private_debt_gdp.round(2),
        'financial_development_index': fin_dev.round(3),

        # Trade
        'trade_openness': trade_openness.round(2),
        'current_account_gdp': current_account_gdp.round(2),
        'fdi_inflows_gdp': fdi_inflows_gdp.round(2),

        # Labor institutions
        'union_density': union_density.round(2),
        'employment_protection': emp_protection.round(2),

        # Crisis indicator
        'crisis_year': crisis,
    })

    all_data.append(country_data)

# Combine all countries
panel_data = pd.concat(all_data, ignore_index=True)

# Save to CSV
panel_data.to_csv('/home/user/Python-learning/datasets/cross_country_panel_data.csv', index=False)

print(f"Generated cross-country panel dataset")
print(f"Countries: {N_COUNTRIES}")
print(f"Years: {START_YEAR} to {END_YEAR}")
print(f"Total observations: {len(panel_data):,}\n")

print("Country types:")
print(panel_data.groupby('capitalism_type')['country_code'].nunique())

print("\nDevelopment levels:")
print(panel_data.groupby('development_level')['country_code'].nunique())

print("\nSample statistics (2024):")
final_year = panel_data[panel_data['year'] == 2024]
print(f"Average GDP growth: {final_year['gdp_growth'].mean():.2f}%")
print(f"Average unemployment: {final_year['unemployment_rate'].mean():.2f}%")
print(f"Average wage share: {final_year['wage_share'].mean():.2f}%")
print(f"Average Gini: {final_year['gini_coefficient'].mean():.3f}")
