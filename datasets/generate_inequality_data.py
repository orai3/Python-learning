"""
Synthetic Income & Wealth Distribution Dataset Generator
==========================================================
Generates comprehensive inequality data including:
- Income distribution by quintiles/deciles
- Wealth distribution (more unequal than income)
- Gini coefficients and other inequality measures
- Functional vs personal distribution
- Trends reflecting financialization and neoliberal period

Theoretical basis: Post-Keynesian, Marxian approaches to distribution
"""

import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(43)

# Configuration
START_YEAR = 1970
YEARS = 55  # 1970-2024

years = np.arange(START_YEAR, START_YEAR + YEARS)

# ============================================================================
# FUNCTIONAL DISTRIBUTION (Class shares)
# ============================================================================
# Wage share declining over time (neoliberal period)
# Start at 67%, decline to 58% by 2020s
wage_share_trend = 67 - 9 * (1 - np.exp(-np.arange(YEARS) / 25))

# Cyclical component (Goodwin cycles)
wage_share_cycle = 2 * np.sin(2 * np.pi * np.arange(YEARS) / 8)

wage_share = wage_share_trend + wage_share_cycle + np.random.normal(0, 0.5, YEARS)
wage_share = np.clip(wage_share, 55, 72)

# Profit share (residual)
profit_share = 100 - wage_share

# Rent share (increasing with financialization)
rent_share = 5 + 3 * (1 - np.exp(-np.arange(YEARS) / 30)) + np.random.normal(0, 0.3, YEARS)
rent_share = np.clip(rent_share, 3, 10)

# Adjust so components sum to 100
profit_share = profit_share - rent_share

# ============================================================================
# PERSONAL INCOME DISTRIBUTION
# ============================================================================
# Gini coefficient for income (rising inequality)
# Start at 0.35 (1970s), rise to 0.42 (2020s)
gini_income_trend = 0.35 + 0.07 * (1 - np.exp(-np.arange(YEARS) / 20))
gini_income = gini_income_trend + np.random.normal(0, 0.01, YEARS)
gini_income = np.clip(gini_income, 0.30, 0.50)

# Generate quintile shares based on Gini
# Higher Gini = more concentrated in top quintile

def gini_to_quintiles(gini):
    """Convert Gini to approximate quintile shares"""
    # These are stylized relationships
    top20 = 35 + (gini - 0.35) * 50  # Top quintile
    bottom20 = 10 - (gini - 0.35) * 20  # Bottom quintile
    q2 = 12 - (gini - 0.35) * 10
    q3 = 17 - (gini - 0.35) * 10
    q4 = 23 - (gini - 0.35) * 10

    # Normalize to sum to 100
    total = top20 + bottom20 + q2 + q3 + q4
    return np.array([bottom20, q2, q3, q4, top20]) / total * 100

quintile_shares = np.array([gini_to_quintiles(g) for g in gini_income])

q1_share = quintile_shares[:, 0]  # Bottom 20%
q2_share = quintile_shares[:, 1]
q3_share = quintile_shares[:, 2]  # Middle 20%
q4_share = quintile_shares[:, 3]
q5_share = quintile_shares[:, 4]  # Top 20%

# Top 10%, 1%, 0.1% shares (rising over time)
top10_share = 25 + 10 * (1 - np.exp(-np.arange(YEARS) / 20)) + np.random.normal(0, 0.5, YEARS)
top1_share = 8 + 7 * (1 - np.exp(-np.arange(YEARS) / 18)) + np.random.normal(0, 0.3, YEARS)
top01_share = 2 + 4 * (1 - np.exp(-np.arange(YEARS) / 15)) + np.random.normal(0, 0.2, YEARS)

# Palma ratio (top 10% / bottom 40%)
bottom40_share = q1_share + q2_share
palma_ratio = top10_share / bottom40_share

# 90/10 ratio (9th decile / 1st decile income)
p90_p10_ratio = 4.0 + 1.5 * (1 - np.exp(-np.arange(YEARS) / 20)) + np.random.normal(0, 0.1, YEARS)

# ============================================================================
# WEALTH DISTRIBUTION (much more unequal)
# ============================================================================
# Wealth Gini (higher and rising faster)
gini_wealth_trend = 0.75 + 0.10 * (1 - np.exp(-np.arange(YEARS) / 18))
gini_wealth = gini_wealth_trend + np.random.normal(0, 0.01, YEARS)
gini_wealth = np.clip(gini_wealth, 0.70, 0.90)

# Wealth quintile shares
def gini_to_wealth_quintiles(gini):
    """Wealth is much more concentrated"""
    top20 = 70 + (gini - 0.75) * 100
    bottom20 = 0.5 - (gini - 0.75) * 2
    q2 = 2 - (gini - 0.75) * 5
    q3 = 5 - (gini - 0.75) * 10
    q4 = 22.5 - (gini - 0.75) * 83

    total = top20 + bottom20 + q2 + q3 + q4
    return np.array([bottom20, q2, q3, q4, top20]) / total * 100

wealth_quintile_shares = np.array([gini_to_wealth_quintiles(g) for g in gini_wealth])

wealth_q1_share = wealth_quintile_shares[:, 0]
wealth_q2_share = wealth_quintile_shares[:, 1]
wealth_q3_share = wealth_quintile_shares[:, 2]
wealth_q4_share = wealth_quintile_shares[:, 3]
wealth_q5_share = wealth_quintile_shares[:, 4]

# Top wealth shares (extreme concentration)
wealth_top10_share = 60 + 15 * (1 - np.exp(-np.arange(YEARS) / 15)) + np.random.normal(0, 0.8, YEARS)
wealth_top1_share = 25 + 15 * (1 - np.exp(-np.arange(YEARS) / 15)) + np.random.normal(0, 0.6, YEARS)
wealth_top01_share = 10 + 12 * (1 - np.exp(-np.arange(YEARS) / 12)) + np.random.normal(0, 0.5, YEARS)

# Share of households with zero/negative net worth
negative_wealth_share = 8 + 7 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 0.5, YEARS)

# ============================================================================
# MEDIAN vs MEAN (inequality indicator)
# ============================================================================
# When mean > median, distribution is right-skewed (inequality)
# Ratio declines over time as inequality rises
median_to_mean_income = 0.85 - 0.10 * (1 - np.exp(-np.arange(YEARS) / 20)) + np.random.normal(0, 0.01, YEARS)
median_to_mean_wealth = 0.25 - 0.10 * (1 - np.exp(-np.arange(YEARS) / 18)) + np.random.normal(0, 0.02, YEARS)

# ============================================================================
# INCOME SOURCES BY QUINTILE
# ============================================================================
# Bottom quintile: mostly wages (when employed) + transfers
bottom_quintile_wage_income = 60 + np.random.normal(0, 3, YEARS)
bottom_quintile_transfer_income = 35 + np.random.normal(0, 3, YEARS)
bottom_quintile_capital_income = 5 + np.random.normal(0, 1, YEARS)

# Middle quintile: mostly wages
middle_quintile_wage_income = 85 + np.random.normal(0, 2, YEARS)
middle_quintile_transfer_income = 10 + np.random.normal(0, 2, YEARS)
middle_quintile_capital_income = 5 + np.random.normal(0, 1, YEARS)

# Top quintile: wages + rising capital income
top_quintile_wage_base = 60
top_quintile_capital_trend = 15 + 20 * (1 - np.exp(-np.arange(YEARS) / 20))
top_quintile_wage_income = top_quintile_wage_base - 0.4 * (top_quintile_capital_trend - 15) + np.random.normal(0, 2, YEARS)
top_quintile_capital_income = top_quintile_capital_trend + np.random.normal(0, 2, YEARS)
top_quintile_transfer_income = 100 - top_quintile_wage_income - top_quintile_capital_income

# ============================================================================
# POVERTY AND VULNERABILITY
# ============================================================================
# Poverty rate (% below poverty line) - U-shaped (fell, then rose)
poverty_rate = 12 + 3 * np.sin(2 * np.pi * np.arange(YEARS) / 50) - 2 * np.exp(-np.arange(YEARS) / 15) + \
               2 * (np.arange(YEARS) > 35) + np.random.normal(0, 0.5, YEARS)
poverty_rate = np.clip(poverty_rate, 8, 18)

# Child poverty (higher and more volatile)
child_poverty_rate = poverty_rate * 1.3 + np.random.normal(0, 0.8, YEARS)

# Elderly poverty (declining due to pensions)
elderly_poverty_rate = 25 - 15 * (1 - np.exp(-np.arange(YEARS) / 20)) + np.random.normal(0, 1, YEARS)

# Poverty gap (how far below poverty line)
poverty_gap = 25 + 5 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 2, YEARS)

# ============================================================================
# INTER-GENERATIONAL MOBILITY
# ============================================================================
# Elasticity of child income to parent income (rising = less mobility)
intergenerational_elasticity = 0.35 + 0.15 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 0.02, YEARS)

# ============================================================================
# RACIAL/GENDER WAGE GAPS (stylized)
# ============================================================================
# Gender wage gap (women's earnings as % of men's - rising but still < 100)
gender_wage_ratio = 60 + 20 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 1, YEARS)
gender_wage_ratio = np.clip(gender_wage_ratio, 60, 85)

# Racial wage gap (minority earnings as % of majority)
racial_wage_ratio = 70 + 10 * (1 - np.exp(-np.arange(YEARS) / 30)) + np.random.normal(0, 1, YEARS)
racial_wage_ratio = np.clip(racial_wage_ratio, 65, 85)

# ============================================================================
# CEO-TO-WORKER PAY RATIO (skyrocketing)
# ============================================================================
ceo_worker_ratio = 30 + 250 * (1 - np.exp(-np.arange(YEARS) / 15)) + np.random.normal(0, 10, YEARS)

# ============================================================================
# WEALTH COMPONENTS
# ============================================================================
# Share of wealth in financial assets (rising with financialization)
financial_wealth_share = 35 + 20 * (1 - np.exp(-np.arange(YEARS) / 20)) + np.random.normal(0, 2, YEARS)

# Housing wealth share
housing_wealth_share = 50 - 10 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 2, YEARS)

# Business equity share
business_wealth_share = 100 - financial_wealth_share - housing_wealth_share

# Homeownership rate (by quintile)
homeownership_bottom_quintile = 40 - 8 * (1 - np.exp(-np.arange(YEARS) / 20)) + np.random.normal(0, 2, YEARS)
homeownership_middle_quintile = 65 - 5 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 2, YEARS)
homeownership_top_quintile = 85 + 5 * (1 - np.exp(-np.arange(YEARS) / 30)) + np.random.normal(0, 1, YEARS)

# ============================================================================
# CONSTRUCT DATAFRAME
# ============================================================================
data = pd.DataFrame({
    'year': years,

    # Functional distribution
    'wage_share': wage_share,
    'profit_share': profit_share,
    'rent_share': rent_share,

    # Income inequality measures
    'gini_income': gini_income,
    'gini_wealth': gini_wealth,
    'palma_ratio': palma_ratio,
    'p90_p10_ratio': p90_p10_ratio,

    # Income quintile shares
    'income_q1_share': q1_share,
    'income_q2_share': q2_share,
    'income_q3_share': q3_share,
    'income_q4_share': q4_share,
    'income_q5_share': q5_share,

    # Top income shares
    'income_top10_pct': top10_share,
    'income_top1_pct': top1_share,
    'income_top01_pct': top01_share,

    # Wealth quintile shares
    'wealth_q1_share': wealth_q1_share,
    'wealth_q2_share': wealth_q2_share,
    'wealth_q3_share': wealth_q3_share,
    'wealth_q4_share': wealth_q4_share,
    'wealth_q5_share': wealth_q5_share,

    # Top wealth shares
    'wealth_top10_pct': wealth_top10_share,
    'wealth_top1_pct': wealth_top1_share,
    'wealth_top01_pct': wealth_top01_share,
    'wealth_negative_share': negative_wealth_share,

    # Median vs mean
    'median_to_mean_income': median_to_mean_income,
    'median_to_mean_wealth': median_to_mean_wealth,

    # Income sources by quintile (% of total income)
    'bottom_q_wage_pct': bottom_quintile_wage_income,
    'bottom_q_transfer_pct': bottom_quintile_transfer_income,
    'bottom_q_capital_pct': bottom_quintile_capital_income,

    'middle_q_wage_pct': middle_quintile_wage_income,
    'middle_q_transfer_pct': middle_quintile_transfer_income,
    'middle_q_capital_pct': middle_quintile_capital_income,

    'top_q_wage_pct': top_quintile_wage_income,
    'top_q_transfer_pct': top_quintile_transfer_income,
    'top_q_capital_pct': top_quintile_capital_income,

    # Poverty
    'poverty_rate': poverty_rate,
    'child_poverty_rate': child_poverty_rate,
    'elderly_poverty_rate': elderly_poverty_rate,
    'poverty_gap': poverty_gap,

    # Mobility
    'intergenerational_elasticity': intergenerational_elasticity,

    # Wage gaps
    'gender_wage_ratio': gender_wage_ratio,
    'racial_wage_ratio': racial_wage_ratio,
    'ceo_worker_pay_ratio': ceo_worker_ratio,

    # Wealth composition
    'financial_wealth_pct': financial_wealth_share,
    'housing_wealth_pct': housing_wealth_share,
    'business_wealth_pct': business_wealth_share,

    # Homeownership by quintile
    'homeownership_bottom_q': homeownership_bottom_quintile,
    'homeownership_middle_q': homeownership_middle_quintile,
    'homeownership_top_q': homeownership_top_quintile,
})

# Save to CSV
data.to_csv('/home/user/Python-learning/datasets/inequality_annual_data.csv', index=False)

print(f"Generated inequality dataset with {len(data)} observations")
print(f"Years: {data['year'].min()} to {data['year'].max()}\n")
print(f"Income Gini: {data['gini_income'].iloc[0]:.3f} (1970) → {data['gini_income'].iloc[-1]:.3f} (2024)")
print(f"Wealth Gini: {data['gini_wealth'].iloc[0]:.3f} (1970) → {data['gini_wealth'].iloc[-1]:.3f} (2024)")
print(f"Top 1% income share: {data['income_top1_pct'].iloc[0]:.1f}% (1970) → {data['income_top1_pct'].iloc[-1]:.1f}% (2024)")
print(f"Wage share: {data['wage_share'].iloc[0]:.1f}% (1970) → {data['wage_share'].iloc[-1]:.1f}% (2024)")
print(f"CEO/Worker ratio: {data['ceo_worker_pay_ratio'].iloc[0]:.0f}x (1970) → {data['ceo_worker_pay_ratio'].iloc[-1]:.0f}x (2024)")
