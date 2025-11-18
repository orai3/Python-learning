"""
Synthetic Financial Crisis Dataset Generator
=============================================
Generates high-frequency data showing financial crisis dynamics
based on Minsky's Financial Instability Hypothesis.

Includes:
- Normal periods with financial stability
- Boom periods with rising fragility (hedge → speculative → Ponzi)
- Crisis periods with sudden collapse
- Recovery periods with deleveraging

Shows multiple crisis episodes with realistic contagion and feedback loops.
Quarterly data from 1990-2024 (140 quarters)
"""

import numpy as np
import pandas as pd
from datetime import datetime

np.random.seed(47)

# Configuration
START_DATE = '1990-Q1'
QUARTERS = 140  # 35 years (1990-2024)

# Create date range
dates = pd.date_range(start='1990-01-01', periods=QUARTERS, freq='Q')

# ============================================================================
# CRISIS PERIODS (manually defined)
# ============================================================================
# We'll model 4 major crises with realistic build-up and recovery

crisis_episodes = [
    {'name': 'Dot-com Bubble', 'buildup': (32, 48), 'crisis': (48, 52), 'recovery': (52, 60)},  # 1998-2002
    {'name': 'Global Financial Crisis', 'buildup': (60, 72), 'crisis': (72, 78), 'recovery': (78, 92)},  # 2005-2013
    {'name': 'COVID-19 Shock', 'buildup': (116, 120), 'crisis': (120, 122), 'recovery': (122, 128)},  # 2019-2022
    {'name': 'Minor Crisis', 'buildup': (18, 26), 'crisis': (26, 28), 'recovery': (28, 32)},  # 1994-1998
]

# Initialize regime indicator
# 0 = Normal, 1 = Boom/Buildup, 2 = Crisis, 3 = Recovery
regime = np.zeros(QUARTERS, dtype=int)

for episode in crisis_episodes:
    buildup_start, buildup_end = episode['buildup']
    crisis_start, crisis_end = episode['crisis']
    recovery_start, recovery_end = episode['recovery']

    regime[buildup_start:buildup_end] = 1
    regime[crisis_start:crisis_end] = 2
    regime[recovery_start:recovery_end] = 3

# ============================================================================
# MINSKY'S FINANCIAL FRAGILITY
# ============================================================================
# Measure of system fragility (0-1, higher = more fragile)

fragility = np.zeros(QUARTERS)

for t in range(1, QUARTERS):
    if regime[t] == 0:  # Normal
        # Gradual increase during calm (complacency)
        fragility[t] = min(fragility[t-1] + np.random.uniform(0, 0.01), 0.4)
    elif regime[t] == 1:  # Boom/Buildup
        # Rapid increase (Minsky moment approaching)
        fragility[t] = min(fragility[t-1] + np.random.uniform(0.01, 0.03), 0.95)
    elif regime[t] == 2:  # Crisis
        # Remains very high during crisis
        fragility[t] = np.random.uniform(0.85, 0.98)
    else:  # Recovery
        # Deleveraging reduces fragility
        fragility[t] = max(fragility[t-1] - np.random.uniform(0.02, 0.05), 0.15)

# Smooth fragility a bit
fragility = np.convolve(fragility, np.ones(3)/3, mode='same')
fragility = np.clip(fragility, 0, 1)

# ============================================================================
# FINANCIAL VARIABLES
# ============================================================================

# Credit growth (very procyclical, leads crises)
credit_growth = np.zeros(QUARTERS)
for t in range(QUARTERS):
    if regime[t] == 0:
        credit_growth[t] = 2.0 + np.random.normal(0, 1)
    elif regime[t] == 1:
        credit_growth[t] = 8.0 + 4 * fragility[t] + np.random.normal(0, 2)
    elif regime[t] == 2:
        credit_growth[t] = -10.0 - 5 * fragility[t] + np.random.normal(0, 3)
    else:
        credit_growth[t] = 0.5 + np.random.normal(0, 1.5)

# Total credit (level, accumulation)
credit = np.zeros(QUARTERS)
credit[0] = 100
for t in range(1, QUARTERS):
    credit[t] = credit[t-1] * (1 + credit_growth[t] / 100)

# Leverage ratio (debt/equity)
leverage = 2.0 + 2.5 * fragility + 0.5 * np.random.randn(QUARTERS)
leverage = np.clip(leverage, 1.0, 6.0)

# Asset prices (boom-bust cycles)
asset_price_growth = np.zeros(QUARTERS)
for t in range(QUARTERS):
    if regime[t] == 0:
        asset_price_growth[t] = 1.5 + np.random.normal(0, 2)
    elif regime[t] == 1:
        asset_price_growth[t] = 5.0 + 10 * fragility[t] + np.random.normal(0, 3)
    elif regime[t] == 2:
        asset_price_growth[t] = -15.0 - 20 * fragility[t] + np.random.normal(0, 5)
    else:
        asset_price_growth[t] = 2.0 + np.random.normal(0, 2.5)

asset_price_index = np.zeros(QUARTERS)
asset_price_index[0] = 100
for t in range(1, QUARTERS):
    asset_price_index[t] = asset_price_index[t-1] * (1 + asset_price_growth[t] / 100)
asset_price_index = np.maximum(asset_price_index, 10)  # Floor

# Credit spreads (risk premium, counter-cyclical)
credit_spread = 2.0 + 5.0 * fragility + np.random.normal(0, 0.5, QUARTERS)
credit_spread = np.clip(credit_spread, 0.5, 10.0)

# Bank lending standards index (higher = tighter)
lending_standards = -2.0 + 4.0 * fragility + np.random.normal(0, 0.5, QUARTERS)
lending_standards = np.clip(lending_standards, -3, 3)

# VIX-like volatility index
vix = 15 + 40 * fragility + np.random.normal(0, 3, QUARTERS)
vix = np.clip(vix, 10, 80)

# ============================================================================
# MINSKY'S BORROWER CATEGORIES
# ============================================================================
# Share of hedge, speculative, and Ponzi borrowers

# Hedge finance (can pay interest and principal from income)
hedge_share = 70 - 40 * fragility + np.random.normal(0, 3, QUARTERS)
hedge_share = np.clip(hedge_share, 20, 80)

# Ponzi finance (cannot even pay interest from income)
ponzi_share = 5 + 25 * (fragility ** 2) + np.random.normal(0, 2, QUARTERS)
ponzi_share = np.clip(ponzi_share, 2, 40)

# Speculative finance (residual)
speculative_share = 100 - hedge_share - ponzi_share

# ============================================================================
# REAL ECONOMY (affected by financial conditions)
# ============================================================================

# GDP growth (depressed during crises)
gdp_growth = np.zeros(QUARTERS)
for t in range(QUARTERS):
    if regime[t] == 0:
        gdp_growth[t] = 0.6 + np.random.normal(0, 0.3)
    elif regime[t] == 1:
        gdp_growth[t] = 0.8 + 0.3 * fragility[t] + np.random.normal(0, 0.2)
    elif regime[t] == 2:
        gdp_growth[t] = -1.5 - 2.0 * fragility[t] + np.random.normal(0, 0.5)
    else:
        gdp_growth[t] = 0.4 + np.random.normal(0, 0.4)

gdp = np.zeros(QUARTERS)
gdp[0] = 100
for t in range(1, QUARTERS):
    gdp[t] = gdp[t-1] * (1 + gdp_growth[t] / 100)

# Unemployment (counter-cyclical)
unemployment = 5.0 - 0.5 * gdp_growth + 3 * (regime == 2) + np.random.normal(0, 0.3, QUARTERS)
unemployment = np.clip(unemployment, 3, 15)

# Investment (very sensitive to financial conditions)
investment_growth = 2 * gdp_growth - 5 * (regime == 2) + np.random.normal(0, 2, QUARTERS)

investment = np.zeros(QUARTERS)
investment[0] = 20
for t in range(1, QUARTERS):
    investment[t] = investment[t-1] * (1 + investment_growth[t] / 100)

# Consumption (more stable)
consumption = gdp * 0.65

# Corporate profits (procyclical, affected by debt service)
profit_margin = 8 - 2 * fragility + np.random.normal(0, 0.5, QUARTERS)
profit_margin = np.clip(profit_margin, 2, 12)
corporate_profits = gdp * profit_margin / 100

# ============================================================================
# DEBT DYNAMICS
# ============================================================================

# Household debt (% of GDP)
household_debt_gdp = 70 + 40 * fragility + np.random.normal(0, 3, QUARTERS)
household_debt_gdp = np.clip(household_debt_gdp, 50, 140)

# Corporate debt (% of GDP)
corporate_debt_gdp = 80 + 50 * fragility + np.random.normal(0, 4, QUARTERS)
corporate_debt_gdp = np.clip(corporate_debt_gdp, 60, 180)

# Debt service ratio (% of income)
household_debt_service = 8 + 6 * fragility + np.random.normal(0, 0.5, QUARTERS)
household_debt_service = np.clip(household_debt_service, 6, 18)

corporate_debt_service = 10 + 8 * fragility + np.random.normal(0, 0.8, QUARTERS)
corporate_debt_service = np.clip(corporate_debt_service, 8, 25)

# ============================================================================
# BANKING SECTOR
# ============================================================================

# Bank capital ratio (falls during booms, rises after crises)
bank_capital_ratio = 12 - 4 * fragility + 2 * (regime == 3) + np.random.normal(0, 0.5, QUARTERS)
bank_capital_ratio = np.clip(bank_capital_ratio, 6, 16)

# Non-performing loan ratio
npl_ratio = 2 + 8 * fragility + 5 * (regime == 2) + np.random.normal(0, 0.5, QUARTERS)
npl_ratio = np.clip(npl_ratio, 1, 20)

# Bank profitability (ROE)
bank_roe = 12 + 5 * (fragility) * (regime != 2) - 20 * (regime == 2) + np.random.normal(0, 2, QUARTERS)
bank_roe = np.clip(bank_roe, -30, 25)

# Financial sector size (% of GDP, growing over time)
finance_sector_gdp = 5 + 3 * (np.arange(QUARTERS) / QUARTERS) + 2 * fragility + np.random.normal(0, 0.3, QUARTERS)
finance_sector_gdp = np.clip(finance_sector_gdp, 4, 12)

# ============================================================================
# CRISIS INDICATORS
# ============================================================================

# Probability of crisis in next 4 quarters (early warning)
crisis_probability = fragility * 100

# Financial stress index (0-100)
financial_stress = (fragility * 60 +
                   (npl_ratio / 20) * 20 +
                   (credit_spread / 10) * 20)
financial_stress = np.clip(financial_stress, 0, 100)

# Systemic risk indicator (tail risk)
systemic_risk = fragility * vix / 20
systemic_risk = np.clip(systemic_risk, 0, 4)

# ============================================================================
# POLICY RESPONSES
# ============================================================================

# Interest rate (policy rate, low during crisis)
interest_rate = np.zeros(QUARTERS)
for t in range(QUARTERS):
    if regime[t] == 2 or regime[t] == 3:
        interest_rate[t] = 0.5 + np.random.uniform(0, 0.5)
    else:
        interest_rate[t] = 2.5 + 2 * (1 - fragility[t]) + np.random.normal(0, 0.3)
interest_rate = np.clip(interest_rate, 0, 6)

# Government deficit (counter-cyclical)
gov_deficit_gdp = -2 + 8 * (regime == 2) + 3 * (regime == 3) + np.random.normal(0, 1, QUARTERS)

# Central bank balance sheet (% of GDP, expands during crises)
cb_balance_sheet = 10 + 5 * (np.arange(QUARTERS) / QUARTERS) + 15 * (regime == 2) + \
                  np.random.normal(0, 1, QUARTERS)
cb_balance_sheet = np.clip(cb_balance_sheet, 8, 40)

# ============================================================================
# CONSTRUCT DATAFRAME
# ============================================================================

data = pd.DataFrame({
    'date': dates,
    'year': dates.year,
    'quarter': dates.quarter,
    'regime': regime,  # 0=Normal, 1=Boom, 2=Crisis, 3=Recovery

    # Minsky framework
    'fragility_index': fragility.round(3),
    'hedge_share': hedge_share.round(2),
    'speculative_share': speculative_share.round(2),
    'ponzi_share': ponzi_share.round(2),

    # Financial variables
    'credit_total': credit.round(2),
    'credit_growth': credit_growth.round(2),
    'leverage_ratio': leverage.round(2),
    'asset_price_index': asset_price_index.round(2),
    'asset_price_growth': asset_price_growth.round(2),
    'credit_spread_bps': (credit_spread * 100).round(0),
    'lending_standards_index': lending_standards.round(2),
    'volatility_index': vix.round(2),

    # Real economy
    'gdp': gdp.round(2),
    'gdp_growth_pct': gdp_growth.round(2),
    'unemployment_rate': unemployment.round(2),
    'investment': investment.round(2),
    'investment_growth': investment_growth.round(2),
    'consumption': consumption.round(2),
    'corporate_profits': corporate_profits.round(2),
    'profit_margin': profit_margin.round(2),

    # Debt
    'household_debt_gdp': household_debt_gdp.round(2),
    'corporate_debt_gdp': corporate_debt_gdp.round(2),
    'household_debt_service': household_debt_service.round(2),
    'corporate_debt_service': corporate_debt_service.round(2),

    # Banking
    'bank_capital_ratio': bank_capital_ratio.round(2),
    'npl_ratio': npl_ratio.round(2),
    'bank_roe': bank_roe.round(2),
    'finance_sector_gdp': finance_sector_gdp.round(2),

    # Crisis indicators
    'crisis_probability': crisis_probability.round(2),
    'financial_stress_index': financial_stress.round(2),
    'systemic_risk_index': systemic_risk.round(2),

    # Policy
    'policy_rate': interest_rate.round(2),
    'gov_deficit_gdp': gov_deficit_gdp.round(2),
    'central_bank_assets_gdp': cb_balance_sheet.round(2),
})

# Add crisis episode names
data['crisis_episode'] = ''
for episode in crisis_episodes:
    crisis_start, crisis_end = episode['crisis']
    data.loc[crisis_start:crisis_end-1, 'crisis_episode'] = episode['name']

# Save to CSV
data.to_csv('/home/user/Python-learning/datasets/financial_crisis_data.csv', index=False)

print(f"Generated financial crisis dataset with {len(data)} observations")
print(f"Period: {data['date'].min()} to {data['date'].max()}\n")

print("Crisis episodes:")
crisis_data = data[data['regime'] == 2]
print(f"Total crisis quarters: {len(crisis_data)}")
print(f"Crisis episodes: {crisis_data['crisis_episode'].unique()}\n")

print("Regime distribution:")
print(data['regime'].value_counts().sort_index())

print(f"\nAverage fragility by regime:")
for reg in range(4):
    regime_name = ['Normal', 'Boom', 'Crisis', 'Recovery'][reg]
    avg_frag = data[data['regime'] == reg]['fragility_index'].mean()
    print(f"  {regime_name}: {avg_frag:.3f}")

print(f"\nCrisis period statistics:")
print(f"  Average GDP growth: {crisis_data['gdp_growth_pct'].mean():.2f}%")
print(f"  Average unemployment: {crisis_data['unemployment_rate'].mean():.2f}%")
print(f"  Average asset price change: {crisis_data['asset_price_growth'].mean():.2f}%")
print(f"  Average Ponzi share: {crisis_data['ponzi_share'].mean():.2f}%")
