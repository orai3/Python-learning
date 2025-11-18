"""
Synthetic Macroeconomic Dataset Generator
==========================================
Generates comprehensive national accounts and macroeconomic indicators
with realistic correlations reflecting Post-Keynesian and heterodox economic theory.

Focus: Long-run growth with cyclical fluctuations, crisis episodes, and structural change
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
START_DATE = '1970-Q1'
QUARTERS = 220  # 55 years of quarterly data (1970-2024)
FREQ = 'Q'

# Initialize date range
dates = pd.date_range(start='1970-01-01', periods=QUARTERS, freq='Q')

# ============================================================================
# CORE GROWTH PROCESS
# ============================================================================
# Long-run trend growth (declining over time - stylized fact)
trend_growth = 0.008 + 0.002 * np.exp(-np.arange(QUARTERS) / 80)

# Business cycle component (heterodox: endogenous, not random shocks)
# Goodwin-style cycle: interactions between growth, employment, and distribution
cycle_period = 32  # ~8 year cycles
cycle = 0.015 * np.sin(2 * np.pi * np.arange(QUARTERS) / cycle_period)

# Financial cycle (Minsky: longer than business cycle)
financial_cycle_period = 60  # ~15 year financial cycles
financial_cycle = 0.01 * np.sin(2 * np.pi * np.arange(QUARTERS) / financial_cycle_period)

# Crisis shocks (1975, 1982, 1991, 2001, 2008, 2020)
crisis_periods = [20, 48, 84, 124, 156, 200]
crisis_shock = np.zeros(QUARTERS)
for crisis in crisis_periods:
    if crisis < QUARTERS:
        # Sharp drop and gradual recovery
        crisis_shock[crisis:crisis+1] = -0.03
        crisis_shock[crisis+1:crisis+4] = -0.02
        crisis_shock[crisis+4:crisis+8] = -0.01

# Total GDP growth rate
gdp_growth = trend_growth + cycle + financial_cycle + crisis_shock + np.random.normal(0, 0.003, QUARTERS)

# GDP level (index, 1970 = 100)
gdp_real = np.zeros(QUARTERS)
gdp_real[0] = 100
for t in range(1, QUARTERS):
    gdp_real[t] = gdp_real[t-1] * (1 + gdp_growth[t])

# ============================================================================
# NATIONAL ACCOUNTS COMPONENTS
# ============================================================================
# Consumption (C): ~60-65% of GDP, procyclical but less volatile
consumption_share = 0.62 + 0.02 * np.sin(2 * np.pi * np.arange(QUARTERS) / 50)
consumption_growth = 0.7 * gdp_growth + np.random.normal(0, 0.002, QUARTERS)
consumption_real = np.zeros(QUARTERS)
consumption_real[0] = gdp_real[0] * consumption_share[0]
for t in range(1, QUARTERS):
    consumption_real[t] = consumption_real[t-1] * (1 + consumption_growth[t])

# Investment (I): ~15-25% of GDP, highly volatile and procyclical
# Post-Keynesian: investment drives growth (not savings)
investment_share = 0.20 + 0.03 * cycle + 0.02 * financial_cycle
investment_growth = 2.0 * gdp_growth + 0.5 * financial_cycle + np.random.normal(0, 0.01, QUARTERS)
investment_real = np.zeros(QUARTERS)
investment_real[0] = gdp_real[0] * investment_share[0]
for t in range(1, QUARTERS):
    investment_real[t] = investment_real[t-1] * (1 + investment_growth[t])

# Government spending (G): ~18-22% of GDP, countercyclical
gov_share = 0.20 - 0.01 * cycle  # Increases in recessions
gov_spending_real = gdp_real * gov_share

# Exports (X): ~15-25% of GDP
export_growth = gdp_growth + np.random.normal(0, 0.005, QUARTERS)
exports_real = np.zeros(QUARTERS)
exports_real[0] = gdp_real[0] * 0.18
for t in range(1, QUARTERS):
    exports_real[t] = exports_real[t-1] * (1 + export_growth[t])

# Imports (M): ~15-25% of GDP, income elastic
import_growth = 1.2 * gdp_growth + np.random.normal(0, 0.005, QUARTERS)
imports_real = np.zeros(QUARTERS)
imports_real[0] = gdp_real[0] * 0.17
for t in range(1, QUARTERS):
    imports_real[t] = imports_real[t-1] * (1 + import_growth[t])

# Verify accounting identity: GDP = C + I + G + (X - M)
# Adjust residual to government spending
calculated_gdp = consumption_real + investment_real + gov_spending_real + exports_real - imports_real
residual = gdp_real - calculated_gdp
gov_spending_real = gov_spending_real + residual

# ============================================================================
# LABOUR MARKET
# ============================================================================
# Employment (millions): grows with trend, lags GDP
employment_growth = 0.6 * gdp_growth + np.random.normal(0, 0.002, QUARTERS)
employment = np.zeros(QUARTERS)
employment[0] = 95  # millions
for t in range(1, QUARTERS):
    employment[t] = employment[t-1] * (1 + employment_growth[t])

# Labour force (growing with population)
labor_force_growth = 0.004 + np.random.normal(0, 0.001, QUARTERS)
labor_force = np.zeros(QUARTERS)
labor_force[0] = 100
for t in range(1, QUARTERS):
    labor_force[t] = labor_force[t-1] * (1 + labor_force_growth[t])

# Unemployment rate (%)
unemployment_rate = ((labor_force - employment) / labor_force) * 100
unemployment_rate = np.clip(unemployment_rate, 3, 15)  # Realistic bounds

# Wages (real): grow with productivity, influenced by unemployment (Phillips curve)
# Post-Keynesian: wage share varies with bargaining power
productivity_growth = gdp_growth - employment_growth
wage_growth = productivity_growth - 0.3 * (unemployment_rate - 6) / 6 + np.random.normal(0, 0.003, QUARTERS)
real_wage_index = np.zeros(QUARTERS)
real_wage_index[0] = 100
for t in range(1, QUARTERS):
    real_wage_index[t] = real_wage_index[t-1] * (1 + wage_growth[t])

# Labour productivity (GDP per worker)
productivity = gdp_real / employment * 100

# ============================================================================
# FUNCTIONAL INCOME DISTRIBUTION (Heterodox focus)
# ============================================================================
# Wage share of GDP: varies cyclically (Goodwin model)
# Counter-cyclical with unemployment, pro-cyclical with worker power
wage_share = 0.65 - 0.04 * np.sin(2 * np.pi * np.arange(QUARTERS) / cycle_period) - 0.02 * (unemployment_rate - 6) / 6
wage_share = np.clip(wage_share, 0.55, 0.72)

# Profit share (residual)
profit_share = 1 - wage_share

# Total compensation of employees
total_compensation = gdp_real * wage_share

# Gross operating surplus (profits + rents)
gross_surplus = gdp_real * profit_share

# ============================================================================
# PRICES AND INFLATION
# ============================================================================
# Inflation: cost-push + demand-pull elements
# Post-Keynesian: conflict inflation (wage-price spiral)
base_inflation = 0.008  # Long-run target
demand_pull = 0.5 * cycle
cost_push = 0.3 * wage_growth
inflation_shocks = np.zeros(QUARTERS)
# Oil shocks (1973, 1979)
if 12 < QUARTERS:
    inflation_shocks[12:16] = 0.015  # 1973 oil shock
if 36 < QUARTERS:
    inflation_shocks[36:40] = 0.012  # 1979 oil shock

inflation_rate = base_inflation + demand_pull + cost_push + inflation_shocks + np.random.normal(0, 0.002, QUARTERS)
inflation_rate = np.clip(inflation_rate, -0.02, 0.15)

# CPI index (1970 = 100)
cpi = np.zeros(QUARTERS)
cpi[0] = 100
for t in range(1, QUARTERS):
    cpi[t] = cpi[t-1] * (1 + inflation_rate[t])

# PPI (more volatile, leads CPI)
ppi_inflation = inflation_rate * 1.2 + np.random.normal(0, 0.003, QUARTERS)
ppi = np.zeros(QUARTERS)
ppi[0] = 100
for t in range(1, QUARTERS):
    ppi[t] = ppi[t-1] * (1 + ppi_inflation[t])

# Unit labour costs (wages relative to productivity)
unit_labor_cost = (real_wage_index / productivity) * 100

# GDP deflator
gdp_deflator = cpi * 0.95 + ppi * 0.05

# Nominal GDP
gdp_nominal = gdp_real * (gdp_deflator / 100)

# ============================================================================
# FINANCIAL SECTOR (Heterodox: financialization over time)
# ============================================================================
# Private debt to GDP ratio (rising trend = financialization)
debt_trend = 0.8 + 0.6 * (1 - np.exp(-np.arange(QUARTERS) / 60))  # Asymptotes to 1.4
debt_cycle = 0.15 * np.sin(2 * np.pi * np.arange(QUARTERS) / financial_cycle_period)
private_debt_gdp = debt_trend + debt_cycle + np.random.normal(0, 0.02, QUARTERS)
private_debt_gdp = np.clip(private_debt_gdp, 0.5, 2.0)

# Total private debt
private_debt = gdp_nominal * private_debt_gdp

# Credit growth (leads investment)
credit_growth = 1.5 * investment_growth + financial_cycle + np.random.normal(0, 0.008, QUARTERS)

# Asset prices (real estate and equity, bubbles aligned with financial cycle)
asset_price_growth = gdp_growth + 1.5 * financial_cycle + 0.5 * credit_growth + np.random.normal(0, 0.01, QUARTERS)
asset_price_index = np.zeros(QUARTERS)
asset_price_index[0] = 100
for t in range(1, QUARTERS):
    asset_price_index[t] = asset_price_index[t-1] * (1 + asset_price_growth[t])

# Interest rates (monetary policy responds to inflation and output gap)
output_gap = (gdp_real / (100 * (1.025 ** np.arange(QUARTERS))) - 1) * 100
policy_rate = 2 + 1.5 * inflation_rate * 100 + 0.5 * output_gap + np.random.normal(0, 0.3, QUARTERS)
policy_rate = np.clip(policy_rate, 0, 20)

# Corporate profits (related to profit share and financial conditions)
corporate_profits = gross_surplus * 0.6 * (1 + 0.3 * financial_cycle)

# Financial sector size (% of GDP - financialization)
finance_sector_share = 0.05 + 0.05 * (1 - np.exp(-np.arange(QUARTERS) / 80))
finance_sector_gdp = gdp_nominal * finance_sector_share

# ============================================================================
# GOVERNMENT AND EXTERNAL SECTOR
# ============================================================================
# Government revenue (taxes - procyclical)
tax_rate = 0.30 + 0.02 * (gdp_growth / 0.01)
gov_revenue = gdp_nominal * tax_rate

# Government spending (nominal)
gov_spending_nominal = gov_spending_real * (gdp_deflator / 100)

# Budget balance (deficit negative)
budget_balance = gov_revenue - gov_spending_nominal
budget_balance_gdp = (budget_balance / gdp_nominal) * 100

# Government debt (accumulation of deficits)
gov_debt = np.zeros(QUARTERS)
gov_debt[0] = gdp_nominal[0] * 0.4
for t in range(1, QUARTERS):
    gov_debt[t] = gov_debt[t-1] - budget_balance[t]

gov_debt_gdp = (gov_debt / gdp_nominal) * 100

# Trade balance (X - M)
trade_balance = exports_real - imports_real
trade_balance_nominal = trade_balance * (gdp_deflator / 100)
trade_balance_gdp = (trade_balance_nominal / gdp_nominal) * 100

# Current account (includes investment income)
net_foreign_assets = np.zeros(QUARTERS)
net_foreign_assets[0] = -gdp_nominal[0] * 0.1
for t in range(1, QUARTERS):
    net_foreign_assets[t] = net_foreign_assets[t-1] + trade_balance_nominal[t]

investment_income = net_foreign_assets * 0.02  # 2% return
current_account = trade_balance_nominal + investment_income
current_account_gdp = (current_account / gdp_nominal) * 100

# Capital flows (residual, balances current account)
capital_inflows = -current_account

# ============================================================================
# CONSTRUCT DATAFRAME
# ============================================================================
data = pd.DataFrame({
    # Time
    'date': dates,
    'year': dates.year,
    'quarter': dates.quarter,

    # GDP and components (real, billions)
    'gdp_real': gdp_real,
    'consumption_real': consumption_real,
    'investment_real': investment_real,
    'gov_spending_real': gov_spending_real,
    'exports_real': exports_real,
    'imports_real': imports_real,

    # GDP growth rates (quarterly, decimal)
    'gdp_growth_rate': gdp_growth,
    'consumption_growth_rate': consumption_growth,
    'investment_growth_rate': investment_growth,

    # Nominal values (billions)
    'gdp_nominal': gdp_nominal,
    'total_compensation': total_compensation,
    'gross_operating_surplus': gross_surplus,

    # Labour market
    'employment': employment,  # millions
    'labor_force': labor_force,
    'unemployment_rate': unemployment_rate,  # percent
    'real_wage_index': real_wage_index,
    'productivity_index': productivity,

    # Functional distribution
    'wage_share': wage_share,
    'profit_share': profit_share,

    # Prices
    'cpi': cpi,
    'ppi': ppi,
    'gdp_deflator': gdp_deflator,
    'inflation_rate': inflation_rate * 100,  # percent
    'unit_labor_cost': unit_labor_cost,

    # Financial sector
    'private_debt': private_debt,
    'private_debt_gdp_ratio': private_debt_gdp,
    'credit_growth_rate': credit_growth,
    'asset_price_index': asset_price_index,
    'policy_interest_rate': policy_rate,
    'corporate_profits': corporate_profits,
    'finance_sector_value': finance_sector_gdp,
    'finance_sector_share': finance_sector_share * 100,

    # Government sector
    'gov_revenue': gov_revenue,
    'gov_spending': gov_spending_nominal,
    'budget_balance': budget_balance,
    'budget_balance_gdp': budget_balance_gdp,
    'gov_debt': gov_debt,
    'gov_debt_gdp': gov_debt_gdp,

    # External sector
    'trade_balance': trade_balance_nominal,
    'trade_balance_gdp': trade_balance_gdp,
    'current_account': current_account,
    'current_account_gdp': current_account_gdp,
    'net_foreign_assets': net_foreign_assets,
    'capital_inflows': capital_inflows,
})

# Save to CSV
data.to_csv('/home/user/Python-learning/datasets/macro_quarterly_data.csv', index=False)

print(f"Generated macroeconomic dataset with {len(data)} observations")
print(f"Date range: {data['date'].min()} to {data['date'].max()}")
print(f"\nSample statistics:")
print(f"Average GDP growth: {data['gdp_growth_rate'].mean()*100:.2f}% per quarter")
print(f"Average unemployment: {data['unemployment_rate'].mean():.2f}%")
print(f"Average wage share: {data['wage_share'].mean()*100:.2f}%")
print(f"Average inflation: {data['inflation_rate'].mean():.2f}%")
print(f"Final private debt/GDP: {data['private_debt_gdp_ratio'].iloc[-1]:.2f}")
