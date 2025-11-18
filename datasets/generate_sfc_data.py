"""
Synthetic Sectoral Balances Dataset (Stock-Flow Consistent)
============================================================
Generates Godley-style sectoral balances following Post-Keynesian SFC approach.

Sectors:
- Households
- Non-financial corporations
- Financial corporations
- Government
- Rest of World (foreign sector)

Key principle: Sum of sectoral balances = 0 (one sector's surplus = another's deficit)
Shows financial flows between sectors and stock accumulation over time.
"""

import numpy as np
import pandas as pd

np.random.seed(44)

# Configuration
START_YEAR = 1970
YEARS = 55
years = np.arange(START_YEAR, START_YEAR + YEARS)

# Initialize GDP (baseline for scaling)
gdp = 100 * (1.025 ** np.arange(YEARS))  # 2.5% average growth

# ============================================================================
# INCOME AND EXPENDITURE BY SECTOR
# ============================================================================

# HOUSEHOLDS
# ----------
# Income: wages + transfers + interest/dividends
household_wage_income = gdp * (0.55 - 0.05 * (1 - np.exp(-np.arange(YEARS) / 25)))
household_transfer_income = gdp * (0.10 + 0.02 * (1 - np.exp(-np.arange(YEARS) / 30)))
household_capital_income = gdp * (0.03 + 0.05 * (1 - np.exp(-np.arange(YEARS) / 20)))

household_total_income = household_wage_income + household_transfer_income + household_capital_income

# Expenditure: consumption + interest paid
household_consumption = gdp * 0.60
household_interest_paid = gdp * (0.01 + 0.02 * (1 - np.exp(-np.arange(YEARS) / 25)))

household_total_expenditure = household_consumption + household_interest_paid

# Net lending/borrowing (saving)
household_balance = household_total_income - household_total_expenditure

# NON-FINANCIAL CORPORATIONS
# --------------------------
# Income: sales - wages - interest
nfc_sales = gdp * 1.0  # Approximately equal to GDP
nfc_wage_costs = household_wage_income * 0.85  # Most wages from corporations
nfc_interest_paid = gdp * (0.02 + 0.03 * (1 - np.exp(-np.arange(YEARS) / 20)))

nfc_gross_profits = nfc_sales - nfc_wage_costs - nfc_interest_paid

# Expenditure: investment + dividends
nfc_investment = gdp * (0.18 + 0.03 * np.sin(2 * np.pi * np.arange(YEARS) / 8))
nfc_dividends_paid = nfc_gross_profits * 0.4

nfc_total_expenditure = nfc_investment + nfc_dividends_paid

# Net lending/borrowing (mostly borrowing for investment)
nfc_balance = nfc_gross_profits - nfc_total_expenditure

# FINANCIAL CORPORATIONS
# ----------------------
# Income: interest received + fees
fc_interest_received = household_interest_paid + nfc_interest_paid
fc_fees = gdp * (0.01 + 0.02 * (1 - np.exp(-np.arange(YEARS) / 25)))  # Rising with financialization

fc_total_income = fc_interest_received + fc_fees

# Expenditure: interest paid + dividends
fc_interest_paid = fc_interest_received * 0.4
fc_dividends_paid = (fc_total_income - fc_interest_paid) * 0.6

fc_total_expenditure = fc_interest_paid + fc_dividends_paid

# Net lending/borrowing (small, intermediary role)
fc_balance = fc_total_income - fc_total_expenditure

# GOVERNMENT
# ----------
# Income: taxes
gov_tax_revenue = gdp * (0.28 + 0.02 * np.sin(2 * np.pi * np.arange(YEARS) / 12))

# Expenditure: spending + transfers + interest on debt
gov_spending = gdp * 0.20
gov_transfers = household_transfer_income
gov_interest_paid = gdp * (0.02 + 0.03 * (1 - np.exp(-np.arange(YEARS) / 20)))

gov_total_expenditure = gov_spending + gov_transfers + gov_interest_paid

# Net lending/borrowing (usually deficit)
gov_balance = gov_tax_revenue - gov_total_expenditure

# REST OF WORLD (Foreign Sector)
# ------------------------------
# Net exports + net foreign income
exports = gdp * (0.15 + 0.05 * np.sin(2 * np.pi * np.arange(YEARS) / 10))
imports = gdp * (0.16 + 0.06 * np.sin(2 * np.pi * np.arange(YEARS) / 10))

net_exports = exports - imports

net_foreign_income = gdp * (-0.01 - 0.02 * (1 - np.exp(-np.arange(YEARS) / 25)))

# Foreign balance (current account, sign reversed)
# Negative = domestic deficit = foreign surplus
foreign_balance = -(net_exports + net_foreign_income)

# ============================================================================
# VERIFY ACCOUNTING IDENTITY
# ============================================================================
# Sum of all sectoral balances must equal zero
total_balance = household_balance + nfc_balance + fc_balance + gov_balance + foreign_balance

# Adjust government balance to ensure identity holds (absorbs rounding errors)
gov_balance = gov_balance - total_balance

# Recalculate government expenditure
gov_total_expenditure = gov_tax_revenue - gov_balance

# ============================================================================
# CUMULATIVE STOCKS (Net Financial Assets by Sector)
# ============================================================================
# Cumulative sum of flows = stock positions
household_nfa = np.cumsum(household_balance)
nfc_nfa = np.cumsum(nfc_balance)
fc_nfa = np.cumsum(fc_balance)
gov_nfa = np.cumsum(gov_balance)  # Negative = government debt
foreign_nfa = np.cumsum(foreign_balance)  # Negative = net foreign assets of domestic economy

# ============================================================================
# FINANCIAL STOCKS AND RATIOS
# ============================================================================
# Household net worth (assets - liabilities)
household_financial_assets = household_nfa + gdp * 2.0  # Gross position
household_liabilities = gdp * (0.5 + 0.6 * (1 - np.exp(-np.arange(YEARS) / 20)))
household_net_worth = household_financial_assets - household_liabilities

# Corporate debt
corporate_debt = -nfc_nfa  # Negative NFA = net debtor
corporate_debt_gdp_ratio = (corporate_debt / gdp) * 100

# Government debt
government_debt = -gov_nfa
government_debt_gdp_ratio = (government_debt / gdp) * 100

# Foreign debt
net_foreign_debt = -foreign_nfa  # Negative foreign NFA = domestic net foreign assets
net_foreign_debt_gdp_ratio = (net_foreign_debt / gdp) * 100

# Total private debt (household + corporate)
private_debt = household_liabilities + corporate_debt
private_debt_gdp_ratio = (private_debt / gdp) * 100

# ============================================================================
# SAVING AND INVESTMENT BALANCES
# ============================================================================
# Private sector balance (households + corporations)
private_balance = household_balance + nfc_balance + fc_balance

# Domestic private balance (excluding foreign)
domestic_balance = household_balance + nfc_balance + fc_balance + gov_balance

# Sectoral balances as % of GDP
household_balance_gdp = (household_balance / gdp) * 100
nfc_balance_gdp = (nfc_balance / gdp) * 100
fc_balance_gdp = (fc_balance / gdp) * 100
gov_balance_gdp = (gov_balance / gdp) * 100
foreign_balance_gdp = (foreign_balance / gdp) * 100
private_balance_gdp = (private_balance / gdp) * 100

# ============================================================================
# LEVERAGE RATIOS
# ============================================================================
# Household debt service ratio (interest paid / income)
household_debt_service_ratio = (household_interest_paid / household_total_income) * 100

# Corporate leverage (debt / equity, approximated)
corporate_equity = nfc_gross_profits * 15  # Stylized
corporate_leverage = corporate_debt / corporate_equity

# ============================================================================
# KALECKI PROFIT EQUATION
# ============================================================================
# Profits = Investment + Government Deficit + Trade Surplus - Household Saving
# This is a key Post-Keynesian identity

kalecki_profits = nfc_investment - gov_balance + net_exports - household_balance
# Should approximately equal nfc_gross_profits (may have small discrepancies)

# ============================================================================
# CONSTRUCT DATAFRAME
# ============================================================================
data = pd.DataFrame({
    'year': years,
    'gdp': gdp,

    # FLOWS (annual, in billions)
    # Household sector
    'hh_wage_income': household_wage_income,
    'hh_transfer_income': household_transfer_income,
    'hh_capital_income': household_capital_income,
    'hh_total_income': household_total_income,
    'hh_consumption': household_consumption,
    'hh_interest_paid': household_interest_paid,
    'hh_total_expenditure': household_total_expenditure,
    'hh_balance': household_balance,
    'hh_balance_gdp_pct': household_balance_gdp,

    # Corporate sector
    'nfc_sales': nfc_sales,
    'nfc_wages': nfc_wage_costs,
    'nfc_interest_paid': nfc_interest_paid,
    'nfc_gross_profits': nfc_gross_profits,
    'nfc_investment': nfc_investment,
    'nfc_dividends': nfc_dividends_paid,
    'nfc_balance': nfc_balance,
    'nfc_balance_gdp_pct': nfc_balance_gdp,

    # Financial sector
    'fc_interest_received': fc_interest_received,
    'fc_fees': fc_fees,
    'fc_total_income': fc_total_income,
    'fc_interest_paid': fc_interest_paid,
    'fc_dividends': fc_dividends_paid,
    'fc_balance': fc_balance,
    'fc_balance_gdp_pct': fc_balance_gdp,

    # Government sector
    'gov_tax_revenue': gov_tax_revenue,
    'gov_spending': gov_spending,
    'gov_transfers': gov_transfers,
    'gov_interest_paid': gov_interest_paid,
    'gov_total_expenditure': gov_total_expenditure,
    'gov_balance': gov_balance,
    'gov_balance_gdp_pct': gov_balance_gdp,

    # Foreign sector
    'exports': exports,
    'imports': imports,
    'net_exports': net_exports,
    'net_foreign_income': net_foreign_income,
    'foreign_balance': foreign_balance,
    'foreign_balance_gdp_pct': foreign_balance_gdp,

    # Aggregate balances
    'private_balance': private_balance,
    'private_balance_gdp_pct': private_balance_gdp,

    # STOCKS (cumulative, in billions)
    'hh_net_financial_assets': household_nfa,
    'nfc_net_financial_assets': nfc_nfa,
    'fc_net_financial_assets': fc_nfa,
    'gov_net_financial_assets': gov_nfa,
    'foreign_net_financial_assets': foreign_nfa,

    # Debt stocks
    'household_liabilities': household_liabilities,
    'corporate_debt': corporate_debt,
    'government_debt': government_debt,
    'private_debt': private_debt,

    # Debt ratios (% of GDP)
    'corporate_debt_gdp': corporate_debt_gdp_ratio,
    'government_debt_gdp': government_debt_gdp_ratio,
    'private_debt_gdp': private_debt_gdp_ratio,
    'net_foreign_debt_gdp': net_foreign_debt_gdp_ratio,

    # Other ratios
    'hh_debt_service_ratio': household_debt_service_ratio,
    'corporate_leverage': corporate_leverage,

    # Kalecki profits
    'kalecki_profits': kalecki_profits,
})

# Save to CSV
data.to_csv('/home/user/Python-learning/datasets/sectoral_balances_data.csv', index=False)

print(f"Generated SFC sectoral balances dataset with {len(data)} observations")
print(f"Years: {data['year'].min()} to {data['year'].max()}\n")

print("Sectoral balance verification (should sum to ~0):")
print(f"Final year sectoral balances (% of GDP):")
print(f"  Household: {data['hh_balance_gdp_pct'].iloc[-1]:.2f}%")
print(f"  Corporate: {data['nfc_balance_gdp_pct'].iloc[-1]:.2f}%")
print(f"  Financial: {data['fc_balance_gdp_pct'].iloc[-1]:.2f}%")
print(f"  Government: {data['gov_balance_gdp_pct'].iloc[-1]:.2f}%")
print(f"  Foreign: {data['foreign_balance_gdp_pct'].iloc[-1]:.2f}%")
total = (data['hh_balance_gdp_pct'].iloc[-1] + data['nfc_balance_gdp_pct'].iloc[-1] +
         data['fc_balance_gdp_pct'].iloc[-1] + data['gov_balance_gdp_pct'].iloc[-1] +
         data['foreign_balance_gdp_pct'].iloc[-1])
print(f"  Total: {total:.2f}% (should be ~0)")

print(f"\nDebt ratios in final year:")
print(f"  Private debt/GDP: {data['private_debt_gdp'].iloc[-1]:.1f}%")
print(f"  Government debt/GDP: {data['government_debt_gdp'].iloc[-1]:.1f}%")
print(f"  Corporate leverage: {data['corporate_leverage'].iloc[-1]:.2f}")
