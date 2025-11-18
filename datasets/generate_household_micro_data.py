"""
Synthetic Household-Level Microdata Generator
==============================================
Generates detailed household-level data with realistic heterogeneity.

Includes:
- Income (wages, capital, transfers) with realistic correlations
- Wealth (housing, financial, business)
- Consumption and saving behavior
- Demographics (age, education, household size, region)
- Labor market status
- Debt and financial vulnerability

Reflects realistic inequality and heterogeneity across households.
"""

import numpy as np
import pandas as pd

np.random.seed(45)

# Configuration
N_HOUSEHOLDS = 50000  # Large micro dataset
YEAR = 2023

print(f"Generating {N_HOUSEHOLDS:,} household records...")

# ============================================================================
# DEMOGRAPHICS
# ============================================================================

# Age of household head (20-90)
age = np.random.beta(2, 2, N_HOUSEHOLDS) * 70 + 20
age = age.astype(int)

# Household size (1-6 persons)
# Younger households tend to be smaller
household_size_prob = np.exp(-(age - 30)**2 / 500)  # Peak at age 30
household_size = np.random.poisson(1.5 + household_size_prob, N_HOUSEHOLDS) + 1
household_size = np.clip(household_size, 1, 6)

# Number of children
children = np.where(age < 50,
                   np.random.poisson(0.8, N_HOUSEHOLDS),
                   np.random.poisson(0.1, N_HOUSEHOLDS))
children = np.clip(children, 0, household_size - 1)

# Education level (0=less than HS, 1=HS, 2=Some college, 3=Bachelor's, 4=Graduate)
# Correlated with age cohort (younger = more educated)
education_base = np.random.choice([0, 1, 2, 3, 4],
                                 N_HOUSEHOLDS,
                                 p=[0.10, 0.25, 0.30, 0.25, 0.10])

# Younger cohorts more educated
age_education_boost = np.where(age < 40, 1, 0) * (np.random.rand(N_HOUSEHOLDS) > 0.5).astype(int)
education = np.clip(education_base + age_education_boost, 0, 4)

# Region (1=Urban, 2=Suburban, 3=Rural)
region = np.random.choice([1, 2, 3],
                         N_HOUSEHOLDS,
                         p=[0.35, 0.45, 0.20])

# Race/ethnicity (1=White, 2=Black, 3=Hispanic, 4=Asian, 5=Other)
race = np.random.choice([1, 2, 3, 4, 5],
                       N_HOUSEHOLDS,
                       p=[0.60, 0.13, 0.18, 0.06, 0.03])

# Gender of household head
gender = np.random.choice([0, 1], N_HOUSEHOLDS, p=[0.55, 0.45])  # 0=Male, 1=Female

# ============================================================================
# LABOR MARKET STATUS
# ============================================================================

# Employment probability (depends on age and education)
employment_prob = 0.6 + 0.1 * education - 0.01 * np.abs(age - 45)
employment_prob = np.clip(employment_prob, 0.3, 0.95)
employed = (np.random.rand(N_HOUSEHOLDS) < employment_prob).astype(int)

# Occupation (for employed, higher = higher skill/pay)
# 1=Service, 2=Manual, 3=Clerical, 4=Professional, 5=Management
occupation = np.zeros(N_HOUSEHOLDS, dtype=int)
occupation[employed == 1] = np.random.choice([1, 2, 3, 4, 5],
                                             employed.sum(),
                                             p=[0.20, 0.20, 0.25, 0.25, 0.10])

# Occupation correlated with education
high_ed = education >= 3
high_ed_employed = (employed == 1) & high_ed
occupation[high_ed_employed] = np.random.choice([3, 4, 5],
                                                high_ed_employed.sum(),
                                                p=[0.2, 0.5, 0.3])

# Hours worked per week
hours = np.zeros(N_HOUSEHOLDS)
hours[employed == 1] = np.random.gamma(15, 2.5, employed.sum())
hours = np.clip(hours, 0, 80)

# ============================================================================
# INCOME
# ============================================================================

# Base wage (depends on education, occupation, age)
base_wage_hourly = (10 + 5 * education +
                   3 * (occupation - 1) +
                   0.5 * np.clip(age - 25, 0, 30) -  # Experience premium
                   0.01 * np.clip(age - 55, 0, 100))  # Decline after 55

# Gender wage gap
base_wage_hourly = np.where(gender == 1,
                           base_wage_hourly * 0.82,  # Women earn 82% of men
                           base_wage_hourly)

# Racial wage gap
race_multiplier = np.ones(N_HOUSEHOLDS)
race_multiplier[race == 2] = 0.75  # Black
race_multiplier[race == 3] = 0.78  # Hispanic
race_multiplier[race == 4] = 1.05  # Asian
base_wage_hourly = base_wage_hourly * race_multiplier

# Add randomness
base_wage_hourly = base_wage_hourly * np.random.lognormal(0, 0.3, N_HOUSEHOLDS)
base_wage_hourly = np.clip(base_wage_hourly, 10, 300)

# Annual wage income
wage_income = base_wage_hourly * hours * 52  # Weekly hours * 52 weeks
wage_income[employed == 0] = 0

# Self-employment income (some households)
self_employed = (np.random.rand(N_HOUSEHOLDS) < 0.12).astype(int)
self_employment_income = np.where(self_employed,
                                 np.random.lognormal(10.5, 1.2, N_HOUSEHOLDS),
                                 0)

# Capital income (interest, dividends, rent)
# Highly concentrated in top of distribution
capital_income_base = np.random.pareto(1.5, N_HOUSEHOLDS) * 1000
# Correlated with age (wealth accumulation)
capital_income = capital_income_base * (1 + 0.02 * np.clip(age - 30, 0, 100))
capital_income = np.clip(capital_income, 0, 5000000)

# Transfer income (Social Security, unemployment, welfare)
# More for elderly, unemployed, low-income
transfer_income = np.zeros(N_HOUSEHOLDS)

# Social Security (elderly)
transfer_income += np.where(age >= 65,
                           np.random.normal(20000, 5000, N_HOUSEHOLDS),
                           0)

# Unemployment benefits
transfer_income += np.where((employed == 0) & (age < 65) & (age > 25),
                           np.random.normal(8000, 2000, N_HOUSEHOLDS),
                           0)

# Other transfers (disability, welfare, child benefits)
transfer_income += np.where(wage_income < 30000,
                           np.random.exponential(3000, N_HOUSEHOLDS),
                           0)

transfer_income = np.clip(transfer_income, 0, 60000)

# Total income
total_income = wage_income + self_employment_income + capital_income + transfer_income

# ============================================================================
# WEALTH
# ============================================================================

# Net worth highly correlated with income and age
# More unequal than income

# Housing wealth
homeowner = (np.random.rand(N_HOUSEHOLDS) < 0.65).astype(int)
# Homeownership increases with income and age
homeownership_prob = 0.3 + 0.00001 * total_income + 0.005 * age
homeowner = (np.random.rand(N_HOUSEHOLDS) < homeownership_prob).astype(int)

home_value = np.where(homeowner,
                     np.random.lognormal(12.4, 0.6, N_HOUSEHOLDS),  # Median ~$250k
                     0)

# Mortgage debt (for homeowners)
mortgage_debt = np.where(homeowner,
                        home_value * np.random.beta(2, 2, N_HOUSEHOLDS) * 0.7,
                        0)

# Net home equity
home_equity = home_value - mortgage_debt

# Financial assets (savings, stocks, bonds)
# Very concentrated
financial_assets = np.random.pareto(2.0, N_HOUSEHOLDS) * total_income * 0.5
financial_assets = financial_assets * (1 + 0.02 * age)  # Accumulates with age
financial_assets = np.clip(financial_assets, 0, 50000000)

# Business equity (for some households)
business_owner = (np.random.rand(N_HOUSEHOLDS) < 0.10).astype(int)
business_equity = np.where(business_owner,
                          np.random.lognormal(11, 1.5, N_HOUSEHOLDS),
                          0)

# Consumer debt (credit cards, auto loans, student loans)
consumer_debt_prob = 0.5 - 0.00001 * total_income
consumer_debt_prob = np.clip(consumer_debt_prob, 0.2, 0.8)
has_consumer_debt = (np.random.rand(N_HOUSEHOLDS) < consumer_debt_prob).astype(int)

consumer_debt = np.where(has_consumer_debt,
                        np.random.lognormal(9, 1, N_HOUSEHOLDS),  # Median ~$8k
                        0)

# Student loan debt (younger, educated households)
student_debt_prob = np.clip((50 - age) / 40 * (education - 1) / 3, 0, 0.7)
has_student_debt = (np.random.rand(N_HOUSEHOLDS) < student_debt_prob).astype(int)

student_debt = np.where(has_student_debt,
                       np.random.lognormal(10.3, 0.7, N_HOUSEHOLDS),  # Median ~$30k
                       0)

# Total assets and liabilities
total_assets = home_equity + financial_assets + business_equity
total_liabilities = mortgage_debt + consumer_debt + student_debt
net_worth = total_assets - total_liabilities

# ============================================================================
# CONSUMPTION AND SAVING
# ============================================================================

# Consumption (depends on income, with diminishing MPC)
# Lower income = higher marginal propensity to consume
mpc = 0.95 - 0.00001 * total_income
mpc = np.clip(mpc, 0.4, 0.98)

consumption = total_income * mpc + np.random.normal(0, 2000, N_HOUSEHOLDS)
consumption = np.clip(consumption, 1000, total_income * 1.1)  # Can borrow

# Saving (residual)
saving = total_income - consumption

# Saving rate
saving_rate = np.where(total_income > 0,
                      (saving / total_income) * 100,
                      0)

# ============================================================================
# FINANCIAL STRESS INDICATORS
# ============================================================================

# Debt-to-income ratio
debt_to_income = np.where(total_income > 0,
                         (total_liabilities / total_income) * 100,
                         999)  # Flag infinite

# Liquid asset poverty (less than 3 months expenses)
monthly_expenses = consumption / 12
liquid_asset_poor = (financial_assets < monthly_expenses * 3).astype(int)

# Housing cost burden (>30% of income on housing)
housing_costs = mortgage_debt * 0.05 + home_value * 0.01  # Mortgage payment + maintenance
housing_cost_burden = np.where(total_income > 0,
                              (housing_costs / total_income) * 100,
                              0)
housing_burdened = (housing_cost_burden > 30).astype(int)

# ============================================================================
# DERIVE ADDITIONAL VARIABLES
# ============================================================================

# Income quintile
income_quintile = pd.qcut(total_income, q=5, labels=[1, 2, 3, 4, 5])

# Wealth quintile
wealth_quintile = pd.qcut(net_worth, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')

# Income class (based on median)
median_income = np.median(total_income)
income_class = np.where(total_income < median_income * 0.67, 1,  # Lower
                       np.where(total_income < median_income * 1.5, 2,  # Middle
                               3))  # Upper

# ============================================================================
# CONSTRUCT DATAFRAME
# ============================================================================
data = pd.DataFrame({
    'household_id': np.arange(1, N_HOUSEHOLDS + 1),
    'year': YEAR,

    # Demographics
    'age': age,
    'household_size': household_size,
    'children': children,
    'education': education,
    'region': region,
    'race': race,
    'gender': gender,

    # Labor market
    'employed': employed,
    'occupation': occupation,
    'hours_worked': hours,
    'self_employed': self_employed,

    # Income
    'wage_income': wage_income.round(2),
    'self_employment_income': self_employment_income.round(2),
    'capital_income': capital_income.round(2),
    'transfer_income': transfer_income.round(2),
    'total_income': total_income.round(2),

    # Wealth
    'homeowner': homeowner,
    'home_value': home_value.round(2),
    'mortgage_debt': mortgage_debt.round(2),
    'home_equity': home_equity.round(2),
    'financial_assets': financial_assets.round(2),
    'business_owner': business_owner,
    'business_equity': business_equity.round(2),
    'consumer_debt': consumer_debt.round(2),
    'student_debt': student_debt.round(2),
    'total_assets': total_assets.round(2),
    'total_liabilities': total_liabilities.round(2),
    'net_worth': net_worth.round(2),

    # Consumption and saving
    'consumption': consumption.round(2),
    'saving': saving.round(2),
    'saving_rate': saving_rate.round(2),

    # Financial stress
    'debt_to_income_ratio': debt_to_income.round(2),
    'liquid_asset_poor': liquid_asset_poor,
    'housing_cost_burden': housing_cost_burden.round(2),
    'housing_burdened': housing_burdened,

    # Classification
    'income_quintile': income_quintile,
    'wealth_quintile': wealth_quintile,
    'income_class': income_class,
})

# Save to CSV
data.to_csv('/home/user/Python-learning/datasets/household_microdata.csv', index=False)

print(f"✓ Generated household microdata with {len(data):,} observations\n")

print("Summary statistics:")
print(f"Median income: ${data['total_income'].median():,.0f}")
print(f"Mean income: ${data['total_income'].mean():,.0f}")
print(f"Median net worth: ${data['net_worth'].median():,.0f}")
print(f"Mean net worth: ${data['net_worth'].mean():,.0f}")
print(f"\nEmployment rate: {data['employed'].mean()*100:.1f}%")
print(f"Homeownership rate: {data['homeowner'].mean()*100:.1f}%")
print(f"Business ownership rate: {data['business_owner'].mean()*100:.1f}%")
print(f"\nLiquid asset poverty rate: {data['liquid_asset_poor'].mean()*100:.1f}%")
print(f"Housing cost burdened: {data['housing_burdened'].mean()*100:.1f}%")
print(f"\nTop 1% income share: {data.nlargest(int(N_HOUSEHOLDS*0.01), 'total_income')['total_income'].sum() / data['total_income'].sum() * 100:.1f}%")
print(f"Top 1% wealth share: {data.nlargest(int(N_HOUSEHOLDS*0.01), 'net_worth')['net_worth'].sum() / data['net_worth'].sum() * 100:.1f}%")
