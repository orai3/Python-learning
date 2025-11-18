# Comprehensive Data Dictionary
## Synthetic Economic Datasets for Heterodox Analysis

This directory contains **7 major synthetic datasets** designed for heterodox economic analysis, with emphasis on Post-Keynesian, Marxian, and Comparative Political Economy approaches.

**Total observations across all datasets:** ~102,000+ records
**Generation date:** 2025-11-18
**Random seeds:** Fixed for reproducibility

---

## 📊 Dataset Overview

| Dataset | File | Observations | Frequency | Period | Focus |
|---------|------|--------------|-----------|--------|-------|
| Macroeconomic | `macro_quarterly_data.csv` | 220 | Quarterly | 1970-2024 | National accounts, labor, finance |
| Inequality | `inequality_annual_data.csv` | 55 | Annual | 1970-2024 | Distribution, class shares |
| Sectoral Balances | `sectoral_balances_data.csv` | 55 | Annual | 1970-2024 | SFC framework, Godley-style |
| Household Micro | `household_microdata.csv` | 50,000 | Cross-section | 2023 | Individual households |
| Panel Data | `cross_country_panel_data.csv` | 1,650 | Annual | 1970-2024 | 30 countries, CPE analysis |
| Financial Crisis | `financial_crisis_data.csv` | 140 | Quarterly | 1990-2024 | Minsky framework |
| Energy & Environment | `energy_environment_data.csv` | 55 | Annual | 1970-2024 | Green transition |

---

## 1. Macroeconomic Dataset

**File:** `macro_quarterly_data.csv`
**Observations:** 220 quarters (1970Q1 - 2024Q4)
**Generator:** `generate_macro_data.py`

### Theoretical Framework
- Post-Keynesian growth dynamics (endogenous cycles)
- Goodwin model (distribution-growth cycles)
- Minsky's financial instability hypothesis
- Kalecki's profit equation

### Key Features
- **Endogenous cycles:** Business cycles emerge from interactions, not external shocks
- **Financial cycles:** Longer-period cycles (15 years) in credit and asset prices
- **Crisis episodes:** 1975, 1982, 1991, 2001, 2008, 2020
- **Financialization:** Rising private debt/GDP ratio over time
- **Declining wage share:** Neoliberal trend

### Variable Groups

#### GDP and Components (real, billions)
- `gdp_real`: Real GDP (index, 1970=100)
- `consumption_real`: Real consumption
- `investment_real`: Real investment (highly volatile, procyclical)
- `gov_spending_real`: Real government spending (countercyclical)
- `exports_real`: Real exports
- `imports_real`: Real imports

#### Growth Rates (quarterly, decimal)
- `gdp_growth_rate`: Quarterly GDP growth
- `consumption_growth_rate`: Consumption growth (less volatile than GDP)
- `investment_growth_rate`: Investment growth (2x GDP volatility)

#### Labor Market
- `employment`: Employment (millions)
- `labor_force`: Labor force (millions)
- `unemployment_rate`: Unemployment rate (%)
- `real_wage_index`: Real wage index (1970=100)
- `productivity_index`: Labor productivity (GDP per worker)

#### Functional Distribution (Heterodox focus)
- `wage_share`: Wage share of GDP (%) - counter-cyclical
- `profit_share`: Profit share of GDP (%)
- Note: Varies with Goodwin cycles (8-year period)

#### Prices
- `cpi`: Consumer Price Index (1970=100)
- `ppi`: Producer Price Index (1970=100)
- `gdp_deflator`: GDP deflator
- `inflation_rate`: Inflation rate (%)
- `unit_labor_cost`: Unit labor costs (wages/productivity)

#### Financial Sector (Financialization)
- `private_debt`: Total private sector debt (billions)
- `private_debt_gdp_ratio`: Private debt as % of GDP (rising trend)
- `credit_growth_rate`: Credit growth rate (leads investment)
- `asset_price_index`: Asset price index (real estate + equity)
- `policy_interest_rate`: Central bank policy rate (%)
- `corporate_profits`: Corporate profits (billions)
- `finance_sector_value`: Financial sector value added
- `finance_sector_share`: Finance as % of GDP (rising)

#### Government Sector
- `gov_revenue`: Government revenue (billions)
- `gov_spending`: Government spending (billions)
- `budget_balance`: Budget balance (billions, negative = deficit)
- `budget_balance_gdp`: Budget balance as % of GDP
- `gov_debt`: Government debt stock (billions)
- `gov_debt_gdp`: Government debt as % of GDP

#### External Sector
- `trade_balance`: Trade balance (billions)
- `trade_balance_gdp`: Trade balance as % of GDP
- `current_account`: Current account balance (billions)
- `current_account_gdp`: Current account as % of GDP
- `net_foreign_assets`: Net foreign assets (cumulative)
- `capital_inflows`: Capital inflows (billions)

### Methodological Notes
1. **Accounting identity enforced:** GDP ≡ C + I + G + (X - M)
2. **Crisis modeling:** Endogenous crises with realistic recovery patterns
3. **Financialization:** Debt/GDP rises from 0.8 to 1.4 over period
4. **Wage share decline:** From ~67% (1970) to ~58% (2020s)

---

## 2. Inequality Dataset

**File:** `inequality_annual_data.csv`
**Observations:** 55 years (1970-2024)
**Generator:** `generate_inequality_data.py`

### Theoretical Framework
- Rising inequality in neoliberal era
- Functional vs personal distribution
- Financialization and top income shares
- Piketty-style wealth concentration

### Variable Groups

#### Functional Distribution (Class shares)
- `wage_share`: Wage share (% of GDP) - declining
- `profit_share`: Profit share (% of GDP) - rising
- `rent_share`: Rent share (% of GDP) - rising with financialization

#### Income Inequality
- `gini_income`: Income Gini coefficient (0.35 → 0.42)
- `gini_wealth`: Wealth Gini coefficient (0.75 → 0.85)
- `palma_ratio`: Top 10% / Bottom 40% income ratio
- `p90_p10_ratio`: 90th/10th percentile income ratio

#### Income Quintiles (% of total income)
- `income_q1_share`: Bottom 20% share
- `income_q2_share`: Second quintile
- `income_q3_share`: Middle 20%
- `income_q4_share`: Fourth quintile
- `income_q5_share`: Top 20% share

#### Top Income Shares
- `income_top10_pct`: Top 10% income share
- `income_top1_pct`: Top 1% income share (rising dramatically)
- `income_top01_pct`: Top 0.1% income share

#### Wealth Quintiles (% of total wealth)
- `wealth_q1_share`: Bottom 20% wealth share (near zero)
- `wealth_q2_share` through `wealth_q5_share`
- Note: Much more unequal than income

#### Top Wealth Shares
- `wealth_top10_pct`: Top 10% wealth share (~75%)
- `wealth_top1_pct`: Top 1% wealth share (~40%)
- `wealth_top01_pct`: Top 0.1% wealth share (~20%)
- `wealth_negative_share`: % with negative net worth (rising)

#### Distribution Indicators
- `median_to_mean_income`: Median/mean ratio (declining)
- `median_to_mean_wealth`: Wealth ratio (very low)

#### Income Sources by Quintile (%)
Bottom quintile:
- `bottom_q_wage_pct`: Wage income (60%)
- `bottom_q_transfer_pct`: Transfers (35%)
- `bottom_q_capital_pct`: Capital income (5%)

Middle quintile:
- `middle_q_wage_pct`: Wages (85%)
- `middle_q_transfer_pct`: Transfers (10%)
- `middle_q_capital_pct`: Capital (5%)

Top quintile:
- `top_q_wage_pct`: Wages (declining)
- `top_q_transfer_pct`: Transfers (minimal)
- `top_q_capital_pct`: Capital income (rising to 35%)

#### Poverty
- `poverty_rate`: Overall poverty rate (%)
- `child_poverty_rate`: Child poverty (higher)
- `elderly_poverty_rate`: Elderly poverty (declining)
- `poverty_gap`: Depth of poverty (%)

#### Mobility & Gaps
- `intergenerational_elasticity`: Income persistence (rising = less mobility)
- `gender_wage_ratio`: Women's/men's earnings (60% → 85%)
- `racial_wage_ratio`: Minority/majority earnings (70% → 80%)
- `ceo_worker_pay_ratio`: CEO/worker ratio (30x → 280x)

#### Wealth Composition (%)
- `financial_wealth_pct`: Share in financial assets (rising)
- `housing_wealth_pct`: Share in housing (declining)
- `business_wealth_pct`: Share in business equity

#### Homeownership by Quintile (%)
- `homeownership_bottom_q`: Bottom 20% (declining)
- `homeownership_middle_q`: Middle 20%
- `homeownership_top_q`: Top 20% (rising)

### Methodological Notes
1. **Neoliberal trends:** Inequality rises from ~1980 onward
2. **Wealth > income inequality:** Gini wealth much higher than income
3. **Capital income concentration:** Top quintile capital share rises dramatically
4. **Realistic correlations:** CEO pay, financialization, declining labor share

---

## 3. Sectoral Balances Dataset (SFC)

**File:** `sectoral_balances_data.csv`
**Observations:** 55 years (1970-2024)
**Generator:** `generate_sfc_data.py`

### Theoretical Framework
- Godley-Lavoie Stock-Flow Consistent framework
- Post-Keynesian sectoral balances approach
- Kalecki profit equation
- Accounting identity: Σ(sectoral balances) = 0

### Sectors
1. **Households**
2. **Non-financial corporations**
3. **Financial corporations**
4. **Government**
5. **Rest of world (foreign)**

### Variable Groups

#### Household Sector Flows (billions)
- `hh_wage_income`: Wage income
- `hh_transfer_income`: Government transfers
- `hh_capital_income`: Interest + dividends
- `hh_total_income`: Total income
- `hh_consumption`: Consumption expenditure
- `hh_interest_paid`: Interest payments on debt
- `hh_total_expenditure`: Total expenditure
- `hh_balance`: Net lending/borrowing (saving)
- `hh_balance_gdp_pct`: Balance as % of GDP

#### Corporate Sector Flows
- `nfc_sales`: Sales revenue
- `nfc_wages`: Wage costs
- `nfc_interest_paid`: Interest on corporate debt
- `nfc_gross_profits`: Gross profits
- `nfc_investment`: Investment expenditure
- `nfc_dividends`: Dividends paid
- `nfc_balance`: Net lending/borrowing (usually negative)
- `nfc_balance_gdp_pct`: Balance as % of GDP

#### Financial Sector Flows
- `fc_interest_received`: Interest income
- `fc_fees`: Fees and commissions (rising)
- `fc_total_income`: Total income
- `fc_interest_paid`: Interest paid
- `fc_dividends`: Dividends paid
- `fc_balance`: Net lending/borrowing (small)
- `fc_balance_gdp_pct`: Balance as % of GDP

#### Government Sector Flows
- `gov_tax_revenue`: Tax revenue
- `gov_spending`: Government purchases
- `gov_transfers`: Transfers to households
- `gov_interest_paid`: Interest on government debt
- `gov_total_expenditure`: Total expenditure
- `gov_balance`: Budget balance (usually deficit)
- `gov_balance_gdp_pct`: Balance as % of GDP

#### Foreign Sector Flows
- `exports`: Exports of goods and services
- `imports`: Imports of goods and services
- `net_exports`: Trade balance
- `net_foreign_income`: Net income from abroad
- `foreign_balance`: Foreign sector balance (= -current account)
- `foreign_balance_gdp_pct`: Balance as % of GDP

#### Aggregate Balances
- `private_balance`: Household + corporate + financial
- `private_balance_gdp_pct`: Private balance as % of GDP

#### Stock Variables (billions, cumulative)
- `hh_net_financial_assets`: Household NFA
- `nfc_net_financial_assets`: Corporate NFA (negative)
- `fc_net_financial_assets`: Financial sector NFA
- `gov_net_financial_assets`: Government NFA (negative = debt)
- `foreign_net_financial_assets`: Foreign NFA

#### Debt Stocks (billions)
- `household_liabilities`: Household debt
- `corporate_debt`: Corporate debt
- `government_debt`: Government debt
- `private_debt`: Household + corporate debt

#### Debt Ratios (% of GDP)
- `corporate_debt_gdp`: Corporate debt/GDP
- `government_debt_gdp`: Government debt/GDP
- `private_debt_gdp`: Private debt/GDP
- `net_foreign_debt_gdp`: Net foreign debt/GDP

#### Other Ratios
- `hh_debt_service_ratio`: Household interest/income (%)
- `corporate_leverage`: Corporate debt/equity

#### Post-Keynesian Identities
- `kalecki_profits`: Kalecki profit equation
  - Profits = Investment + Gov Deficit + Trade Surplus - HH Saving

### Methodological Notes
1. **Accounting rigor:** All sectoral balances sum to zero by construction
2. **Stock-flow consistency:** Flows accumulate to stocks
3. **Financialization:** Private debt/GDP rises over time
4. **Kalecki equation:** Verified in data (profits from spending)

---

## 4. Household Microdata

**File:** `household_microdata.csv`
**Observations:** 50,000 households (cross-section, 2023)
**Generator:** `generate_household_micro_data.py`

### Theoretical Framework
- Heterogeneous agent modeling
- Realistic inequality and heterogeneity
- Correlations between income, wealth, demographics
- Financial fragility indicators

### Variable Groups

#### Identifiers
- `household_id`: Unique identifier (1-50000)
- `year`: Survey year (2023)

#### Demographics
- `age`: Age of household head (20-90)
- `household_size`: Number of persons (1-6)
- `children`: Number of children
- `education`: Education level (0=<HS, 1=HS, 2=Some college, 3=BA, 4=Grad)
- `region`: Region (1=Urban, 2=Suburban, 3=Rural)
- `race`: Race/ethnicity (1=White, 2=Black, 3=Hispanic, 4=Asian, 5=Other)
- `gender`: Gender of head (0=Male, 1=Female)

#### Labor Market
- `employed`: Employment status (0/1)
- `occupation`: Occupation type (1=Service, 2=Manual, 3=Clerical, 4=Professional, 5=Management)
- `hours_worked`: Weekly hours worked
- `self_employed`: Self-employment status (0/1)

#### Income ($)
- `wage_income`: Annual wage and salary income
- `self_employment_income`: Self-employment income
- `capital_income`: Interest, dividends, rent (Pareto distributed)
- `transfer_income`: Social Security, unemployment, welfare
- `total_income`: Total annual income

#### Wealth ($)
- `homeowner`: Homeownership status (0/1)
- `home_value`: Home value (if owner)
- `mortgage_debt`: Mortgage debt
- `home_equity`: Net home equity
- `financial_assets`: Savings, stocks, bonds (highly concentrated)
- `business_owner`: Business ownership (0/1)
- `business_equity`: Business equity value
- `consumer_debt`: Credit cards, auto loans
- `student_debt`: Student loan debt
- `total_assets`: Sum of all assets
- `total_liabilities`: Sum of all debts
- `net_worth`: Total assets - total liabilities

#### Consumption & Saving ($)
- `consumption`: Annual consumption expenditure
- `saving`: Annual saving (income - consumption)
- `saving_rate`: Saving rate (%)

#### Financial Stress Indicators
- `debt_to_income_ratio`: Total debt/income (%)
- `liquid_asset_poor`: <3 months expenses in liquid assets (0/1)
- `housing_cost_burden`: Housing costs/income (%)
- `housing_burdened`: Housing costs >30% of income (0/1)

#### Classification
- `income_quintile`: Income quintile (1-5)
- `wealth_quintile`: Wealth quintile (1-5)
- `income_class`: Income class (1=Lower, 2=Middle, 3=Upper)

### Distributions
- **Income:** Log-normal with Pareto tail
- **Wealth:** Pareto distribution (more unequal than income)
- **Capital income:** Highly concentrated (Pareto α=1.5)
- **Age-wealth profile:** Realistic life-cycle accumulation

### Methodological Notes
1. **Realistic heterogeneity:** 50,000 households with diverse characteristics
2. **Correlations enforced:** Education → occupation → income → wealth
3. **Inequality:** Top 1% income share ~7%, wealth share ~13%
4. **Financial fragility:** 21.7% liquid asset poor, 18.5% housing burdened
5. **Gaps modeled:** Gender (18% wage gap), racial (25% gap)

---

## 5. Cross-Country Panel Data

**File:** `cross_country_panel_data.csv`
**Observations:** 1,650 (30 countries × 55 years)
**Generator:** `generate_panel_data.py`

### Theoretical Framework
- Varieties of capitalism (VoC)
- Comparative political economy
- Development economics
- Institutional economics

### Countries (N=30)

**Liberal Market Economies (LME):** USA, GBR, CAN, AUS, MEX

**Coordinated Market Economies (CME):** DEU, SWE, NLD, AUT, DNK

**Mediterranean/Mixed (MME):** FRA, ITA, ESP, GRC

**Developmental States:** JPN, KOR, TWN, THA

**State-led:** CHN, RUS, VNM

**Mixed/Emerging:** IND, BRA, TUR, POL, ZAF, IDN, ARG, NGA, BGD

### Variable Groups

#### Identifiers
- `country_code`: ISO 3-letter code
- `country_name`: Full country name
- `year`: Year (1970-2024)
- `capitalism_type`: VoC category (LME, CME, MME, Developmental, State-led, Mixed, Transition)
- `region`: Geographic region
- `development_level`: Advanced, Emerging, Developing
- `size_category`: Large, Medium, Small

#### Macroeconomic Variables
- `gdp`: GDP (billions, constant prices)
- `gdp_growth`: Annual GDP growth (%)
- `gdp_per_capita`: GDP per capita
- `unemployment_rate`: Unemployment rate (%)
- `inflation_rate`: Inflation rate (%)
- `population`: Population (millions)

#### Distribution
- `wage_share`: Wage share of GDP (%)
  - CME: Higher (~68%)
  - LME: Lower (~60%)
- `gini_coefficient`: Gini coefficient
  - CME: Lower (~0.28)
  - LME: Higher (~0.38)

#### Government
- `gov_spending_gdp`: Government spending (% of GDP)
  - CME: Higher (~45%)
  - LME: Lower (~35%)
- `gov_debt_gdp`: Government debt (% of GDP)

#### Finance
- `private_debt_gdp`: Private sector debt (% of GDP, rising)
- `financial_development_index`: Financial development (0-1)

#### Trade
- `trade_openness`: (Exports + Imports) / GDP (%)
  - Small countries: Higher
  - Large countries: Lower
- `current_account_gdp`: Current account balance (% of GDP)
  - Export-oriented (CME, CHN, JPN): Surplus
  - Others: Deficit
- `fdi_inflows_gdp`: FDI inflows (% of GDP)

#### Labor Institutions
- `union_density`: Union membership (% of workforce)
  - CME: High (~60%), declining slowly
  - LME: Low (~35%), declining fast
- `employment_protection`: Employment protection index (0-6)
  - CME/MME: Strong (~2.5)
  - LME: Weak (~1.0)

#### Crisis Indicator
- `crisis_year`: Binary crisis indicator (0/1)

### Country-Specific Parameters
Each country has calibrated:
- **Growth rate:** Higher for emerging/developing
- **Volatility:** Higher for developing, lower for CME
- **Inequality:** Varies by VoC type
- **Institutional quality:** Affects crisis probability

### Methodological Notes
1. **Panel structure:** Balanced panel (all countries, all years)
2. **VoC differences:** Systematic variation in institutions, outcomes
3. **Development levels:** Catch-up growth for emerging markets
4. **Crises:** Country-specific crisis probabilities
5. **Institutions:** Realistic labor market and financial institutions

---

## 6. Financial Crisis Dataset

**File:** `financial_crisis_data.csv`
**Observations:** 140 quarters (1990Q1-2024Q4)
**Generator:** `generate_crisis_data.py`

### Theoretical Framework
- Minsky's Financial Instability Hypothesis
- Endogenous crisis dynamics
- Boom-bust-recovery cycles
- Fragility accumulation

### Regimes
- **0 = Normal:** Stable growth, moderate fragility
- **1 = Boom/Buildup:** Rising fragility, credit boom, asset bubble
- **2 = Crisis:** Collapse, deleveraging, recession
- **3 = Recovery:** Gradual healing, reduced fragility

### Crisis Episodes Modeled
1. **Minor Crisis** (1994-1998)
2. **Dot-com Bubble** (1998-2002)
3. **Global Financial Crisis** (2005-2013)
4. **COVID-19 Shock** (2019-2022)

### Variable Groups

#### Time & Regime
- `date`: Quarter date
- `year`, `quarter`: Year and quarter
- `regime`: Regime indicator (0-3)
- `crisis_episode`: Named crisis episode

#### Minsky Framework
- `fragility_index`: System fragility (0-1)
  - Normal: ~0.3
  - Boom: ~0.7
  - Crisis: ~0.9
- `hedge_share`: % hedge finance borrowers (declining in boom)
- `speculative_share`: % speculative finance borrowers
- `ponzi_share`: % Ponzi finance borrowers (rising in boom)

#### Financial Variables
- `credit_total`: Total credit outstanding
- `credit_growth`: Credit growth rate (%)
  - Boom: 8-12%
  - Crisis: -10 to -15%
- `leverage_ratio`: Debt/equity ratio
- `asset_price_index`: Asset price index
- `asset_price_growth`: Asset price growth (%)
  - Boom: 10-15%
  - Crisis: -30 to -40%
- `credit_spread_bps`: Credit spread (basis points)
- `lending_standards_index`: Bank lending standards (-3 to +3)
- `volatility_index`: VIX-like volatility (10-80)

#### Real Economy
- `gdp`: GDP level
- `gdp_growth_pct`: Quarterly GDP growth (%)
- `unemployment_rate`: Unemployment (%)
- `investment`: Investment level
- `investment_growth`: Investment growth (very volatile)
- `consumption`: Consumption
- `corporate_profits`: Corporate profits
- `profit_margin`: Profit margin (%)

#### Debt Dynamics
- `household_debt_gdp`: Household debt (% of GDP)
- `corporate_debt_gdp`: Corporate debt (% of GDP)
- `household_debt_service`: Household debt service (% of income)
- `corporate_debt_service`: Corporate debt service (% of income)

#### Banking Sector
- `bank_capital_ratio`: Bank capital ratio (%)
- `npl_ratio`: Non-performing loan ratio (%)
- `bank_roe`: Bank return on equity (%)
- `finance_sector_gdp`: Financial sector size (% of GDP)

#### Crisis Indicators
- `crisis_probability`: Probability of crisis in next year (%)
- `financial_stress_index`: Financial stress (0-100)
- `systemic_risk_index`: Systemic risk measure

#### Policy Responses
- `policy_rate`: Central bank policy rate (%)
  - Crisis: Near zero
  - Normal: 2-4%
- `gov_deficit_gdp`: Government deficit (% of GDP)
- `central_bank_assets_gdp`: CB balance sheet (% of GDP)

### Methodological Notes
1. **Endogenous dynamics:** Crises emerge from boom dynamics
2. **Minsky stages:** Hedge → Speculative → Ponzi progression
3. **Feedback loops:** Asset prices ↔ credit ↔ fragility
4. **Realistic timing:** Boom lasts 3-4 years, crisis 1-2 years, recovery 2-4 years
5. **Policy responses:** Interest rate cuts, fiscal expansion, QE

---

## 7. Energy & Environment Dataset

**File:** `energy_environment_data.csv`
**Observations:** 55 years (1970-2024)
**Generator:** `generate_energy_environment_data.py`

### Theoretical Framework
- Ecological economics
- Energy transitions
- Just transition and distributional effects
- Political economy of energy systems

### Variable Groups

#### Baseline
- `year`: Year
- `gdp`: GDP
- `population`: Population

#### Total Energy
- `total_energy_consumption`: Total energy consumption
- `energy_intensity`: Energy/GDP ratio (declining)

#### Energy by Sector (% and absolute)
- `industrial_share`: Industry (% of total, declining)
- `transport_share`: Transport (% of total)
- `residential_share`: Residential (% of total, declining)
- `commercial_share`: Commercial (% of total, rising)
- `industrial_energy`, `transport_energy`, `residential_energy`, `commercial_energy`: Absolute values

#### Energy by Source (% and absolute)
- `coal_share`: Coal (%, declining: 35% → 13%)
- `oil_share`: Oil (%, declining: 40% → 30%)
- `gas_share`: Natural gas (%, stable: 15% → 25%)
- `nuclear_share`: Nuclear (%, peaked ~2000)
- `renewables_share`: Renewables (%, rising: 5% → 20%)
- `coal_energy`, `oil_energy`, `gas_energy`, `nuclear_energy`, `renewables_energy`: Absolute values

#### Renewable Breakdown
- `hydro_energy`: Hydroelectric (dominant early, declining share)
- `wind_energy`: Wind power (rising)
- `solar_energy`: Solar power (rapid recent growth)

#### Emissions
- `co2_emissions`: CO2 emissions (million tonnes)
- `ghg_emissions`: Total GHG emissions (CO2-equivalent)
- `emissions_per_capita`: Per capita emissions
- `emissions_intensity`: Emissions/GDP ratio (declining)

#### Energy Prices
- `oil_price_usd_barrel`: Oil price ($/barrel)
  - Shocks in 1973, 1979, 2008
- `gas_price_usd_mmbtu`: Natural gas price ($/MMBtu)
- `coal_price_usd_ton`: Coal price ($/ton)
- `electricity_price_cents_kwh`: Electricity price (cents/kWh)
- `solar_cost_usd_watt`: Solar PV cost ($/W, falling: $50 → $1)
- `wind_cost_usd_watt`: Wind cost ($/W, falling: $8 → $1.5)

#### Environmental Indicators
- `air_quality_index`: Air quality (100=1970, lower better)
- `renewables_investment_gdp`: Renewable investment (% of GDP)
- `energy_rd_gdp`: Energy R&D spending (% of GDP)
- `carbon_price_usd_tonne`: Carbon price ($/tonne, post-2005)

#### Employment (thousands)
- `fossil_employment_thousands`: Fossil fuel employment (declining)
- `renewable_employment_thousands`: Renewable energy employment (rising)

#### Energy Security
- `import_dependency_pct`: Energy import dependency (%)
- `energy_diversity_index`: Energy source diversity (0-1)

#### Distributional Effects (%)
- `avg_energy_burden_pct`: Average % of income spent on energy
- `low_income_energy_burden_pct`: Low-income energy burden (2.5x average)
- `energy_poverty_rate`: % unable to afford adequate energy

#### Regional Disparities (%)
- `rural_renewables_share`: Renewable share in rural areas (higher)
- `urban_renewables_share`: Renewable share in urban areas (lower)

### Methodological Notes
1. **Energy transition:** Clear shift from coal → gas → renewables
2. **Decoupling:** Energy intensity and emissions intensity both declining
3. **Technology costs:** Realistic learning curves for solar/wind
4. **Just transition:** Employment shifts, distributional burdens modeled
5. **Policy:** Carbon pricing introduced ~2005

---

## 📖 Usage Guidelines

### For Heterodox Economic Analysis

These datasets are designed to support:

1. **Post-Keynesian Macro:**
   - Endogenous cycles in macro data
   - SFC sectoral balances
   - Minsky crisis dynamics
   - Kalecki profit equation

2. **Marxian Economics:**
   - Functional distribution (wage/profit shares)
   - Class analysis
   - Reserve army of labor (unemployment)
   - Financialization trends

3. **Comparative Political Economy:**
   - Panel data with VoC categories
   - Institutional variation
   - Cross-country inequality patterns
   - Labor market institutions

4. **Distributional Economics:**
   - Comprehensive inequality measures
   - Household-level heterogeneity
   - Top income/wealth shares
   - Poverty and mobility

5. **Ecological Economics:**
   - Energy transitions
   - Just transition employment effects
   - Energy poverty
   - Emissions trajectories

### Python Analysis Examples

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
macro = pd.read_csv('datasets/macro_quarterly_data.csv')
inequality = pd.read_csv('datasets/inequality_annual_data.csv')

# Plot wage share over time
plt.plot(macro['date'], macro['wage_share'])
plt.title('Declining Wage Share (Neoliberal Era)')
plt.ylabel('Wage Share (%)')
plt.show()

# Analyze crisis dynamics
crisis = pd.read_csv('datasets/financial_crisis_data.csv')
crisis_periods = crisis[crisis['regime'] == 2]
print(crisis_periods.groupby('crisis_episode').mean())
```

### Recommended Analyses

1. **Wage-led vs Profit-led Growth:** Regress growth on wage share
2. **Minsky Moments:** Identify fragility thresholds predicting crises
3. **Varieties of Capitalism:** Compare outcomes across VoC types
4. **Inequality Trends:** Decompose rising inequality by source
5. **Sectoral Balances:** Test Godley's 3-balance identity
6. **Energy Transition:** Analyze employment effects of decarbonization
7. **Panel Regressions:** Institutions and growth/inequality
8. **Microsimulations:** Use household data for policy analysis

---

## 🔧 Data Generation

All datasets are generated from Python scripts with fixed random seeds for reproducibility.

### Requirements
```
numpy>=1.20
pandas>=1.3
matplotlib>=3.3
```

### Regenerate Data
```bash
cd datasets
python generate_macro_data.py
python generate_inequality_data.py
python generate_sfc_data.py
python generate_household_micro_data.py
python generate_panel_data.py
python generate_crisis_data.py
python generate_energy_environment_data.py
```

---

## 📚 Theoretical References

### Post-Keynesian
- Godley, W. & Lavoie, M. (2007). *Monetary Economics*
- Kalecki, M. (1971). *Selected Essays on the Dynamics of the Capitalist Economy*
- Minsky, H. (1986). *Stabilizing an Unstable Economy*

### Marxian
- Piketty, T. (2014). *Capital in the Twenty-First Century*
- Shaikh, A. (2016). *Capitalism: Competition, Conflict, Crises*

### Comparative Political Economy
- Hall, P. & Soskice, D. (2001). *Varieties of Capitalism*
- Wade, R. (1990). *Governing the Market*

### Ecological Economics
- Jackson, T. (2017). *Prosperity Without Growth*
- IPCC reports on emissions trajectories

---

## ⚠️ Limitations

1. **Synthetic Data:** Not real-world data, used for methods development
2. **Simplified Dynamics:** Real economies more complex
3. **Stylized Facts:** Focus on broad patterns, not precise calibration
4. **Correlations:** Some correlations imposed, may not match all contexts
5. **Crisis Timing:** Manually specified, not fully endogenous

**Use for:** Learning, methods development, teaching, proof-of-concept
**Do not use for:** Policy recommendations without real data validation

---

## 📧 Feedback

This is a learning exercise. Datasets prioritize:
- Heterodox economic theories
- Realistic stylized facts
- Educational value
- Methodological diversity

Suggestions for additional variables or datasets welcome!

---

**Generated:** 2025-11-18
**Python Version:** 3.11+
**License:** Educational use
