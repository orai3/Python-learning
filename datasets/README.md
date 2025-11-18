# Synthetic Economic Datasets for Heterodox Analysis

A comprehensive collection of synthetic datasets designed for heterodox economic analysis, focusing on Post-Keynesian, Marxian, and Comparative Political Economy approaches.

## 🎯 Quick Start

```python
import pandas as pd

# Load macroeconomic data
macro = pd.read_csv('datasets/macro_quarterly_data.csv')
macro['date'] = pd.to_datetime(macro['date'])

# Basic analysis
print(f"Wage share 1970: {macro['wage_share'].iloc[0]:.1f}%")
print(f"Wage share 2024: {macro['wage_share'].iloc[-1]:.1f}%")
print(f"Private debt/GDP 1970: {macro['private_debt_gdp_ratio'].iloc[0]:.1f}")
print(f"Private debt/GDP 2024: {macro['private_debt_gdp_ratio'].iloc[-1]:.1f}")
```

## 📊 Available Datasets

| Dataset | File | N | Period | Key Features |
|---------|------|---|--------|--------------|
| **Macro** | macro_quarterly_data.csv | 220 | 1970-2024 | National accounts, financialization |
| **Inequality** | inequality_annual_data.csv | 55 | 1970-2024 | Gini, class shares, top incomes |
| **SFC Balances** | sectoral_balances_data.csv | 55 | 1970-2024 | Godley framework, Kalecki |
| **Households** | household_microdata.csv | 50,000 | 2023 | Individual households, wealth |
| **Panel** | cross_country_panel_data.csv | 1,650 | 1970-2024 | 30 countries, VoC |
| **Crisis** | financial_crisis_data.csv | 140 | 1990-2024 | Minsky framework |
| **Energy** | energy_environment_data.csv | 55 | 1970-2024 | Green transition |

**Total:** 102,000+ observations across 7 datasets

## 📖 Documentation

See **[DATA_DICTIONARY.md](DATA_DICTIONARY.md)** for:
- Detailed variable definitions
- Theoretical frameworks
- Methodological notes
- Usage examples
- References

## 🔍 Key Features by Dataset

### 1. Macroeconomic Data
- Endogenous business cycles (Goodwin model)
- Financial cycles (Minsky)
- Crisis episodes (1975, 1982, 1991, 2001, 2008, 2020)
- Declining wage share (neoliberal trend)
- Rising financialization

### 2. Inequality Data
- Income Gini: 0.35 → 0.42
- Wealth Gini: 0.75 → 0.85
- CEO/Worker ratio: 30x → 280x
- Top 1% income share: 8% → 15%
- Functional distribution (class shares)

### 3. Sectoral Balances (SFC)
- Godley-Lavoie framework
- 5 sectors (household, corporate, financial, government, foreign)
- Stock-flow consistency enforced
- Kalecki profit equation verified
- Accounting identity: Σ(balances) = 0

### 4. Household Microdata
- 50,000 individual households
- Realistic income/wealth distributions
- Demographics, education, region
- Financial fragility indicators
- Top 1% wealth share: 13%

### 5. Cross-Country Panel
- 30 countries × 55 years
- Varieties of Capitalism (LME, CME, MME, etc.)
- Development levels (Advanced, Emerging, Developing)
- Institutional variation (unions, employment protection)
- Comparative analysis ready

### 6. Financial Crisis Data
- Minsky's Financial Instability Hypothesis
- Fragility index (0-1)
- Hedge/Speculative/Ponzi finance shares
- 4 crisis episodes modeled
- Boom-bust-recovery dynamics

### 7. Energy & Environment
- Energy transition (coal → renewables)
- Emissions trajectories
- Just transition (employment shifts)
- Energy poverty indicators
- Renewable cost reductions

## 💡 Use Cases

### For Students
- Learn heterodox economic theories with data
- Practice pandas, matplotlib, econometrics
- Understand stylized facts
- Develop critical analysis skills

### For Researchers
- Test new methodologies
- Develop models before using real data
- Teaching materials for pluralist courses
- Proof-of-concept analyses

### For Policy Analysis
- Scenario modeling
- Distributional impact assessment
- Energy transition planning
- Crisis early warning systems

**⚠️ Important:** These are synthetic data for learning. Always validate methods with real-world data before policy use.

## 🔧 Technical Details

### Requirements
```bash
pip install numpy pandas matplotlib
```

### Data Generation
All datasets generated with fixed random seeds for reproducibility:
```bash
python generate_macro_data.py
python generate_inequality_data.py
# ... etc
```

### File Formats
- All files: CSV format
- Dates: ISO format (YYYY-MM-DD)
- Numbers: Decimal notation
- Missing values: None (complete datasets)

## 📈 Example Analyses

### 1. Wage-Led vs Profit-Led Growth
```python
import pandas as pd
import statsmodels.api as sm

macro = pd.read_csv('macro_quarterly_data.csv')
macro['wage_share_lag'] = macro['wage_share'].shift(1)

X = sm.add_constant(macro[['wage_share_lag']].dropna())
y = macro['gdp_growth_rate'].iloc[1:]

model = sm.OLS(y, X).fit()
print(model.summary())
```

### 2. Inequality Trends
```python
import matplotlib.pyplot as plt

ineq = pd.read_csv('inequality_annual_data.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(ineq['year'], ineq['gini_income'], label='Income')
ax1.plot(ineq['year'], ineq['gini_wealth'], label='Wealth')
ax1.legend()
ax1.set_title('Rising Inequality')

ax2.plot(ineq['year'], ineq['wage_share'])
ax2.set_title('Declining Wage Share')

plt.tight_layout()
plt.show()
```

### 3. Crisis Dynamics
```python
crisis = pd.read_csv('financial_crisis_data.csv')

# Compare normal vs crisis periods
normal = crisis[crisis['regime'] == 0]
crisis_periods = crisis[crisis['regime'] == 2]

print("Normal periods:")
print(f"  GDP growth: {normal['gdp_growth_pct'].mean():.2f}%")
print(f"  Ponzi share: {normal['ponzi_share'].mean():.1f}%")

print("\nCrisis periods:")
print(f"  GDP growth: {crisis_periods['gdp_growth_pct'].mean():.2f}%")
print(f"  Ponzi share: {crisis_periods['ponzi_share'].mean():.1f}%")
```

### 4. Varieties of Capitalism
```python
panel = pd.read_csv('cross_country_panel_data.csv')

# Compare capitalism types
comparison = panel.groupby('capitalism_type').agg({
    'wage_share': 'mean',
    'gini_coefficient': 'mean',
    'union_density': 'mean',
    'gov_spending_gdp': 'mean'
})

print(comparison)
```

## 🎓 Learning Path

### Beginner
1. Start with **macro_quarterly_data.csv**
2. Plot time series (GDP, unemployment, wage share)
3. Calculate correlations
4. Identify crisis periods

### Intermediate
1. Use **inequality_annual_data.csv**
2. Decompose inequality trends
3. Analyze **household_microdata.csv**
4. Create Lorenz curves, calculate Gini from micro data

### Advanced
1. Analyze **sectoral_balances_data.csv**
2. Verify SFC accounting identities
3. Test Kalecki profit equation
4. Build **financial_crisis_data.csv** early warning models
5. Panel regressions with **cross_country_panel_data.csv**

## 📚 Theoretical Foundations

### Post-Keynesian
- Endogenous money and credit
- Investment drives growth (not savings)
- Wage-led vs profit-led growth regimes
- Financial instability hypothesis (Minsky)
- Stock-flow consistency (Godley)

### Marxian
- Labor theory of value (implicit in wage/profit shares)
- Reserve army of labor (unemployment)
- Falling rate of profit (profit share dynamics)
- Financialization and fictitious capital

### Comparative Political Economy
- Varieties of capitalism
- Institutional complementarities
- Coordinated vs liberal market economies
- Development state models

### Ecological Economics
- Energy-GDP decoupling
- Just transition
- Distributional effects of climate policy
- Renewable energy transitions

## 🌟 Heterodox Features

Unlike mainstream datasets, these emphasize:

1. **Power and Distribution**
   - Class shares of income
   - Wage bargaining and unions
   - CEO pay ratios

2. **Financial Instability**
   - Endogenous crises
   - Minsky dynamics
   - Debt-driven cycles

3. **Institutions Matter**
   - VoC categories
   - Labor market institutions
   - Different growth models

4. **Inequality is Central**
   - Comprehensive measures
   - Wealth > income inequality
   - Top shares, poverty

5. **Sustainability**
   - Energy transitions
   - Just transition
   - Energy poverty

## ⚖️ License & Attribution

**For Educational Use**

These datasets are synthetic, generated for learning purposes. Free to use for:
- Academic courses
- Research methods development
- Self-study
- Teaching materials

**Citation (if used in teaching/research):**
```
Synthetic Economic Datasets for Heterodox Analysis (2025)
Python-learning/datasets
Generated: 2025-11-18
```

## 🤝 Contributing

Suggestions for improvements:
- Additional variables
- New datasets (e.g., input-output tables, firm-level data)
- Better calibration
- Additional crisis episodes
- More countries in panel

## 📧 Questions?

Review **DATA_DICTIONARY.md** for detailed documentation.

---

**Generated:** 2025-11-18
**Total Size:** ~102,000 observations
**Format:** CSV
**Language:** Python 3.11+
