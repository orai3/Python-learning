# Advanced Heterodox Economic Research Toolkit

A comprehensive, production-quality Python toolkit for heterodox economic research and analysis. Built for economists, researchers, and students interested in Post-Keynesian, Marxian, Sraffian, and Kaleckian approaches to political economy.

**Author:** Claude
**License:** MIT
**Python Version:** 3.8+

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Toolkit Structure](#toolkit-structure)
4. [Theoretical Models](#theoretical-models)
5. [PyQt6 Applications](#pyqt6-applications)
6. [Empirical Frameworks](#empirical-frameworks)
7. [Historical Models](#historical-models)
8. [Usage Examples](#usage-examples)
9. [Academic Applications](#academic-applications)
10. [Contributing](#contributing)
11. [References](#references)

---

## 🎯 Overview

This toolkit provides **research-grade implementations** of major heterodox economic models and analytical frameworks. Unlike toy examples, these are complete, documented, production-quality implementations suitable for:

- **Academic research** and publication
- **Teaching** heterodox economics
- **Policy analysis** from heterodox perspectives
- **Replication studies** of key empirical papers
- **Comparative analysis** of mainstream vs heterodox approaches

### Key Features

✅ **Complete theoretical models** with extensive mathematical documentation
✅ **Professional PyQt6 applications** for model building and analysis
✅ **Empirical toolkits** for working with real-world data
✅ **Historical model implementations** from foundational texts
✅ **Publication-quality visualizations** and LaTeX export
✅ **Comprehensive testing and validation** of SFC consistency
✅ **Extensive references** to academic literature

---

## 💻 Installation

### Requirements

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0

# For PyQt6 applications (optional but recommended)
PyQt6>=6.2.0

# For advanced statistical analysis
statsmodels>=0.13.0
scikit-learn>=1.0.0
```

### Install

```bash
# Clone or download this repository
cd advanced-heterodox-research

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## 📁 Toolkit Structure

```
advanced-heterodox-research/
├── theoretical_models/          # Section 1: Advanced theoretical implementations
│   ├── godley_lavoie_sfc.py        # Multi-sector SFC model (1000+ lines)
│   ├── keen_minsky_model.py        # Financial instability dynamics
│   ├── sraffa_production.py        # Production prices and reswitching
│   ├── goodwin_keen_integration.py # Distribution & debt dynamics
│   └── kaleckian_structural.py     # Kaleckian macro model
│
├── pyqt_applications/           # Section 2: Research-grade GUI applications
│   ├── sfc_model_builder.py        # Interactive SFC model development
│   ├── macro_data_analysis.py      # Heterodox macro data analysis
│   └── input_output_tool.py        # Input-output analysis
│
├── empirical_frameworks/        # Section 3: Empirical analysis frameworks
│   ├── profit_rate_decomposition.py   # Marxian profit rate analysis
│   ├── wage_led_profit_led.py         # Bhaduri-Marglin estimation
│   └── sectoral_balances.py           # MMT-style sectoral analysis
│
├── historical_models/           # Section 4: Classic heterodox models
│   ├── kalecki_models.py           # Kalecki's collected works
│   ├── marx_reproduction.py        # Marx's reproduction schemas
│   ├── sraffa_standard.py          # Sraffa's standard system
│   └── minsky_stages.py            # Minsky's financial stages
│
├── comparative_frameworks/      # Section 5: Mainstream vs heterodox
│   ├── growth_theory_comparison.py    # Solow vs Kaleckian
│   ├── distribution_comparison.py     # Multiple distribution theories
│   └── money_banking_comparison.py    # Loanable funds vs endogenous money
│
├── academic_tools/              # Section 6: Research workflow tools
│   ├── literature_database.py      # Heterodox literature management
│   ├── replication_framework.py    # For replicating empirical studies
│   └── teaching_material_gen.py    # Generate teaching materials
│
├── examples/                    # Example usage and tutorials
│   ├── sfc_baseline.png
│   ├── keen_minsky_baseline.png
│   └── tutorial_notebooks/
│
├── tests/                       # Unit tests
├── docs/                        # Additional documentation
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── setup.py                     # Package installation
```

---

## 🔬 Theoretical Models (Section 1)

### 1. Godley-Lavoie Multi-Sector SFC Model

**File:** `theoretical_models/godley_lavoie_sfc.py`

Complete implementation of a 5-sector Stock-Flow Consistent model following Godley & Lavoie (2007).

**Features:**
- Balance sheet matrix with full SFC accounting
- Transaction flow matrix
- Portfolio allocation (Tobinesque)
- Endogenous money creation
- Fiscal and monetary policy rules
- Iterative Gauss-Seidel solver
- Validation functions ensuring consistency

**Example Usage:**

```python
from theoretical_models.godley_lavoie_sfc import SFCModel, SFCParameters

# Create model with default parameters
model = SFCModel()

# Simulate baseline
baseline = model.simulate(periods=50)

# Fiscal stimulus scenario
def fiscal_stimulus(t, params):
    if t == 10:
        params.g_bar *= 1.2  # 20% spending increase

fiscal_sim = model.simulate(periods=50, shock_fn=fiscal_stimulus)

# Validate consistency
validation = model.validate_consistency(model.states[-1], model.states[-2])
print(f"Balance sheet consistent: {validation['balance_sheet']['assets_equal_liabilities']}")
```

**Key Results:**
- Demonstrates stock-flow consistency principles
- Shows fiscal multiplier effects
- Illustrates sectoral balance accounting

**References:**
- Godley & Lavoie (2007): *Monetary Economics*

---

### 2. Keen-Minsky Dynamic Model

**File:** `theoretical_models/keen_minsky_model.py`

Steve Keen's formalization of Minsky's Financial Instability Hypothesis with debt-deflation dynamics.

**Features:**
- Coupled differential equations (employment, wage share, debt)
- Endogenous money through credit creation
- Fisher debt-deflation mechanism
- Bifurcation analysis tools
- Phase space visualization
- Crisis identification algorithms

**Example Usage:**

```python
from theoretical_models.keen_minsky_model import KeenMinskyModel

model = KeenMinskyModel()

# Simulate
df = model.simulate(t_max=200, t_points=5000)

# Identify crises
crises = model.identify_crises()
print(f"Number of crises: {len(crises)}")

# Minsky stages
df_minsky = model.minsky_stages()
print(df_minsky['minsky_stage'].value_counts())

# Bifurcation analysis
r_range = np.linspace(0.01, 0.10, 20)
bifurc_df = model.bifurcation_analysis('r_debt', r_range)
```

**Key Results:**
- Endogenous business cycles and crises
- Debt-driven instability
- Minsky moment dynamics

**References:**
- Keen (1995, 2013): Financial Instability Hypothesis
- Minsky (1986): *Stabilizing an Unstable Economy*

---

### 3. Sraffian Production Model

**File:** `theoretical_models/sraffa_production.py`

Complete implementation of Piero Sraffa's production price system with reswitching analysis.

**Features:**
- Production prices for any input-output system
- Wage-profit rate frontier
- Standard commodity calculation
- Joint production and fixed capital
- Reswitching demonstration
- Labor values vs prices

**Example Usage:**

```python
from theoretical_models.sraffa_production import (
    SraffaModel, ProductionTechnique, ReswitchingAnalysis
)

# Define production technique
A = np.array([[0.2, 0.3], [0.3, 0.1]])  # Input coefficients
l = np.array([0.5, 0.6])  # Labor inputs

technique = ProductionTechnique(
    name="Alpha",
    A=A,
    l=l,
    commodity_names=['Corn', 'Iron']
)

model = SraffaModel(technique)

# Calculate maximum profit rate
R = model.maximum_profit_rate()

# Wage-profit frontier
frontier_df = model.wage_profit_frontier(n_points=200)

# Production prices at different profit rates
for r_pct in [0, 25, 50, 75]:
    r = (r_pct / 100) * R
    p = model.production_prices(r, w=1.0)
    print(f"r = {r:.4f}: p = {p}")

# Demonstrate reswitching
reswitch = ReswitchingAnalysis([tech_alpha, tech_beta])
if reswitch.detect_reswitching():
    print("Reswitching detected! Capital theory invalid.")
```

**Key Results:**
- Prices depend on distribution (critique of neoclassical theory)
- Reswitching invalidates simple capital stories
- Labor values vs production prices

**References:**
- Sraffa (1960): *Production of Commodities by Means of Commodities*
- Cambridge Capital Controversy literature

---

### 4. Goodwin-Keen Integration

**File:** `theoretical_models/goodwin_keen_integration.py`

Combines Goodwin's predator-prey growth cycle with Keen's debt dynamics.

**Features:**
- 3D phase space (employment, wage share, debt)
- Class conflict dynamics
- Financial fragility overlay
- Cycle statistics and characterization

**Example Usage:**

```python
from theoretical_models.goodwin_keen_integration import GoodwinKeenModel

model = GoodwinKeenModel()

# Simulate
df = model.simulate(t_max=300, t_points=10000)

# Analyze cycles
stats = model.cycle_statistics()
print(f"Cycle period: {stats['cycle_period']:.2f}")
print(f"Crisis frequency: {stats['crisis_frequency']:.2%}")

# 3D phase space
lambda_arr, omega_arr, d_arr = model.phase_space_3d()
```

**Key Results:**
- Interaction of distributive and financial cycles
- Complex dynamics from two feedback mechanisms
- How debt amplifies distributional conflict

**References:**
- Goodwin (1967): Growth Cycle
- Keen (2013): Integrated model

---

### 5. Kaleckian Structural Model

**File:** `theoretical_models/kaleckian_structural.py`

Multi-sector Kaleckian model with wage-led vs profit-led regimes.

**Features:**
- Markup pricing (degree of monopoly)
- Investment function (profits + accelerator)
- Wage-led vs profit-led classification
- Paradox of thrift demonstration
- Capacity utilization dynamics

**Example Usage:**

```python
from theoretical_models.kaleckian_structural import KaleckianModel

model = KaleckianModel()

# Check demand regime
is_wage_led = model.is_wage_led()
print(f"Regime: {'Wage-led' if is_wage_led else 'Profit-led'}")

# Equilibrium
u_eq = model.equilibrium_utilization()
g_eq = model.equilibrium_growth_rate()

# Paradox of thrift
pot_results = model.paradox_of_thrift()
print(f"Higher saving → growth change: {pot_results['growth_change']:.4f}")

# Regime analysis
pi_range = np.linspace(0.2, 0.6, 100)
regime_df = model.regime_analysis(pi_range)
```

**Key Results:**
- Distribution affects growth
- Paradox of thrift and paradox of costs
- Demand-determined output

**References:**
- Kalecki (1971): *Selected Essays*
- Bhaduri & Marglin (1990): Wage-led vs profit-led

---

## 🖥️ PyQt6 Applications (Section 2)

### SFC Model Development Environment

**File:** `pyqt_applications/sfc_model_builder.py`

Professional-grade GUI application for building and simulating SFC models.

**Features:**
- Visual matrix editors (balance sheet, transaction flows)
- Equation editor with syntax checking
- Parameter management with sensitivity analysis
- Model save/load (JSON format)
- Real-time simulation with progress tracking
- Publication-quality plot export
- LaTeX table generation for papers
- Consistency validation

**Running the Application:**

```bash
python pyqt_applications/sfc_model_builder.py
```

**Architecture:**
- Full MVC pattern
- Background thread for simulations
- Extensible plugin system
- Proper separation of concerns

**Use Cases:**
- Teaching SFC methodology
- Building custom macro models
- Research on monetary economics
- Scenario analysis for policy

---

## 📊 Empirical Frameworks (Section 3)

### Profit Rate Decomposition Toolkit

**File:** `empirical_frameworks/profit_rate_decomposition.py`

Comprehensive framework for Marxian profit rate analysis using national accounts data.

**Features:**
- Multiple decomposition methods
- Trend analysis (linear regression)
- Structural break tests (Chow test)
- Counteracting tendencies identification
- International comparisons
- Publication-quality visualizations

**Example Usage:**

```python
from empirical_frameworks.profit_rate_decomposition import (
    ProfitRateData, ProfitRateDecomposition
)

# Load your data (from OECD, EU KLEMS, etc.)
data = ProfitRateData(
    year=years,
    output=Y,
    capital_stock=K,
    depreciation=delta_K,
    wages=W,
    profits=P
)

# Create decomposition
decomp = ProfitRateDecomposition(data)

# Standard decomposition: r = (Π/Y) * (Y/K)
df = decomp.standard_decomposition()

# Trend analysis
trends = decomp.trend_analysis(start_year=1960, end_year=2020)
print(f"Annual profit rate change: {trends['profit_rate']['annual_growth_rate']*100:.2f}%")

# Test for structural breaks
breaks = decomp.structural_break_test()
significant = breaks[breaks['significant']]
print(f"Significant breaks: {significant['break_year'].tolist()}")

# Counteracting tendencies
counter_df = decomp.counteracting_tendencies()

# Visualize
from empirical_frameworks.profit_rate_decomposition import plot_profit_rate_analysis
fig = plot_profit_rate_analysis(decomp)
```

**Data Sources:**
- OECD National Accounts
- EU KLEMS Database
- Penn World Tables
- National statistical offices (BEA, ONS, etc.)

**References:**
- Duménil & Lévy (1993): *The Economics of the Profit Rate*
- Shaikh & Tonak (1994): *Measuring the Wealth of Nations*
- Basu & Vasudevan (2013): Technology, distribution, profit rate

---

## 📜 Historical Models (Section 4)

### Kalecki's Collected Models

**File:** `historical_models/kalecki_models.py`

Computational implementations of Michał Kalecki's key contributions.

**Models Included:**

1. **Profit Equation (1935)**
   - "Workers spend what they earn, capitalists earn what they spend"
   - Shows profits determined by capitalist spending

2. **Investment Function (1943)**
   - I = f(P, ΔK, ΔP)
   - Profits, capital stock changes, profit changes

3. **Business Cycle Model (1937)**
   - Endogenous cycles from investment-profit dynamics

4. **Degree of Monopoly Pricing (1938)**
   - Markup pricing theory
   - Distribution from market structure

5. **Political Business Cycle (1943)**
   - Capitalist opposition to full employment

**Example Usage:**

```python
from historical_models.kalecki_models import (
    kalecki_profit_equation,
    KaleckiBusinessCycle,
    PoliticalBusinessCycle
)

# Profit equation
result = kalecki_profit_equation(c_w=1.0, c_p=0.7, i=100, params=...)
print(f"Profits = {result['profits']:.2f}")

# Business cycle simulation
cycle_model = KaleckiBusinessCycle()
df = cycle_model.simulate(t_max=100)

# Political economy of employment
pbc = PoliticalBusinessCycle()
pbc.political_power_business = 0.8  # Business dominance
eq = pbc.political_equilibrium(y_potential=1000)
print(f"Equilibrium unemployment: {eq['unemployment_rate']*100:.1f}%")
```

**References:**
- Kalecki (1971): *Selected Essays on the Dynamics of the Capitalist Economy*
- Kalecki (1943): Political Aspects of Full Employment

---

## 🎓 Usage Examples

### Complete Research Workflow

```python
# 1. Build theoretical model
from theoretical_models.godley_lavoie_sfc import SFCModel

model = SFCModel()
baseline = model.simulate(periods=100)

# 2. Analyze empirical data
from empirical_frameworks.profit_rate_decomposition import *

data = load_oecd_data('USA', start=1960, end=2020)  # Your data loading function
decomp = ProfitRateDecomposition(data)
trends = decomp.trend_analysis()

# 3. Compare with historical models
from historical_models.kalecki_models import KaleckiBusinessCycle

kalecki = KaleckiBusinessCycle()
kalecki_sim = kalecki.simulate(t_max=100)

# 4. Generate publication-ready outputs
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: SFC simulation
axes[0].plot(baseline['t'], baseline['y_r'])
axes[0].set_title('SFC Model: Real GDP')

# Plot 2: Empirical profit rate
plot_profit_rate_analysis(decomp)

# Plot 3: Kalecki cycle
axes[2].plot(kalecki_sim['t'], kalecki_sim['P'])
axes[2].set_title('Kalecki: Profits')

plt.tight_layout()
plt.savefig('research_output.png', dpi=300)

# 5. Export LaTeX tables for paper
# (Use export functions from models)
```

---

## 🎯 Academic Applications

### Use Case 1: Teaching Heterodox Economics

```python
# Demonstrate paradox of thrift
from theoretical_models.kaleckian_structural import KaleckianModel

model = KaleckianModel()

# Show students what happens when saving increases
pot_results = model.paradox_of_thrift()

print("Paradox of Thrift Demonstration:")
print(f"Baseline growth: {pot_results['baseline_growth']*100:.2f}%")
print(f"After saving increase: {pot_results['high_saving_growth']*100:.2f}%")
print(f"Paradox confirmed: {pot_results['paradox_confirmed']}")
```

### Use Case 2: Replication Study

```python
# Replicate Basu & Vasudevan (2013) profit rate decomposition for US

from empirical_frameworks.profit_rate_decomposition import *

# Load BEA data (your data loading code)
us_data = load_bea_data()

decomp = ProfitRateDecomposition(us_data)

# Replicate their decomposition
trends_1948_2007 = decomp.trend_analysis(start_year=1948, end_year=2007)

# Test their structural break hypothesis (1980)
breaks = decomp.structural_break_test(candidate_years=[1980])

# Compare results with paper
```

### Use Case 3: Policy Analysis

```python
# Analyze fiscal multiplier in different regimes

from theoretical_models.kaleckian_structural import KaleckianModel

# Wage-led economy
model_wl = KaleckianModel(params_wage_led)
# Profit-led economy
model_pl = KaleckianModel(params_profit_led)

# Simulate fiscal expansion in both
# Compare results
# Generate policy brief
```

---

## 📖 References

### Books

- Godley, W., & Lavoie, M. (2007). *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth*. Palgrave Macmillan.
- Kalecki, M. (1971). *Selected Essays on the Dynamics of the Capitalist Economy 1933-1970*. Cambridge University Press.
- Sraffa, P. (1960). *Production of Commodities by Means of Commodities: Prelude to a Critique of Economic Theory*. Cambridge University Press.
- Minsky, H. (1986). *Stabilizing an Unstable Economy*. Yale University Press.
- Lavoie, M. (2014). *Post-Keynesian Economics: New Foundations*. Edward Elgar.

### Key Papers

- Goodwin, R. (1967). A Growth Cycle. In C. H. Feinstein (Ed.), *Socialism, Capitalism and Economic Growth*.
- Keen, S. (1995). Finance and Economic Breakdown: Modeling Minsky's Financial Instability Hypothesis. *Journal of Post Keynesian Economics*, 17(4), 607-635.
- Bhaduri, A., & Marglin, S. (1990). Unemployment and the real wage. *Cambridge Journal of Economics*, 14(4), 375-393.
- Kalecki, M. (1943). Political Aspects of Full Employment. *The Political Quarterly*, 14(4), 322-330.

---

## 🤝 Contributing

Contributions are welcome! Areas for expansion:

1. Additional empirical frameworks (sectoral balances, wage-led/profit-led estimation)
2. More historical models (Marx, Sraffa extensions)
3. International comparisons and data import tools
4. Additional PyQt6 applications
5. Tutorial notebooks and examples
6. Unit tests

---

## 📄 License

MIT License - free for academic and educational use.

---

## 🙏 Acknowledgments

Built on the theoretical foundations of:
- Wynne Godley & Marc Lavoie (SFC modeling)
- Michał Kalecki (effective demand, distribution)
- Piero Sraffa (production prices, capital theory critique)
- Hyman Minsky (financial instability)
- Steve Keen (monetary Minsky models)
- Richard Goodwin (growth cycles)
- Joan Robinson, Nicholas Kaldor, Luigi Pasinetti (Cambridge tradition)

---

## 📞 Support

For questions, issues, or academic collaboration:
- Review the code documentation (extensive docstrings throughout)
- Check examples in `examples/` directory
- Consult the referenced academic papers

---

**Happy researching! 🎓📊**

*Last updated: January 2025*
