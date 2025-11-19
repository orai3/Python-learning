---
# Agent-Based Macroeconomic Model
### A Heterodox Computational Economics Framework

**Author:** Built for heterodox economics research and teaching
**Version:** 1.0.0
**License:** MIT

---

## Overview

This is a comprehensive **Agent-Based Macroeconomic Model (ABM)** designed for heterodox economics research, teaching, and policy analysis. Unlike Representative Agent models (DSGE), this ABM captures:

- **Heterogeneous agents** with diverse behaviors and characteristics
- **Emergent macro dynamics** from micro-level interactions
- **Involuntary unemployment** and coordination failures
- **Endogenous business cycles** without exogenous shocks
- **Financial instability** and credit cycles (Minskyan dynamics)
- **Distributional dynamics** and inequality evolution
- **Realistic market mechanisms** with frictions and rationing

---

## Key Features

### 🏭 **1000 Heterogeneous Firms**
- Post-Keynesian markup pricing (not marginal cost = price)
- Investment based on profits and animal spirits (Kaleckian)
- Adaptive expectations (not rational expectations)
- Endogenous bankruptcies and firm entry/exit
- Financial fragility classification (Minsky: hedge/speculative/Ponzi)

### 👥 **5000 Heterogeneous Households**
- Consumption driven by current income (Keynesian)
- Heterogeneous marginal propensities to consume by wealth class
- Job search with frictions and reservation wages
- Portfolio decisions (deposits, assets)
- Precautionary saving under uncertainty

### 🏦 **Banking Sector**
- **Endogenous money creation** (NOT loanable funds!)
- Credit rationing based on borrower risk (Stiglitz-Weiss)
- Pro-cyclical lending standards
- Default risk and bank fragility
- Capital adequacy constraints

### 🏛️ **Government & Central Bank**
- Fiscal policy: spending, taxes, transfers
- Automatic stabilizers and discretionary policy
- Monetary policy: Taylor rule or discretionary
- Lender of last resort in crises
- Stock-flow consistent accounting

### 📊 **Market Mechanisms**
- Labor market: search & matching, involuntary unemployment
- Goods market: quantity rationing, inventories
- Credit market: rationing (not market-clearing)
- Decentralized, sequential matching (not Walrasian auctioneer)

---

## Installation

### Requirements
```bash
pip install numpy matplotlib seaborn
```

### Setup
```bash
# Clone or download the repository
cd Python-learning

# The ABM is in the abm_macro/ directory
# Run examples:
python abm_example_comprehensive.py
```

---

## Quick Start

### Basic Simulation

```python
from abm_macro import MacroeconomyABM, ABMVisualizer

# Create economy
economy = MacroeconomyABM(
    n_firms=1000,
    n_households=5000,
    n_banks=10,
    random_seed=42
)

# Run simulation
results = economy.run(n_periods=300)

# Visualize
viz = ABMVisualizer(economy)
viz.plot_macro_dashboard(save_path='dashboard.png')
viz.plot_distributional_analysis(save_path='distribution.png')
viz.plot_business_cycle_analysis(save_path='cycles.png')
```

### Policy Experiments

```python
from abm_macro import PolicyExperiment

# Create custom policy experiment
experiment = PolicyExperiment("Fiscal Expansion")

# Add interventions at specific times
experiment.add_intervention(50, 'government', 'government_spending_target', 200.0)
experiment.add_intervention(50, 'central_bank', 'interest_rate', 0.01)

# Run with policy
results = economy.run(n_periods=300, policy_experiment=experiment)
```

### Compare with Representative Agent Model

```python
from abm_macro import RepresentativeAgentModel, compare_abm_vs_representative

# Run both models
abm = MacroeconomyABM(n_firms=500, n_households=2000, n_banks=10)
abm_results = abm.run(n_periods=200)

ra_model = RepresentativeAgentModel()
ra_results = ra_model.simulate(n_periods=200)

# Compare
compare_abm_vs_representative(abm_results.time_series, ra_results,
                              save_path='comparison.png')
```

---

## Theoretical Foundations

This ABM synthesizes insights from heterodox economics traditions:

### Post-Keynesian Economics
- **Effective demand principle**: Output determined by demand, not supply
- **Markup pricing**: Firms set prices using markup over costs (Kalecki)
- **Investment instability**: Animal spirits, fundamental uncertainty
- **Endogenous money**: Banks create credit, not intermediation

### Minskyan Finance
- **Financial Instability Hypothesis**: Stability → instability
- **Financial positions**: Hedge, speculative, Ponzi
- **Pro-cyclical credit**: Banks ease standards in booms, tighten in busts
- **Debt deflation**: Bankruptcies → credit crunch → recession

### Kaleckian Distribution
- **Wage share affects demand**: High inequality → demand constraints
- **Profits and investment**: Profits drive investment, investment drives profits
- **Degree of monopoly**: Markup reflects market power

### Complexity Economics
- **Heterogeneous agents**: No representative agent
- **Bounded rationality**: Adaptive expectations, satisficing
- **Path dependence**: History matters
- **Emergence**: Macro from micro interactions

---

## What ABM Captures That Representative Agent Models Miss

| Feature | ABM | Representative Agent |
|---------|-----|---------------------|
| **Heterogeneity** | ✅ Firms and households differ | ❌ Single agent |
| **Involuntary Unemployment** | ✅ Emerges from rationing | ❌ Assumes full employment |
| **Inequality Dynamics** | ✅ Wealth/income distribution evolves | ❌ N/A (single agent) |
| **Endogenous Cycles** | ✅ From micro interactions | ❌ Only exogenous shocks |
| **Financial Instability** | ✅ Minskyan boom-bust | ❌ Often no finance |
| **Bankruptcies** | ✅ Creative destruction | ❌ Immortal agent |
| **Credit Rationing** | ✅ Equilibrium phenomenon | ❌ Market clearing |
| **Coordination Failures** | ✅ Demand-side recessions | ❌ Only supply shocks |
| **Network Effects** | ✅ Contagion, spillovers | ❌ Isolated agent |
| **Distributional Effects** | ✅ Inequality affects demand | ❌ Irrelevant |

---

## Example Use Cases

### 1. **Teaching Heterodox Macro**
- Show students how macro emerges from micro
- Illustrate Keynesian vs neoclassical assumptions
- Demonstrate policy multipliers with heterogeneity

### 2. **Research Applications**
- Test heterodox hypotheses computationally
- Explore financial instability dynamics
- Analyze distributional effects of policy

### 3. **Policy Analysis**
- Compare fiscal vs monetary policy effectiveness
- Test Job Guarantee or UBI proposals
- Analyze inequality-growth tradeoffs

### 4. **Crisis Analysis**
- Simulate financial crises and contagion
- Test policy responses (austerity vs stimulus)
- Study debt deflation dynamics

---

## Model Validation

### Stylized Facts Reproduced
The model generates realistic macro patterns:

1. ✅ **Business cycles**: Irregular fluctuations without exogenous shocks
2. ✅ **Firm size distribution**: Power law (few large, many small)
3. ✅ **Wealth inequality**: High Gini (0.6-0.8), fat-tailed distribution
4. ✅ **Wage share dynamics**: Cyclical fluctuations around trend
5. ✅ **Credit cycles**: Pro-cyclical credit growth
6. ✅ **Bankruptcy clusters**: Waves of firm failures in downturns
7. ✅ **Unemployment persistence**: Hysteresis effects
8. ✅ **Goodwin cycles**: Unemployment-wage share dynamics

### Calibration
Parameters calibrated to empirical regularities:
- Labor share (α) ≈ 0.65-0.70
- Average markup ≈ 20-30%
- MPC by wealth class: workers (0.85-0.98), wealthy (0.40-0.65)
- Unemployment target ≈ 5%
- Depreciation rate ≈ 5% per period

---

## Visualization Tools

### 1. **Macro Dashboard**
- GDP and components (C, I, G)
- Unemployment and inflation
- Distribution (wage share, Gini)
- Credit and financial indicators

### 2. **Distributional Analysis**
- Lorenz curves (wealth, income)
- Consumption by wealth class
- Firm size distribution
- Profit rate distribution

### 3. **Business Cycle Analysis**
- Goodwin cycles (unemployment vs wage share)
- Investment-profit relationship (Kaleckian)
- Credit-GDP dynamics (Minskyan)
- Financial fragility indicators

### 4. **Animations**
- Real-time evolution of the economy
- All major aggregates updating simultaneously
- Save as GIF for presentations

---

## Pre-Built Policy Experiments

### 1. **Fiscal Austerity**
Tests impact of spending cuts in recession.
```python
experiment = PolicyExperiment.create_austerity_experiment()
```

### 2. **Quantitative Easing**
Ultra-low interest rates + bank support.
```python
experiment = PolicyExperiment.create_qe_experiment()
```

### 3. **Green New Deal**
Large fiscal expansion + job creation.
```python
experiment = PolicyExperiment.create_green_new_deal_experiment()
```

### 4. **Custom Experiments**
Create your own interventions at any time step.

---

## Output Files

Running the comprehensive example generates:

- `abm_dashboard.png` - Full macro dashboard
- `abm_distribution.png` - Distributional analysis
- `abm_cycles.png` - Business cycle dynamics
- `policy_comparison.png` - Policy experiment comparison
- `abm_vs_ra.png` - ABM vs Representative Agent
- `job_guarantee.png` - Custom policy example
- `abm_animation.gif` - Animated economy evolution

---

## Academic Applications

### Potential Papers
1. "Financial Instability in Agent-Based Macro Models: A Minskyan Approach"
2. "Distribution and Demand: Testing Kaleckian Hypotheses with ABM"
3. "Policy Effectiveness under Heterogeneity: ABM vs DSGE"
4. "Teaching Heterodox Macro: An Agent-Based Approach"

### Conference Presentations
- Association for Heterodox Economics (AHE)
- Post-Keynesian Economics Society (PKES)
- International Network for Economic Method (INEM)
- Eastern Economic Association (EEA)

### Teaching Materials
- Undergraduate: Principles of Macroeconomics
- Graduate: Advanced Macroeconomics, Computational Economics
- Workshops: Python for Economists, ABM Methods

---

## Extensions & Future Work

### Potential Extensions
1. **International Trade**: Multi-country model with exchange rates
2. **Environmental**: Carbon emissions, green investment
3. **Innovation**: Schumpeterian technological change
4. **Housing Market**: Real estate, mortgage debt
5. **Labor Unions**: Collective bargaining, wage setting
6. **Government Forms**: Democracy, autocracy effects

### Technical Improvements
1. GPU acceleration for larger simulations
2. Network visualization of firm-bank connections
3. Machine learning for pattern recognition
4. Sensitivity analysis tools
5. Calibration to real economy data

---

## References

### ABM Methodology
- Dosi, G., et al. (2010). "Schumpeter meeting Keynes: A policy-friendly model of endogenous growth and business cycles." *Journal of Economic Dynamics and Control*.
- Dawid, H., et al. (2019). "Agent-based macroeconomics." *Handbook of Computational Economics, Vol. 4*.

### Heterodox Theory
- Kalecki, M. (1971). *Selected Essays on the Dynamics of the Capitalist Economy*.
- Minsky, H. (1986). *Stabilizing an Unstable Economy*.
- Godley, W., & Lavoie, M. (2007). *Monetary Economics: An Integrated Approach*.

### Critique of Mainstream
- Kirman, A. (1992). "Whom or What Does the Representative Individual Represent?" *Journal of Economic Perspectives*.
- Stiglitz, J. (2018). "Where modern macroeconomics went wrong." *Oxford Review of Economic Policy*.

---

## Support & Contribution

### Getting Help
- Check the comprehensive example: `abm_example_comprehensive.py`
- Review individual modules for detailed comments
- Examine visualization output for model behavior

### Contributing
This model is designed for extension and modification:
- Add new agent behaviors in respective files
- Create new policy experiments
- Implement alternative market mechanisms
- Extend to multi-sector or multi-country settings

### Citation
If you use this model in research or teaching:

```
Agent-Based Macroeconomic Model (2024)
A Heterodox Computational Economics Framework
https://github.com/[your-repo]
```

---

## Contact

For academic collaboration, questions, or heterodox economics discussion:
- This is a teaching and research tool for the economics community
- Contributions and extensions welcome
- Suitable for undergraduate/graduate teaching and research

---

**Built with heterodox economics principles. No representative agents harmed in the making of this model.**

---
