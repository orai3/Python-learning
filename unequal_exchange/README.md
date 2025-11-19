```markdown
# Unequal Exchange Framework

A comprehensive computational framework for dependency theory and unequal exchange analysis, implementing Emmanuel, Amin, Prebisch-Singer, and other heterodox economic models.

## Overview

This framework provides production-ready tools for analyzing:
- **Unequal exchange** through international trade (Emmanuel, Amin)
- **Global value chain rent extraction**
- **Terms of trade dynamics** (Prebisch-Singer)
- **Transfer pricing and IP rent flows**
- **Super-exploitation metrics**
- **Multi-country input-output analysis**
- **Policy simulations** (South-South cooperation, delinking, industrial policy)

## Features

### 🎯 Theoretical Models
- **Emmanuel Model** (1972): Wage differentials and value transfers
- **Amin Model** (1974): Extended unequal exchange with productivity & super-exploitation
- **Prebisch-Singer**: Terms of trade deterioration analysis
- **Multi-country I-O**: Leontief framework for global value chains

### 📊 Analysis Tools
- Labor value calculations (direct & vertically integrated)
- GVC rent extraction (monopoly, IP, brand)
- Super-exploitation metrics (wage-productivity gaps, labor share)
- Transfer pricing estimation
- Intellectual property rent flows
- Value appropriation patterns

### 📈 Visualizations
- Core-periphery network diagrams
- Historical value transfer trends
- Smile curves (GVC value distribution)
- Exploitation metrics comparisons
- Terms of trade charts
- Policy simulation trajectories

### 🔧 Policy Simulations
- South-South cooperation scenarios
- Delinking strategies (moderate, radical, selective)
- Industrial policy packages
- Alternative integration models
- Comparative scenario analysis

### 💻 Interactive Application
- PyQt6 GUI for exploratory analysis
- Data loading and generation
- Model execution and parameter tuning
- Real-time visualization
- Results export (CSV, PDF)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/unequal-exchange-framework
cd unequal-exchange-framework

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
PyQt6>=6.0.0
```

## Quick Start

### 1. Generate Synthetic Data

```python
from unequal_exchange.data.synthetic_generator import generate_datasets

# Generate complete historical dataset (1960-2020)
datasets = generate_datasets(output_dir='./data/')

# Available datasets:
# - gdp: GDP and growth rates by country
# - trade: Bilateral trade flows by sector
# - wages: Wage and productivity data
# - terms_of_trade: Price indices and ToT
# - financial: FDI, debt, profit repatriation, IP payments
```

### 2. Run Emmanuel Model

```python
from unequal_exchange.models.emmanuel import EmmanuelModel, EmmanuelParameters
from unequal_exchange.core.theoretical_base import ProductionData, CountryCategory

# Initialize model
params = EmmanuelParameters(global_profit_rate=0.15)
model = EmmanuelModel(parameters=params)

# Add countries
usa_data = ProductionData(
    gross_output=1000,
    labor_hours=100,
    wage_rate=50,
    capital_stock=500,
    intermediate_inputs=400
)
model.add_country('USA', CountryCategory.CORE, usa_data)

bangladesh_data = ProductionData(
    gross_output=100,
    labor_hours=120,
    wage_rate=5,
    capital_stock=50,
    intermediate_inputs=40
)
model.add_country('Bangladesh', CountryCategory.PERIPHERY, bangladesh_data)

# Set trade flows
import pandas as pd
trade_matrix = pd.DataFrame({
    'USA': [0, 50],
    'Bangladesh': [80, 0]
}, index=['USA', 'Bangladesh'])
model.set_trade_flows(trade_matrix)

# Calculate value transfers
transfers = model.calculate_value_transfers()
print(transfers)

# Get summary statistics
stats = model.get_summary_statistics()
print(f"Core net gain: ${stats['core_net_gain']:.2f}M")
print(f"Periphery net loss: ${stats['periphery_net_loss']:.2f}M")
```

### 3. Analyze Global Value Chains

```python
from unequal_exchange.analysis.gvc_rents import GVCRentExtractor, ValueChainSegment

# Create GVC analyzer
analyzer = GVCRentExtractor()

# Define iPhone value chain
iphone_chain = [
    ValueChainSegment(
        name="R&D/Design", country="USA",
        value_added=300, labor_cost=50,
        capital_intensity=0.8, market_power=0.9,
        barriers_to_entry=0.9, ip_intensity=0.95
    ),
    ValueChainSegment(
        name="Manufacturing", country="China",
        value_added=50, labor_cost=40,
        capital_intensity=0.6, market_power=0.2,
        barriers_to_entry=0.3, ip_intensity=0.1
    ),
    ValueChainSegment(
        name="Marketing/Retail", country="USA",
        value_added=250, labor_cost=30,
        capital_intensity=0.5, market_power=0.95,
        barriers_to_entry=0.9, ip_intensity=0.8
    )
]

analyzer.add_value_chain("iPhone", iphone_chain)

# Analyze value distribution
distribution = analyzer.calculate_value_distribution("iPhone")
print(distribution)

# Analyze smile curve
smile = analyzer.analyze_smile_curve("iPhone")
print(f"Smile intensity: {smile['smile_intensity']:.2f}")

# Calculate lead firm extraction
lead_firm = analyzer.calculate_lead_firm_extraction(
    "iPhone",
    ["R&D/Design", "Marketing/Retail"]
)
print(f"Lead firm value share: {lead_firm['lead_firm_value_share']:.1f}%")
```

### 4. Multi-Country Input-Output Analysis

```python
from unequal_exchange.io_framework.multi_country import MultiCountryIOTable, IOTableMetadata

# Define countries and sectors
metadata = IOTableMetadata(
    countries=['USA', 'China', 'Germany'],
    sectors=['Agriculture', 'Manufacturing', 'Services'],
    year=2020
)

# Create IO table
io_table = MultiCountryIOTable(metadata)

# Set matrices (example with random data)
import numpy as np
Z = np.random.rand(9, 9) * 100  # 3 countries × 3 sectors
F = np.random.rand(9, 3) * 50
VA = np.random.rand(9) * 30
L = np.random.rand(9) * 1000  # Labor hours

io_table.set_intermediate_use(Z)
io_table.set_final_demand(F)
io_table.set_value_added(VA)
io_table.set_labor_input(L)

# Calculate gross output
x = io_table.calculate_gross_output()

# Calculate Leontief inverse
B = io_table.calculate_leontief_inverse()

# Decompose value added
va_decomp = io_table.decompose_value_added()
print("Value added by source embedded in final demand:")
print(va_decomp)

# Calculate embodied labor
labor_decomp = io_table.calculate_embodied_labor()

# Analyze GVC participation
gvc_participation = io_table.calculate_gvc_participation()
print(gvc_participation)

# Analyze value appropriation
value_appropriation = io_table.calculate_value_appropriation()
print(value_appropriation)
```

### 5. Policy Simulations

```python
from unequal_exchange.policy.simulations import PolicySimulator

# Initialize simulator
simulator = PolicySimulator()

# Simulate South-South cooperation
south_countries = ['Brazil', 'India', 'Nigeria', 'Indonesia']
cooperation_results = simulator.simulate_south_south_cooperation(
    south_countries=south_countries,
    cooperation_intensity=0.7,  # 70% cooperation
    years=20
)

print(cooperation_results)

# Simulate delinking strategy
delinking_results = simulator.simulate_delinking(
    country='Bangladesh',
    delinking_strategy='moderate',
    years=30
)

print(f"GDP impact: {delinking_results.iloc[-1]['gdp_vs_baseline_pct']:.2f}%")
print(f"Value transfer reduction: {delinking_results.iloc[-1]['transfer_reduction']:.2f}")

# Simulate industrial policy
industrial_results = simulator.simulate_industrial_policy(
    country='Vietnam',
    policy_package='comprehensive',
    years=25
)

print(f"Manufacturing export share: {industrial_results.iloc[-1]['manufacturing_export_share']*100:.1f}%")
```

### 6. Visualizations

```python
from unequal_exchange.visualization.core_periphery_plots import CorePeripheryVisualizer

# Create visualizer
viz = CorePeripheryVisualizer(style='academic')

# Plot value transfer network
country_categories = {
    'USA': 'core', 'Germany': 'core',
    'China': 'semi_periphery', 'Brazil': 'semi_periphery',
    'Bangladesh': 'periphery', 'Nigeria': 'periphery'
}

fig = viz.plot_value_transfer_network(
    transfers_df=transfers,
    country_categories=country_categories,
    min_transfer=10
)
fig.savefig('value_transfer_network.png', dpi=300)

# Plot historical transfers
historical_data = pd.DataFrame({
    'year': list(range(1960, 2021)) * 3,
    'country': ['USA']*61 + ['China']*61 + ['Bangladesh']*61,
    'value_transfer': np.random.randn(183) * 10,
    'category': ['core']*61 + ['semi_periphery']*61 + ['periphery']*61
})

fig = viz.plot_historical_transfers(historical_data)
fig.savefig('historical_transfers.png', dpi=300)

# Plot smile curve
smile_data = pd.DataFrame({
    'segment': ['R&D', 'Component Mfg', 'Assembly', 'Marketing', 'Retail'],
    'value_added_pct': [35, 15, 10, 25, 15],
    'position': [1, 2, 3, 4, 5]
})

fig = viz.plot_smile_curve(smile_data)
fig.savefig('smile_curve.png', dpi=300)

# Plot policy comparison
fig = viz.plot_policy_comparison(
    scenarios=cooperation_results,
    metric='total_value_transfer'
)
fig.savefig('policy_comparison.png', dpi=300)
```

### 7. Launch Interactive Application

```python
# From command line:
python -m unequal_exchange.gui.main_application

# Or from Python:
from unequal_exchange.gui.main_application import main
main()
```

## Project Structure

```
unequal_exchange/
├── core/                      # Theoretical foundations
│   ├── theoretical_base.py    # Base classes for UE models
│   └── __init__.py
├── models/                    # Specific model implementations
│   ├── emmanuel.py            # Emmanuel unequal exchange model
│   ├── amin.py                # Amin extended model
│   ├── prebisch_singer.py     # Prebisch-Singer ToT model
│   └── __init__.py
├── io_framework/              # Input-output analysis
│   ├── multi_country.py       # Multi-country IO tables
│   └── __init__.py
├── analysis/                  # Analysis tools
│   ├── value_transfers.py     # Value transfer calculations
│   ├── gvc_rents.py           # GVC rent extraction
│   ├── super_exploitation.py  # Super-exploitation metrics
│   ├── transfer_pricing.py    # Transfer pricing analysis
│   └── __init__.py
├── policy/                    # Policy simulations
│   ├── simulations.py         # Policy scenario simulator
│   └── __init__.py
├── visualization/             # Visualization tools
│   ├── core_periphery_plots.py
│   └── __init__.py
├── data/                      # Data generation
│   ├── synthetic_generator.py
│   └── __init__.py
├── gui/                       # PyQt6 application
│   ├── main_application.py
│   └── __init__.py
├── README.md
└── __init__.py
```

## Theoretical Background

### Emmanuel's Unequal Exchange (1972)

Arghiri Emmanuel argued that international trade systematically transfers value from low-wage to high-wage countries. Key mechanism:
- International capital mobility → equalized profit rates globally
- Labor immobility → persistent wage differentials
- Result: "Equal" exchange at prices of production masks unequal exchange of labor

**Formula**: Value transfer = Labor exported × (Northern wage - Southern wage)

### Amin's Extended Model (1974)

Samir Amin extended Emmanuel by incorporating:
- Productivity differentials between North and South
- Super-exploitation: wages below value of labor-power even accounting for productivity
- Blocked development: periphery locked into low-productivity activities

**Total transfer** = Emmanuel transfer + Productivity gap transfer + Super-exploitation transfer

### Prebisch-Singer Thesis (1950)

Raúl Prebisch and Hans Singer independently identified secular deterioration in terms of trade for primary commodity exporters:
- Income elasticity for primary goods < manufactures (Engel's Law)
- Productivity gains in manufacturing → lower prices (passed to consumers)
- Productivity gains in primary production → lower prices (competitive markets)
- Result: Continuous income transfer from periphery to core

### Global Value Chains

Modern extension analyzing:
- Smile curve: High value at R&D/design and marketing, low value in manufacturing
- Lead firm governance and rent extraction
- Intellectual monopoly (Durand & Milberg)
- Platform capitalism rents

## Academic Use

This framework is designed for:
- **Teaching**: Heterodox economics courses (political economy, development)
- **Research**: Empirical analysis of unequal exchange, replication studies
- **Policy Analysis**: Evaluating alternative development strategies

### Citing This Work

```bibtex
@software{unequal_exchange_framework,
  title = {Unequal Exchange Framework: Computational Tools for Dependency Theory},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/unequal-exchange-framework}
}
```

## References

### Core Texts

1. **Emmanuel, A. (1972)**. *Unequal Exchange: A Study of the Imperialism of Trade*. Monthly Review Press.

2. **Amin, S. (1974)**. *Accumulation on a World Scale*. Monthly Review Press.

3. **Prebisch, R. (1950)**. *The Economic Development of Latin America and its Principal Problems*. ECLA.

4. **Singer, H. (1950)**. "The Distribution of Gains from Trade between Investing and Borrowing Countries." *American Economic Review*.

### Contemporary Research

5. **Hickel, J., Sullivan, D., & Zoomkawala, H. (2021)**. "Plunder in the Post-Colonial Era." *New Political Economy*.

6. **Smith, J. (2016)**. *Imperialism in the Twenty-First Century*. Monthly Review Press.

7. **Cope, Z. (2019)**. *The Wealth of (Some) Nations*. Pluto Press.

8. **Durand, C., & Milberg, W. (2020)**. "Intellectual Monopoly in Global Value Chains." *Review of International Political Economy*.

## Contributing

Contributions welcome! Areas for development:
- Real data integration (WIOD, OECD, UN Comtrade)
- Additional models (Wallerstein WSA, Frank underdevelopment)
- Enhanced visualizations
- Econometric estimation routines

## License

MIT License - See LICENSE file

## Contact

For questions, suggestions, or collaboration:
- GitHub Issues: [github.com/yourusername/unequal-exchange-framework/issues]
- Email: your.email@example.com

---

**Acknowledgments**: This framework builds on decades of work by heterodox economists challenging neoclassical trade theory. It aims to make dependency theory accessible for computational analysis and policy evaluation.
```