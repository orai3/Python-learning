# Historical Economic Data Analysis Toolkit

**Advanced Python toolkit for analyzing long-run capitalist development**

A comprehensive, production-ready toolkit for heterodox economic analysis of historical capitalism. Implements cutting-edge methods from Regulation School, Marxian economics, World-Systems Theory, and critical political economy.

## Overview

This toolkit provides:

- **Synthetic Data Generation**: Realistic 150+ year datasets with structural breaks, long waves, crises
- **Periodization Algorithms**: Statistical detection of regime changes and structural breaks
- **Long Wave Analysis**: Kondratiev cycle identification, spectral analysis, multi-frequency decomposition
- **Crisis Analysis**: Detection, severity measurement, clustering analysis, recovery dynamics
- **Hegemonic Transition Analysis**: Arrighi-style systemic cycles, power transition detection
- **Replication Studies**: Brenner, Arrighi, Duménil-Lévy frameworks implemented
- **Advanced Visualization**: Publication-quality plots for academic research

## Theoretical Foundations

### 1. Regulation School
- **Regimes of accumulation**: Fordism, post-Fordism, neoliberalism
- **Modes of regulation**: Institutional configurations coordinating capitalism
- **Periodization**: Extensive vs intensive accumulation, competitive vs monopolistic regulation

**Key scholars**: Michel Aglietta, Robert Boyer, Alain Lipietz

### 2. Long Wave Theory
- **Kondratiev waves**: ~50-60 year cycles driven by technological revolutions
- **Schumpeterian innovation cycles**: Creative destruction and clustering
- **Mandel's late capitalism**: Modified long wave theory incorporating class struggle

**Key scholars**: Nikolai Kondratiev, Joseph Schumpeter, Ernest Mandel, Carlota Perez

### 3. World-Systems Theory / Hegemonic Cycles
- **Systemic cycles of accumulation**: Material expansion → Financial expansion (signal crisis)
- **Hegemonic transitions**: Rise and decline of successive hegemons
- **Center-periphery dynamics**: Unequal exchange and hierarchical world economy

**Key scholars**: Giovanni Arrighi, Immanuel Wallerstein, Fernand Braudel

### 4. Marxian Political Economy
- **Law of tendential fall in profit rate**: Central contradiction of capitalism
- **Crisis theory**: Overaccumulation, realization crises, profit squeeze
- **Class struggle**: Labor/capital conflict shaping accumulation dynamics
- **Financialization**: As response to profitability crisis

**Key scholars**: Karl Marx, David Harvey, Robert Brenner, Gérard Duménil & Dominique Lévy

### 5. Post-Keynesian Economics
- **Stock-Flow Consistent modeling**: Rigorous accounting framework
- **Endogenous money**: Credit-driven dynamics
- **Kalecki/Minsky**: Profit equation, financial instability hypothesis

**Key scholars**: Hyman Minsky, Wynne Godley, Marc Lavoie, Thomas Palley

## Installation

```bash
# Clone repository
cd historical-economics-toolkit

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
import sys
sys.path.append('modules')

from data_generator import HistoricalEconomicDataGenerator
from periodization import RegulationSchoolPeriodization, StructuralBreakDetector
from long_wave_analysis import LongWaveAnalyzer
from crisis_hegemony import CrisisAnalyzer, HegemonyAnalyzer
from visualization import HistoricalPlotter

# Generate synthetic data (150 years, 5 countries)
generator = HistoricalEconomicDataGenerator(
    start_year=1870,
    end_year=2020,
    frequency='A'
)

countries = ['USA', 'UK', 'Germany', 'France', 'Japan']
data = generator.generate_complete_dataset(countries=countries)

# Detect structural breaks
detector = StructuralBreakDetector(data)
breaks = detector.bai_perron_test('gdp_growth', max_breaks=5)

# Identify Kondratiev waves
wave_analyzer = LongWaveAnalyzer(data)
waves = wave_analyzer.identify_kondratiev_waves('gdp')

# Analyze crises
crisis_analyzer = CrisisAnalyzer(data)
crises = crisis_analyzer.detect_crises(variable='gdp_growth')

# Detect hegemonic transitions
hegemony_analyzer = HegemonyAnalyzer(data)
transitions = hegemony_analyzer.detect_hegemonic_transitions()

# Visualize
plotter = HistoricalPlotter(data)
fig = plotter.plot_long_run_trends(
    variables=['gdp_growth', 'wage_share', 'profit_rate'],
    log_scale=False
)
fig.savefig('../docs/long_run_trends.png', dpi=300)
```

## Module Documentation

### 1. Data Generation (`data_generator.py`)

Generates synthetic historical economic data with realistic properties:

- **Regime-specific dynamics**: Growth rates, volatility vary by historical period
- **Long waves**: ~50-60 year Kondratiev cycles embedded
- **Crisis periods**: Realistic financial/economic crises with clustering
- **Distributional dynamics**: Wage/profit shares, inequality following historical patterns
- **Institutional variables**: Coordination indices, labor militancy, financialization

**Key class**: `HistoricalEconomicDataGenerator`

**Output variables**:
- GDP, GDP growth (with trend, cycles, crises)
- Wage share, profit share
- Gini coefficient
- Financialization index
- Labor militancy index
- Institutional coordination index
- Profit rate
- Crisis indicator
- Hegemony index

### 2. Periodization (`periodization.py`)

Detects structural breaks and regime changes:

**Statistical methods**:
- Chow test (known break points)
- CUSUM test (unknown breaks)
- Bai-Perron sequential testing
- Correlation break detection

**Economic frameworks**:
- Regulation School regime identification (cluster analysis)
- Regime-switching models (volatility-based)
- Growth regime classification

**Key classes**:
- `StructuralBreakDetector`
- `RegimeSwitchingModel`
- `RegulationSchoolPeriodization`

### 3. Long Wave Analysis (`long_wave_analysis.py`)

Analyzes Kondratiev long waves:

**Methods**:
- Spectral analysis (Fourier decomposition)
- Band-pass filtering (isolate 40-65 year cycles)
- Peak/trough dating
- Phase classification (expansion/contraction)
- Multi-frequency decomposition (Schumpeterian framework)

**Key classes**:
- `LongWaveAnalyzer`
- `SchumpeterianCycles`
- `TechnologyRevolutions`

### 4. Crisis & Hegemony Analysis (`crisis_hegemony.py`)

Crisis detection and hegemonic cycle analysis:

**Crisis analysis**:
- Algorithmic crisis detection
- Severity measurement (peak-to-trough, duration, output loss)
- Clustering analysis
- Frequency by period
- Systemic crisis identification

**Hegemony analysis**:
- Arrighi's systemic cycles
- Hegemonic transition detection
- Material vs financial expansion phases
- Interregnum periods

**Key classes**:
- `CrisisAnalyzer`
- `HegemonyAnalyzer`

### 5. Visualization (`visualization.py`)

Publication-quality plotting:

**Plot types**:
- Long-run trends with regime shading
- Crisis timelines
- Kondratiev wave decomposition
- Hegemonic cycle diagrams
- Distribution dynamics (wage share, inequality)
- Profit squeeze phase diagrams
- Comparative cross-country analysis
- Summary dashboards

**Key class**: `HistoricalPlotter`

### 6. Replication Studies (`replications/major_studies.py`)

Implements major heterodox frameworks:

#### Robert Brenner - Long Downturn Thesis
- Profit rate trend analysis (1945-present)
- Periodization: Golden Age → Long Downturn → Neoliberal (incomplete restoration)
- Investment-profit relationship

#### Giovanni Arrighi - Systemic Cycles
- Historical cycles (Genoa → Dutch → British → US)
- Material vs financial expansion phases
- US hegemonic decline analysis

#### Gérard Duménil & Dominique Lévy - Class Power
- Structural crises (1970s, 2008)
- Neoliberal restoration analysis
- Class power indices (labor vs capital)

**Key classes**:
- `BrennerAnalysis`
- `ArrighiAnalysis`
- `DumenilLevyAnalysis`

## Example Analyses

### Example 1: Detecting the Neoliberal Transition

```python
from periodization import RegulationSchoolPeriodization

# Identify regimes using institutional variables
periodization = RegulationSchoolPeriodization(data)

results = periodization.identify_regimes(
    variables=['wage_share', 'financialization', 'institutional_coordination', 'labor_militancy'],
    n_regimes=4
)

# Label regimes
labeled = periodization.label_historical_regimes(results['regime_characteristics'])
print(labeled)
```

### Example 2: Replicating Brenner's Long Downturn

```python
from replications.major_studies import BrennerAnalysis

brenner = BrennerAnalysis(data)

# Analyze profit rate trends
profit_analysis = brenner.calculate_profit_rate_trend(
    profit_rate_var='profit_rate',
    start_year=1945
)

print(profit_analysis['interpretation'])

# Investment-profit relationship
inv_profit = brenner.analyze_investment_profit_relationship()
print(inv_profit['correlation_analysis'])
```

### Example 3: Arrighi's US Hegemonic Cycle

```python
from replications.major_studies import ArrighiAnalysis

arrighi = ArrighiAnalysis(data)

# Analyze US cycle
us_cycle = arrighi.analyze_us_cycle(start_year=1945)
print(us_cycle)

# Identify accumulation phases
phases = arrighi.identify_accumulation_phases()
print(phases['phase_periods'])
```

### Example 4: Crisis Clustering Analysis

```python
from crisis_hegemony import CrisisAnalyzer

crisis_analyzer = CrisisAnalyzer(data)

# Detect all crises
crises = crisis_analyzer.detect_crises(threshold=-0.02)

# Analyze clustering
clustering = crisis_analyzer.analyze_crisis_clustering(crises)
print(f"Coefficient of variation: {clustering['coefficient_of_variation']}")

# Crisis frequency by period
freq = crisis_analyzer.crisis_frequency_by_period(crises)
print(freq)
```

## Data Dictionary

### Generated Variables

| Variable | Description | Range | Source/Method |
|----------|-------------|-------|---------------|
| `gdp` | Real GDP (log scale) | Continuous | Trend + long wave + cycles + shocks |
| `gdp_growth` | GDP growth rate | -0.3 to 0.15 | Percent change |
| `wage_share` | Labor share of income | 0.5-0.7 | Regime-specific + Goodwin dynamics |
| `profit_share` | Capital share (1 - wage_share) | 0.3-0.5 | Residual |
| `gini` | Gini coefficient | 0.25-0.6 | Historical pattern |
| `financialization` | Financial sector size/importance | 0.15-0.75 | Regime-specific trend |
| `labor_militancy` | Strike activity index | 0.05-0.85 | Historical pattern |
| `institutional_coordination` | Degree of regulation | 0.15-0.85 | Regime-specific |
| `profit_rate` | Rate of profit | 0.03-0.20 | Marx-Kalecki framework |
| `crisis` | Binary crisis indicator | 0 or 1 | GDP growth threshold |
| `hegemony` | Hegemonic power index | 0.25-0.95 | Arrighi framework |

## Academic Applications

### Research Uses

1. **Testing heterodox theories**: Empirical validation of theoretical propositions
2. **Periodization studies**: Identifying regime changes, structural breaks
3. **Comparative political economy**: Cross-country institutional analysis
4. **Crisis prediction**: Leading indicators, early warning systems
5. **Long-run dynamics**: Secular trends, cyclical patterns
6. **Distributional analysis**: Inequality, functional distribution over time

### Teaching Applications

1. **Graduate seminars**: Heterodox macro, political economy, economic history
2. **Methods courses**: Time series, structural breaks, regime switching
3. **Workshops**: Introduce students to Python for economic analysis
4. **Thesis/dissertation**: Research toolkit for historical analysis

### Publication Potential

1. **Software paper**: Journal of Open Source Software, Review of Keynesian Economics
2. **Methodological contribution**: Comparative periodization methods
3. **Substantive analysis**: Using toolkit for original empirical research
4. **Teaching materials**: Pedagogy in heterodox economics

## Theoretical References

### Core Texts

**Regulation School**:
- Aglietta, M. (1979). *A Theory of Capitalist Regulation*. Verso.
- Boyer, R., & Saillard, Y. (2002). *Regulation Theory: The State of the Art*. Routledge.

**Long Waves**:
- Mandel, E. (1980). *Long Waves of Capitalist Development*. Cambridge UP.
- Perez, C. (2002). *Technological Revolutions and Financial Capital*. Edward Elgar.
- Freeman, C., & Louçã, F. (2001). *As Time Goes By*. Oxford UP.

**World-Systems / Hegemony**:
- Arrighi, G. (1994). *The Long Twentieth Century*. Verso.
- Wallerstein, I. (2004). *World-Systems Analysis*. Duke UP.

**Marxian Political Economy**:
- Brenner, R. (2006). *The Economics of Global Turbulence*. Verso.
- Duménil, G., & Lévy, D. (2004). *Capital Resurgent*. Harvard UP.
- Duménil, G., & Lévy, D. (2011). *The Crisis of Neoliberalism*. Harvard UP.
- Harvey, D. (2005). *A Brief History of Neoliberalism*. Oxford UP.

**Post-Keynesian**:
- Minsky, H. (1986). *Stabilizing an Unstable Economy*. Yale UP.
- Godley, W., & Lavoie, M. (2007). *Monetary Economics*. Palgrave.

## Extensions & Future Development

**Potential additions**:

1. **Agent-based modeling**: Simulate capitalist dynamics from micro foundations
2. **Stock-Flow Consistent models**: Implement Godley-Lavoie framework
3. **Input-output analysis**: Leontief/Sraffa inter-industry analysis
4. **Ecological economics**: Material throughput, energy flows, environmental limits
5. **Global value chains**: Contemporary imperialism, unequal exchange
6. **Real data integration**: FRED, OECD, national statistical offices
7. **Interactive dashboards**: Streamlit/Dash web applications
8. **Machine learning**: Pattern recognition in historical data

## Contributing

This is an open academic project. Contributions welcome:

- Additional replication studies
- New theoretical frameworks
- Improved statistical methods
- Bug fixes and optimizations
- Documentation improvements
- Example analyses

## Citation

If you use this toolkit in academic work, please cite:

```
Historical Economic Data Analysis Toolkit (2024)
Advanced Python toolkit for analyzing long-run capitalist development
https://github.com/[your-repo]
```

## License

MIT License - Free for academic and educational use

## Contact

For questions, collaborations, or academic use:
- Issues: GitHub issue tracker
- Discussions: GitHub discussions

---

**Note**: This toolkit generates *synthetic* data for methodological development and teaching. For substantive research, use real historical data from established sources (Maddison Project, FRED, OECD, national statistics).

**Disclaimer**: Reflects heterodox/pluralist economic perspectives. Not all economists accept these theoretical frameworks. Use critically and engage with mainstream alternatives.
