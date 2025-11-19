# Quick Start Guide

## 5-Minute Quick Start

### 1. Generate Data

```python
import sys
sys.path.append('modules')

from data_generator import HistoricalEconomicDataGenerator

# Create generator
generator = HistoricalEconomicDataGenerator(
    start_year=1870,
    end_year=2020
)

# Generate data for one country
data = generator.generate_complete_dataset(countries=['USA'])

print(data.head())
print(f"\nShape: {data.shape}")
print(f"Variables: {data.columns.tolist()}")
```

### 2. Detect Crises

```python
from crisis_hegemony import CrisisAnalyzer

analyzer = CrisisAnalyzer(data)
crises = analyzer.detect_crises(variable='gdp_growth', threshold=-0.02)

print(f"\nDetected {len(crises)} crises:")
print(crises[['start_year', 'end_year', 'severity', 'duration']])
```

### 3. Identify Long Waves

```python
from long_wave_analysis import LongWaveAnalyzer

wave_analyzer = LongWaveAnalyzer(data)
waves = wave_analyzer.identify_kondratiev_waves('gdp')

print(f"\nFound {waves['n_waves']} Kondratiev waves")
print(f"Average period: {waves['average_period']:.1f} years")
```

### 4. Visualize

```python
from visualization import HistoricalPlotter
import matplotlib.pyplot as plt

plotter = HistoricalPlotter(data)

fig = plotter.plot_long_run_trends(
    variables=['gdp_growth', 'wage_share', 'profit_rate']
)

plt.show()
```

## Complete Analysis

Run the full example:

```bash
cd examples
python complete_analysis.py
```

This will:
- Generate 150 years of data for 5 countries
- Detect structural breaks
- Identify regimes (Regulation School)
- Find Kondratiev waves
- Analyze crises
- Examine hegemonic cycles
- Replicate Brenner, Arrighi, Duménil-Lévy
- Create 8+ publication-quality visualizations
- Export all results to CSV

Output saved to:
- `data/` - CSV files with results
- `docs/` - PNG visualizations (300 DPI)

## Key Concepts

### Periodization

**What**: Divide history into distinct regimes/periods
**Why**: Capitalism changes over time - different "rules of the game"
**How**: Statistical breaks + institutional analysis

```python
from periodization import RegulationSchoolPeriodization

reg_school = RegulationSchoolPeriodization(data)
regimes = reg_school.identify_regimes(
    variables=['wage_share', 'financialization',
               'institutional_coordination', 'labor_militancy'],
    n_regimes=4
)

print(regimes['regime_characteristics'])
```

### Long Waves (Kondratiev Cycles)

**What**: ~50-60 year cycles in capitalist development
**Why**: Technological revolutions, institutional change
**How**: Spectral analysis, band-pass filtering

```python
from long_wave_analysis import LongWaveAnalyzer

analyzer = LongWaveAnalyzer(data)
spectral = analyzer.spectral_analysis('gdp')

print("Dominant periods:", spectral['dominant_periods'][:3])
```

### Crisis Analysis

**What**: Identify and characterize economic crises
**Why**: Understand crisis patterns, severity, clustering
**How**: Growth thresholds, duration analysis, clustering tests

```python
from crisis_hegemony import CrisisAnalyzer

analyzer = CrisisAnalyzer(data)
crises = analyzer.detect_crises()

# Analyze clustering
clustering = analyzer.analyze_crisis_clustering(crises)
print(f"Clustering: {clustering['clustering_interpretation']}")

# Frequency by period
freq = analyzer.crisis_frequency_by_period(crises)
print(freq)
```

### Hegemonic Cycles

**What**: Rise and decline of dominant powers in world system
**Why**: Understand power transitions, systemic change
**How**: Arrighi's framework - material/financial expansion phases

```python
from crisis_hegemony import HegemonyAnalyzer

analyzer = HegemonyAnalyzer(data)

# Detect transitions
transitions = analyzer.detect_hegemonic_transitions()

# Classify accumulation phases
phases = analyzer.classify_accumulation_phase()

print(phases.groupby('accumulation_phase').size())
```

## Replication Studies

### Brenner: Long Downturn

```python
from replications.major_studies import BrennerAnalysis

brenner = BrennerAnalysis(data)
profit_trends = brenner.calculate_profit_rate_trend()

print(profit_trends['interpretation'])
print(profit_trends['periods'])
```

### Arrighi: Systemic Cycles

```python
from replications.major_studies import ArrighiAnalysis

arrighi = ArrighiAnalysis(data)

# Get historical cycles
cycles = arrighi.get_systemic_cycles()
print(cycles)

# Analyze current cycle
us_cycle = arrighi.analyze_us_cycle()
print(us_cycle)
```

### Duménil-Lévy: Class Power

```python
from replications.major_studies import DumenilLevyAnalysis

dl = DumenilLevyAnalysis(data)

# Structural crises
crises = dl.identify_structural_crises()
print(crises['structural_crises'])

# Neoliberal restoration
neo = dl.analyze_neoliberal_restoration()
print(neo['interpretation'])
```

## Common Workflows

### Workflow 1: Periodization Study

```python
# 1. Generate data
data = generator.generate_complete_dataset(['USA'])

# 2. Detect breaks
from periodization import StructuralBreakDetector
detector = StructuralBreakDetector(data)
breaks = detector.bai_perron_test('gdp_growth', max_breaks=5)

# 3. Identify regimes
from periodization import RegulationSchoolPeriodization
reg_school = RegulationSchoolPeriodization(data)
regimes = reg_school.identify_regimes(n_regimes=4)

# 4. Label regimes
labeled = reg_school.label_historical_regimes(
    regimes['regime_characteristics']
)

# 5. Visualize
plotter = HistoricalPlotter(data)
fig = plotter.plot_long_run_trends(
    variables=['gdp_growth', 'wage_share'],
    regime_periods=labeled.to_dict('records')
)
```

### Workflow 2: Crisis Study

```python
# 1. Detect crises
analyzer = CrisisAnalyzer(data)
crises = analyzer.detect_crises()

# 2. Analyze characteristics
systemic = analyzer.identify_systemic_crises(crises)
clustering = analyzer.analyze_crisis_clustering(crises)
freq = analyzer.crisis_frequency_by_period(crises)

# 3. Visualize
plotter = HistoricalPlotter(data)
fig = plotter.plot_crisis_timeline(crises)
```

### Workflow 3: Long Wave Study

```python
# 1. Spectral analysis
wave_analyzer = LongWaveAnalyzer(data)
spectral = wave_analyzer.spectral_analysis('gdp')

# 2. Identify waves
waves = wave_analyzer.identify_kondratiev_waves('gdp')

# 3. Phase classification
phases = wave_analyzer.phase_classification('gdp')

# 4. Schumpeterian decomposition
schump = SchumpeterianCycles(data)
decomp = schump.extract_all_cycles('gdp')

# 5. Visualize
plotter = HistoricalPlotter(data)
fig = plotter.plot_kondratiev_decomposition(
    'gdp',
    waves['long_wave_series'],
    waves['waves']
)
```

## Tips & Best Practices

### For Research

1. **Start with theory**: Which framework guides your analysis?
2. **Use real data**: Replace synthetic data with historical sources
3. **Validate results**: Compare with historical record
4. **Document assumptions**: Be transparent about methods
5. **Sensitivity analysis**: Test robustness to parameter choices

### For Teaching

1. **Start simple**: One variable, one method
2. **Build complexity**: Add variables, compare methods
3. **Connect to history**: Match results to historical events
4. **Encourage critique**: Compare heterodox vs mainstream
5. **Hands-on practice**: Students run their own analyses

### Troubleshooting

**Problem**: No breaks detected
**Solution**: Adjust thresholds, try different variables, longer time series

**Problem**: Too many breaks detected
**Solution**: Increase minimum segment length, stricter significance levels

**Problem**: Waves not found
**Solution**: Check time series length (need 100+ years), try different variables

**Problem**: Visualizations cluttered
**Solution**: Reduce number of regimes/periods shown, simplify plot

## Next Steps

1. **Explore**: Run `examples/complete_analysis.py`
2. **Experiment**: Modify parameters, try different variables
3. **Adapt**: Apply to your research questions
4. **Extend**: Add new methods, frameworks
5. **Contribute**: Share improvements via GitHub

## Resources

- **README.md**: Full documentation
- **examples/**: Complete working examples
- **modules/**: Source code with docstrings
- **docs/**: Generated visualizations

## Support

- Check documentation in module docstrings
- Review example scripts
- See theoretical references in README.md
- Open GitHub issue for bugs/questions

---

**Happy analyzing! Bring critical political economy to your research.**
