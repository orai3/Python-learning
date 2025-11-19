# Heterodox Macro Dashboard

**Version 1.0.0**

A production-ready PyQt6 application for academic economic research using heterodox theoretical frameworks.

## Overview

The Heterodox Macro Dashboard is an academic research tool that enables economists and researchers to analyze macroeconomic data through multiple heterodox theoretical lenses. Unlike mainstream economic tools that typically embody a single theoretical perspective, this dashboard embraces **economic pluralism** by implementing three major heterodox schools:

- **Post-Keynesian Economics** - Stock-flow consistent modeling, endogenous money, effective demand
- **Marxian Political Economy** - Class analysis, exploitation, crisis tendencies
- **Institutionalist Economics** - Power relations, institutional change, comparative systems

## Features

### 1. Data Management
- Load pre-configured economic datasets (macro, inequality, sectoral balances, crisis data)
- Import custom CSV files
- Data exploration and summary statistics
- Export processed datasets in multiple formats (CSV, Excel, Stata)

### 2. Theoretical Framework Analysis
- **Post-Keynesian Analysis**
  - Sectoral financial balances (Godley approach)
  - Wage and profit share dynamics
  - Financial fragility indicators (Minsky)
  - Capacity utilization and effective demand

- **Marxian Analysis**
  - Rate of profit and tendency to fall
  - Rate of surplus value (exploitation)
  - Organic composition of capital
  - Reserve army of labor effects

- **Institutionalist Analysis**
  - Financialization indicators
  - Power relations and inequality
  - Government size and countervailing power
  - Cumulative causation patterns

### 3. Publication-Quality Visualizations
- Time series charts with multiple variables
- Sectoral balance diagrams
- Lorenz curves for inequality analysis
- Correlation matrices
- Rate of profit with trend analysis
- Wage vs profit share dynamics

### 4. Comprehensive Report Generation
- Multi-framework comparative analysis
- Theoretical interpretations
- Policy implications
- Data methodology documentation
- Export to text files for academic use

## Installation

### Requirements
- Python 3.8 or higher
- PyQt6
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- seaborn >= 0.11.0

### Setup

1. **Clone or navigate to the repository:**
```bash
cd /path/to/Python-learning/phase4-heterodox-dashboard
```

2. **Install dependencies:**
```bash
pip install -r ../requirements.txt
```

3. **Ensure datasets are available:**
The application expects datasets in `../datasets/` relative to the application directory. The Phase 1 dataset generation should have created these files.

## Usage

### Starting the Application

```bash
python main.py
```

Or make it executable:
```bash
chmod +x main.py
./main.py
```

### Basic Workflow

1. **Load Data**
   - On startup, the application automatically loads default datasets
   - Or use `File > Load Default Datasets`
   - Import custom data via `File > Load Custom Dataset`

2. **Explore Data**
   - Navigate to the "Data Management" tab
   - Select datasets from dropdown
   - View summary statistics and data preview
   - Export datasets if needed

3. **Run Framework Analysis**
   - Navigate to "Framework Analysis" tab
   - Select a theoretical framework (Post-Keynesian, Marxian, or Institutionalist)
   - Click "Run Analysis"
   - View results in three sub-tabs:
     - **Analysis Results**: Quantitative indicators and interpretations
     - **Visualizations**: Generate charts for your analysis
     - **Theoretical Background**: Read about the framework's foundations

4. **Generate Reports**
   - Navigate to "Report Generation" tab
   - Select which frameworks to include
   - Choose additional sections (theory, data summary)
   - Click "Generate Report"
   - Export to text file

### Menu Options

**File Menu:**
- Load Default Datasets (Ctrl+L)
- Load Custom Dataset
- Exit (Ctrl+Q)

**Analysis Menu:**
- Quick access to each framework analysis
- Compare All Frameworks

**Help Menu:**
- About
- User Guide
- Theoretical Background

## Architecture

The application follows a **Model-View-Controller (MVC)** pattern:

```
phase4-heterodox-dashboard/
├── main.py                 # Application entry point
├── models/                 # Data models and business logic
│   ├── data_model.py       # Dataset loading and management
│   └── frameworks.py       # Theoretical framework implementations
├── views/                  # PyQt6 UI components
│   ├── main_window.py      # Main application window
│   ├── data_view.py        # Data management interface
│   ├── analysis_view.py    # Framework analysis interface
│   └── report_view.py      # Report generation interface
├── controllers/            # Application controllers
│   ├── data_controller.py  # Data operations controller
│   └── analysis_controller.py  # Analysis operations controller
└── utils/                  # Utility modules
    ├── visualizations.py   # Chart generation
    └── calculations.py     # Economic calculations
```

## Theoretical Foundations

### Post-Keynesian Economics

**Core Principles:**
- Effective demand determines output (Say's Law rejected)
- Money is endogenous and credit-driven
- Fundamental uncertainty (not probabilistic risk)
- Historical time and path dependence
- Distribution affects aggregate demand

**Key Models:**
- Kalecki profit equation: `Profits = Investment + Capitalist Consumption - Worker Saving`
- Godley sectoral balances: `Private + Government + Foreign = 0`
- Minsky's Financial Instability Hypothesis

**References:**
- Lavoie, M. (2014). *Post-Keynesian Economics: New Foundations*. Edward Elgar.
- Godley, W., & Lavoie, M. (2007). *Monetary Economics*. Palgrave Macmillan.
- Kalecki, M. (1971). *Selected Essays on the Dynamics of the Capitalist Economy*.

### Marxian Political Economy

**Core Principles:**
- Labor theory of value
- Surplus value extraction (exploitation)
- Capital accumulation driven by profit
- Tendency of rate of profit to fall
- Class struggle over distribution

**Key Indicators:**
- Rate of profit: `r = s/(c+v)` where s=surplus, c=constant capital, v=variable capital
- Rate of surplus value: `e = s/v` (exploitation rate)
- Organic composition of capital: `OCC = c/v`

**References:**
- Marx, K. (1867/1990). *Capital, Volume I*. Penguin Classics.
- Shaikh, A. (2016). *Capitalism: Competition, Conflict, Crises*. Oxford University Press.
- Foley, D. (1986). *Understanding Capital*. Harvard University Press.

### Institutionalist Economics

**Core Principles:**
- Institutions shape economic behavior
- Power relations drive outcomes
- Social provisioning (not utility maximization)
- Cumulative causation and path dependence
- Technological-institutional co-evolution

**Key Concepts:**
- Ceremonial vs instrumental behavior (Veblen)
- Countervailing power (Galbraith)
- Cumulative causation (Myrdal)
- Varieties of capitalism (comparative analysis)

**References:**
- Veblen, T. (1899). *The Theory of the Leisure Class*.
- Galbraith, J.K. (1967). *The New Industrial State*.
- Hodgson, G. (2015). *Conceptualizing Capitalism*. University of Chicago Press.

## Data Sources

The application is designed to work with:

1. **Included Synthetic Datasets** (Phase 1)
   - Macro quarterly data
   - Inequality annual data
   - Sectoral balances (SFC)
   - Financial crisis data
   - Cross-country panel data

2. **Real Data Sources** (for future enhancement)
   - FRED (Federal Reserve Economic Data)
   - OECD statistics
   - National statistical agencies
   - BIS (Bank for International Settlements)
   - Penn World Table

## Example Workflows

### Workflow 1: Analyzing Distribution and Demand Regime

1. Load macro and inequality datasets
2. Run Post-Keynesian analysis
3. Generate "Wage vs Profit Share" visualization
4. Check capacity utilization indicator
5. Interpretation: Is the economy wage-led or profit-led?

### Workflow 2: Assessing Financial Fragility

1. Load macro and crisis datasets
2. Run Post-Keynesian analysis
3. Examine debt-to-GDP trends
4. Generate sectoral balances chart
5. Check Minsky fragility indicators
6. Export analysis report

### Workflow 3: Comparative Framework Analysis

1. Load all default datasets
2. Use "Compare All Frameworks" from Analysis menu
3. Review how different schools interpret same data
4. Generate comprehensive report with all frameworks
5. Export for academic paper or teaching materials

## Academic Use Cases

### Research
- Multi-framework analysis for academic papers
- Replication studies using different theoretical lenses
- Comparative institutional analysis
- Historical economic analysis

### Teaching
- Demonstrate heterodox economic theories with real data
- Compare mainstream vs heterodox interpretations
- Interactive classroom demonstrations
- Student projects and assignments

### Policy Analysis
- Evaluate policies from multiple perspectives
- Understand distributional implications
- Assess financial stability risks
- Compare international experiences

## Extending the Application

### Adding New Datasets

```python
# In Data Management tab, use "Load Custom CSV"
# Or programmatically:
data_controller.load_dataset('new_data', '/path/to/file.csv')
```

### Adding New Frameworks

Create a new class inheriting from `EconomicFramework`:

```python
class NewFramework(EconomicFramework):
    def get_name(self) -> str:
        return "New Framework"

    def get_key_indicators(self) -> List[str]:
        return ["Indicator1", "Indicator2"]

    def analyze(self, data_model) -> Dict:
        # Implement analysis logic
        pass

    def get_theoretical_notes(self) -> str:
        # Return theoretical background
        pass
```

### Adding New Visualizations

Add methods to `ChartGenerator` in `utils/visualizations.py`:

```python
def plot_new_chart(self, data: pd.DataFrame) -> Figure:
    # Create matplotlib figure
    pass
```

## Citation

If you use this tool in academic research, please cite:

```
Heterodox Macro Dashboard (2025). Version 1.0.0.
Academic research tool for pluralist economic analysis.
```

## License

MIT License - Free for academic and educational use

## Contributing

Contributions welcome! Areas for enhancement:
- Additional theoretical frameworks (Ecological Economics, Feminist Economics)
- Real-time data fetching from FRED, OECD APIs
- Advanced econometric methods
- Machine learning integration
- Web-based version

## Support

For questions, bug reports, or feature requests:
- Create an issue in the repository
- Contact: [Add contact information]

## Acknowledgments

This application builds on the work of generations of heterodox economists who have challenged mainstream orthodoxy and developed rich alternative frameworks for understanding economic reality.

Special thanks to the economists whose theories are implemented here:
- Michal Kalecki, Joan Robinson, Nicholas Kaldor, Hyman Minsky, Wynne Godley, Marc Lavoie
- Karl Marx, Rosa Luxemburg, Paul Sweezy, Anwar Shaikh, Duncan Foley, David Harvey
- Thorstein Veblen, John Commons, John Kenneth Galbraith, Gunnar Myrdal, Ha-Joon Chang

## Version History

**1.0.0** (2025-11-19)
- Initial release
- Three theoretical frameworks (PK, Marxian, Institutionalist)
- Full data management and analysis capabilities
- Publication-quality visualizations
- Comprehensive report generation
