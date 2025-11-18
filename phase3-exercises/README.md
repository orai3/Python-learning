# Phase 3: Analysis Exercise Library - Heterodox Economics

## Overview

This directory contains a comprehensive collection of Python exercises for economic analysis from heterodox perspectives (Post-Keynesian, Marxian, Structuralist, Institutionalist). Each exercise progresses from basic to research-grade analysis.

**Total Exercises:** 8 comprehensive modules covering 6 major areas
**Theoretical Frameworks:** Post-Keynesian, Kaleckian, Minskyan, Sraffian, Godley SFC
**Skill Level:** Intermediate to Advanced

---

## Directory Structure

```
phase3-exercises/
├── time_series/
│   └── 01_trend_extraction_exercises.py
├── distributional/
│   └── 02_inequality_analysis_exercises.py
├── sectoral/
│   └── 03_sectoral_analysis_exercises.py
├── financial/
│   └── 04_financial_cycles_exercises.py
├── sfc_accounting/
│   └── 05_sfc_accounting_exercises.py
├── input_output/
│   └── 06_input_output_exercises.py
├── goodwin_cycles/
│   └── (included in 06_input_output_exercises.py)
└── README.md (this file)
```

---

## Module Descriptions

### 1. Time Series Analysis (`time_series/`)

**File:** `01_trend_extraction_exercises.py`

**Focus:** Analyzing economic time series with structural change emphasis

**Exercises:**
- **Exercise 1:** Hodrick-Prescott Filter (with heterodox critique)
  - Implementation from scratch
  - Multiple λ sensitivity analysis
  - Post-Keynesian critique of mechanical filtering

- **Exercise 2:** Structural Breaks (Chow Test)
  - Wage share regime changes (neoliberal transition)
  - F-statistics across potential break points
  - Kaleckian analysis of distributional regimes

- **Exercise 3:** Rolling Window Analysis
  - Time-varying investment-capacity relationship
  - Financialization effects on accelerator mechanism
  - Confidence interval estimation

**Key Concepts:** HP filter, Chow test, rolling regressions, structural breaks, regime shifts

**Heterodox Themes:**
- Rejection of stable parameters assumption
- Importance of institutional change
- Path dependence and hysteresis
- Critique of NAIRU and natural rate concepts

**Extension Challenges:**
- Bai-Perron multiple break test
- Hamilton filter alternative
- State-space models with time-varying NAIRU
- Markov-switching regime models
- Wavelet analysis for time-frequency relationships

**Run Time:** ~30 seconds
**Output:** 3 high-quality visualizations

---

### 2. Distributional Analysis (`distributional/`)

**File:** `02_inequality_analysis_exercises.py`

**Focus:** Inequality measurement and interpretation

**Exercises:**
- **Exercise 1:** Lorenz Curves & Gini Coefficient
  - Multiple historical scenarios (Nordic, Post-War, Contemporary, Gilded Age)
  - Gini calculation from first principles
  - Percentile share decomposition (P10, P50, P90, P99)

- **Exercise 2:** Palma Ratio & Alternative Measures
  - Comparison across inequality metrics
  - Sensitivity analysis to distributional changes
  - Bottom squeeze, top surge, middle erosion scenarios
  - Atkinson index, Theil index, decile ratios

**Key Concepts:** Lorenz curve, Gini coefficient, Palma ratio, Atkinson index, functional vs personal distribution

**Heterodox Themes:**
- Inequality as policy outcome, not market natural
- Power relations in distribution
- Wage-led vs profit-led growth regimes
- Critique of marginal productivity theory

**Extension Challenges:**
- Functional vs personal distribution decomposition (Theil)
- Generalized Entropy indices family
- Wealth concentration (Pareto distributions)
- Distributional National Accounts (DINA)
- Inequality decomposition by income source
- Inequality-growth regime modeling

**Run Time:** ~20 seconds
**Output:** 2 comprehensive visualizations

---

### 3. Sectoral Analysis (`sectoral/`)

**File:** `03_sectoral_analysis_exercises.py`

**Focus:** Structural change and sectoral dynamics

**Exercises:**
- **Exercise 1:** Shift-Share Analysis
  - Deindustrialization case study
  - Decomposition: National, Industry Mix, Competitive effects
  - Regional vs national employment dynamics

- **Exercise 2:** Structural Decomposition Analysis (SDA)
  - CO2 emissions decomposition (scale, composition, intensity)
  - LMDI (Logarithmic Mean Divisia Index) method
  - Environmental Kuznets Curve analysis

**Key Concepts:** Shift-share, structural decomposition, deindustrialization, EKC

**Heterodox Themes:**
- Structuralist development economics
- Comparative political economy
- Ecological economics integration
- Industrial policy rationale
- Just transition concerns

**Extension Challenges:**
- Multi-period shift-share
- Spatial shift-share with neighbor effects
- Full input-output SDA
- Inequality decomposition by sector
- Productivity decomposition (Baily-Hulten-Campbell)
- Consumption-based emissions accounting

**Run Time:** ~25 seconds
**Output:** 2 detailed visualizations

---

### 4. Financial Analysis (`financial/`)

**File:** `04_financial_cycles_exercises.py`

**Focus:** Financial instability and credit cycles

**Exercises:**
- **Exercise 1:** Minsky's Financial Instability Hypothesis
  - Simulation of three financing regimes (Hedge, Speculative, Ponzi)
  - Endogenous asset price boom-bust
  - Leverage dynamics and interest coverage
  - Identification of "Minsky moments"

- **Exercise 2:** Credit Cycle Analysis
  - Spectral decomposition (FFT)
  - Multiple overlapping cycles (Kitchin, Juglar, Kuznets)
  - Band-pass filtering for cycle extraction
  - Turning point identification
  - Credit-GDP correlation dynamics

**Key Concepts:** Minsky hypothesis, credit cycles, spectral analysis, financial fragility

**Heterodox Themes:**
- "Stability is destabilizing"
- Endogenous financial instability
- Rejection of efficient markets
- Finance as driver of real economy
- Money manager capitalism

**Extension Challenges:**
- Steve Keen's full Minsky model (differential equations)
- BIS credit-to-GDP gap methodology
- Financial Conditions Index (PCA)
- Debt-deflation dynamics (Irving Fisher)
- Godley sectoral financial balances
- Agent-based crisis contagion model

**Run Time:** ~35 seconds
**Output:** 2 complex visualizations

---

### 5. Stock-Flow Consistent (SFC) Accounting (`sfc_accounting/`)

**File:** `05_sfc_accounting_exercises.py`

**Focus:** SFC methodology for macroeconomic coherence

**Exercises:**
- **Exercise 1:** Sectoral Balances
  - Three-sector model (Private, Government, Foreign)
  - Fundamental identity: (S-I) + (T-G) + (M-X) ≡ 0
  - Historical simulation (US 1980-2020)
  - Godley's "unsustainable processes" analysis

- **Exercise 2:** Balance Sheet & Transaction Flow Matrices
  - Multi-sector balance sheets (Households, Firms, Banks, Govt)
  - Transaction flow matrix construction
  - Stock-flow consistency verification
  - Net worth evolution

**Key Concepts:** Sectoral balances, SFC matrices, stock-flow consistency, Godley tables

**Heterodox Themes:**
- Post-Keynesian SFC modeling
- MMT theoretical foundations
- Critique of government deficit hysteria
- Twin deficits analysis
- Financial fragility in balance sheets

**Extension Challenges:**
- Dynamic SFC model simulation (Godley & Lavoie Model SIM)
- Open economy 2-country model
- Portfolio choice extension (Tobin)
- Banking sector elaboration
- MMT framework with Job Guarantee
- Climate-integrated SFC model

**Run Time:** ~20 seconds
**Output:** 2 accounting visualizations

---

### 6. Input-Output & Goodwin Cycles (`input_output/`, `goodwin_cycles/`)

**File:** `06_input_output_exercises.py`

**Focus:** Interdependencies and distributive cycles

**Exercises:**
- **Exercise 1:** Leontief Input-Output Model
  - Technical coefficients matrix construction
  - Leontief inverse calculation
  - Output multipliers
  - Demand shock propagation

- **Exercise 2:** Goodwin Growth Cycle
  - Class struggle dynamics simulation
  - Wage share and employment rate oscillations
  - Phase diagram (limit cycle)
  - Predator-prey formulation

**Key Concepts:** I-O tables, Leontief inverse, multipliers, Goodwin cycle, functional distribution

**Heterodox Themes:**
- Structuralist interdependence
- Sraffian price theory potential
- Marxian reserve army of labor
- Endogenous cycles from class conflict
- Critique of NAIRU

**Extension Challenges:**
- Sraffian price model
- Environmental I-O (Leontief)
- CGE model development
- Goodwin-Keen integration
- Kaleckian extensions to Goodwin
- Multi-region I-O

**Run Time:** ~25 seconds
**Output:** 2 visualizations per exercise

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scipy
```

All exercises are self-contained and can run independently.

### Running Exercises

Each file can be run standalone:

```bash
# Time series analysis
python phase3-exercises/time_series/01_trend_extraction_exercises.py

# Distributional analysis
python phase3-exercises/distributional/02_inequality_analysis_exercises.py

# Sectoral analysis
python phase3-exercises/sectoral/03_sectoral_analysis_exercises.py

# Financial analysis
python phase3-exercises/financial/04_financial_cycles_exercises.py

# SFC accounting
python phase3-exercises/sfc_accounting/05_sfc_accounting_exercises.py

# Input-Output & Goodwin
python phase3-exercises/input_output/06_input_output_exercises.py
```

### Output

Each exercise generates:
- Detailed console output with interpretations
- High-quality PNG visualizations (300 DPI)
- Statistical summaries
- Economic interpretation sections
- Extension challenge descriptions

---

## Pedagogical Structure

Each exercise follows this template:

1. **Problem Description**
   - Real economic question
   - Heterodox theoretical framing
   - Relevant economist/school reference

2. **Complete Implementation**
   - From-scratch algorithms (no black boxes)
   - Commented code explaining logic
   - NumPy/Pandas best practices

3. **Visualization**
   - Publication-quality plots
   - Multiple perspectives on same data
   - Clear labeling and legends

4. **Economic Interpretation**
   - Heterodox perspective on results
   - Policy implications
   - Critique of mainstream alternatives
   - Real-world examples

5. **Extension Challenges**
   - 4-6 advanced challenges per exercise
   - Research-grade complexity
   - Academic references provided

---

## Theoretical Frameworks Covered

### Post-Keynesian Economics
- Effective demand primacy
- Endogenous money
- Fundamental uncertainty
- Historical time and path dependence
- Stock-flow consistency

### Kaleckian Economics
- Profit equation and class conflict
- Degree of monopoly
- Political business cycle
- Wage-led vs profit-led growth

### Minskyan Finance Theory
- Financial Instability Hypothesis
- Hedge-Speculative-Ponzi taxonomy
- Money manager capitalism
- Asset price dynamics

### Sraffian/Neo-Ricardian
- Production of commodities framework
- Critique of marginal productivity
- Price and distribution

### Godley SFC Methodology
- Sectoral balances
- Stock-flow consistency
- Balance sheet dynamics
- Financial fragility indicators

### Structuralist Development
- Input-output interdependence
- Structural heterogeneity
- Import-substitution vs export-led
- Prebisch-Singer hypothesis

---

## Applications to Real Data

Each exercise can be extended with real data:

### Data Sources

**National Accounts:**
- FRED (Federal Reserve Economic Data)
- OECD Statistics
- BEA (Bureau of Economic Analysis)
- National statistical agencies

**Distribution:**
- WID (World Inequality Database)
- LIS (Luxembourg Income Study)
- OECD Income Distribution Database

**Financial:**
- BIS (Bank for International Settlements)
- Flow of Funds (Federal Reserve)
- IMF Financial Soundness Indicators

**Input-Output:**
- BEA I-O Tables (US)
- WIOD (World Input-Output Database)
- Eora MRIO

**Sectoral Balances:**
- NIPA Tables
- Flow of Funds Accounts
- National balance of payments

---

## Learning Pathway

### Beginner → Intermediate
1. Start with **Time Series (Exercise 1)** - HP filter
2. Move to **Distributional (Exercise 1)** - Lorenz/Gini
3. Try **Sectoral (Exercise 1)** - Shift-share

### Intermediate → Advanced
1. **Financial (Exercise 1)** - Minsky simulation
2. **SFC (Exercise 1)** - Sectoral balances
3. **Time Series (Exercise 2)** - Structural breaks

### Advanced → Research
1. **Financial (Exercise 2)** - Credit cycles
2. **SFC (Exercise 2)** - Full matrices
3. **Input-Output** - Both exercises
4. Attempt extension challenges

---

## Extension Challenges Summary

Total challenges across all exercises: **30+**

**Difficulty Levels:**
- ⭐ Intermediate: Extend existing code
- ⭐⭐ Advanced: New implementation required
- ⭐⭐⭐ Research: Novel application/methodology

**Example Research-Grade Challenges:**
- Implement Bai-Perron test (⭐⭐⭐)
- Build Kalman filter for time-varying NAIRU (⭐⭐⭐)
- Construct Distributional National Accounts (⭐⭐⭐)
- Full dynamic SFC model with portfolio choice (⭐⭐⭐)
- Agent-based financial crisis model (⭐⭐⭐)
- Climate-integrated SFC model (⭐⭐⭐)

---

## Academic Integration

### For Teaching
- Each exercise is self-contained lecture material
- Combines theory, implementation, and interpretation
- Progressive difficulty within each module
- Real-world examples throughout

### For Research
- Reusable code for analysis
- Extension challenges → dissertation chapters
- Heterodox methods often missing from econometrics texts
- Publication-quality visualizations

### For Self-Study
- No prerequisites beyond intermediate Python
- All algorithms explained from first principles
- Economic intuition emphasized
- References to key papers/books

---

## References

### Key Textbooks
- Godley & Lavoie (2007) - *Monetary Economics*
- Lavoie (2014) - *Post-Keynesian Economics*
- Miller & Blair (2009) - *Input-Output Analysis*
- Piketty (2014) - *Capital in the Twenty-First Century*
- Keen (2011) - *Debunking Economics*

### Methodological Papers
- Hamilton (2018) - HP filter critique
- Borio (2014) - Financial cycles
- Palma (2011) - Palma ratio
- Godley (1999) - Seven unsustainable processes
- Goodwin (1967) - Growth cycle

### Data & Software
- Godley & Lavoie SFC models (EViews code available)
- Keen's Minsky software
- WIOD for I-O tables

---

## Future Additions

Potential Phase 3B exercises:
- Network analysis of production networks
- Agent-based modeling (Dosi, Fagiolo, et al.)
- Modern Monetary Theory (MMT) applications
- Ecological economics (material flows)
- Comparative institutional analysis
- Gender and care economy
- Varieties of Capitalism (VoC)

---

## Contributing

Found an error? Have an improvement?
- Issues welcome
- Pull requests for extensions encouraged
- Additional exercises appreciated

---

## License

Educational use freely permitted.
If adapted for publication, please cite.

---

## Acknowledgments

Theoretical frameworks:
- Wynne Godley (SFC)
- Hyman Minsky (Financial instability)
- Michal Kalecki (Class conflict, effective demand)
- Richard Goodwin (Growth cycles)
- Wassily Leontief (Input-Output)
- José Gabriel Palma (Inequality)

Contemporary researchers continuing these traditions:
- Marc Lavoie, Steve Keen, Randall Wray, James Galbraith,
  Engelbert Stockhammer, Özlem Onaran, Thomas Palley,
  Claudio Borio, Branko Milanovic, and many others.

---

**End of README**

For questions or discussions, consult the references or join heterodox economics communities:
- Post-Keynesian Economics Society (PKES)
- Association for Heterodox Economics (AHE)
- International Confederation of Associations for Pluralism in Economics (ICAPE)
