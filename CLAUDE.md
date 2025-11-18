# Claude Token Burn Session: Python Economics & Modelling
**Time Available:** ~2 hours  
**Budget:** $250 Claude API tokens  
**Date:** [Today's date]

## Background & Context

### Technical Skills
- **Primary:** PHP/Laravel web development (~1 year professional)
- **Python:** Intermediate - numpy, matplotlib, Monte Carlo methods
- **Programming fundamentals:** Solid (loops, functions, OOP concepts)
- **New territory:** PyQt, advanced pandas, economic simulation architectures

### Domain Knowledge
- Economics degree
- Strong interest in heterodox economics:
  - Post-Keynesian (PK)
  - Marxian economics
  - Comparative Political Economy (CPE)
- Economic history & history of economic thought
- Familiar with mainstream models but prefer pluralist/critical approaches

### Goals
1. **Primary:** Build Python competency in data science/visualisation for economics
2. **Secondary:** Learn PyQt for building economic simulation tools
3. **Aspirational:** Create academic-grade tools for heterodox economic analysis
4. **Long-term:** Potential route back into academia

---

## Token Burn Strategy

### Phase 1: Synthetic Dataset Generation (30-40 mins)
Generate large, realistic datasets covering:

**Core Economics:**
- National accounts data (GDP, consumption, investment, exports)
- Labour market indicators (employment, wages, productivity)
- Financial sector (credit, asset prices, debt ratios)
- Price indices (CPI, PPI, unit labour costs)

**Heterodox Focus Areas:**
- Income/wealth distribution (Gini coefficients, quintile shares)
- Sectoral balances (Godley-style SFC data)
- Class shares of income (wage share, profit share)
- International trade (trade balances, capital flows)
- Financialisation indicators (debt-to-GDP, financial sector size)

**Demographics & Energy:**
- Household-level consumption and income
- Energy consumption by sector and source
- Regional economic disparities

**Prompt Template:**
"Generate comprehensive synthetic dataset with [X] rows covering [domain]. Include realistic correlations reflecting [economic theory]. Ensure data suitable for [specific analysis]. Output as CSV with data dictionary explaining methodology and variable relationships."

---

### Phase 2: PyQt Application Development (40-50 mins)

**Learning Progression:**
1. **Basic widgets** - Simple calculators (multiplier, growth models)
2. **Interactive plots** - Real-time visualisation with matplotlib
3. **Economic models** - IS-LM, AD-AS, Kaleckian growth models
4. **Advanced simulations** - Stock-Flow Consistent models, agent-based

**Heterodox Economics Applications:**
- Kalecki profit equation simulator
- Goodwin growth cycle (class struggle model)
- Minsky's Financial Instability Hypothesis visualisation
- Godley-Lavoie SFC model interface
- Input-output table analysis (Leontief/Sraffa)

**Prompt Template:**
"Create complete PyQt6 application for [economic model]. Include: parameter controls, real-time plotting, theoretical explanation in UI, data export. Full code with comments explaining PyQt architecture and economic logic. Target audience: economics students/researchers."

---

### Phase 3: Analysis Exercise Library (20-30 mins)

**Focus Areas:**
- Time series analysis (trend extraction, structural breaks)
- Distributional analysis (Lorenz curves, Palma ratio)
- Sectoral analysis (shift-share, decomposition)
- Financial analysis (Minsky moments, credit cycles)
- Comparative analysis (cross-country, institutional)

**Heterodox Techniques:**
- Stock-Flow Consistent accounting
- Input-output analysis
- Goodwin-style cycle decomposition
- Regime-switching models
- Institutional comparison matrices

**Prompt Template:**
"Generate [N] Python exercises for [analysis type] in heterodox economics. Each includes: problem description referencing [economist/school], full pandas/numpy solution, visualization, economic interpretation, and extension challenges. Progress from basic to research-grade analysis."

---

### Phase 4: Capstone Application (20-30 mins if time permits)

**Target: Academic-Grade Research Tool**

Options:
1. **Heterodox Macro Dashboard**
   - Load real data (FRED, OECD, national sources)
   - Multiple theoretical frameworks (PK, Marxian, Institutionalist)
   - Comparative visualisation
   - Report generation

2. **SFC Model Builder**
   - Interactive balance sheet/flow matrix construction
   - Parameter estimation from data
   - Scenario analysis
   - Export for publication

3. **Distributional Economics Toolkit**
   - Load micro/macro distribution data
   - Calculate multiple inequality measures
   - Functional vs personal distribution
   - Historical comparison tools

**Prompt Template:**
"Build production-ready PyQt6 application for [purpose]. Architecture: MVC pattern, modular design. Features: [list]. Include: complete codebase, requirements.txt, documentation, example workflows for academic research. Code should be publication-quality with proper citations for economic methods."

---

## Quick-Fire Prompts (if time remaining)

### More Datasets
- "Extend [previous dataset] to 100,000 rows with additional variables: [list]"
- "Generate crisis scenario data: normal periods + 3 financial crises with realistic contagion"
- "Create panel data: 30 countries, 50 years, quarterly frequency, [variables]"

### More Code Examples
- "20 matplotlib visualization examples for economic data: distribution plots, time series, correlation matrices, flow diagrams"
- "15 pandas operations for heterodox economic analysis with worked examples"
- "10 complete PyQt economic calculator widgets with different UI patterns"

### Theory + Code
- "Explain [economic model] with full Python implementation: equations, simulation, visualization, sensitivity analysis"
- "Compare mainstream vs heterodox approaches to [topic] with side-by-side Python implementations"

---

## Post-Session Actions

1. **Save everything** - Don't read it all now, organize later
2. **Create project structure:**
```
   economics-python/
   ├── datasets/
   ├── exercises/
   ├── pyqt-apps/
   ├── notebooks/
   └── docs/
```
3. **Pick ONE dataset + ONE PyQt app** - Actually work through them this week
4. **Schedule:** 2-3 hours/week minimum to use the materials
5. **Academic path:** Once competent, identify which tool could become a research paper

---

## Notes

- **Don't get perfectionist** - Generate volume now, refine later
- **Prioritise heterodox content** - It's underrepresented in standard tutorials
- **Think modular** - Code should be reusable across projects
- **Document as you go** - Your future self will thank you
- **Academic focus:** Tools should handle real research workflows, not just toy examples

---

## Potential Academic Outputs

- Software paper on toolkit for heterodox macro analysis
- Replication study using generated tools
- Teaching materials for pluralist economics courses
- Collaboration with heterodox research centres (PKE, INET, etc.)
