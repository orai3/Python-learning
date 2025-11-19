# Example Academic Research Workflows

This document provides detailed, step-by-step workflows for using the Heterodox Macro Dashboard in academic research. Each workflow addresses a specific research question using appropriate theoretical frameworks and methodologies.

---

## Table of Contents

1. [Workflow 1: Wage-Led vs Profit-Led Growth Regime](#workflow-1-wage-led-vs-profit-led-growth-regime)
2. [Workflow 2: Financial Fragility Assessment (Minsky)](#workflow-2-financial-fragility-assessment-minsky)
3. [Workflow 3: Rate of Profit Trends and Crisis Tendencies](#workflow-3-rate-of-profit-trends-and-crisis-tendencies)
4. [Workflow 4: Sectoral Balance Analysis (Godley)](#workflow-4-sectoral-balance-analysis-godley)
5. [Workflow 5: Inequality and Exploitation Analysis](#workflow-5-inequality-and-exploitation-analysis)
6. [Workflow 6: Financialization and Institutional Change](#workflow-6-financialization-and-institutional-change)
7. [Workflow 7: Comparative Political Economy Analysis](#workflow-7-comparative-political-economy-analysis)
8. [Workflow 8: Crisis Anatomy - Multi-Framework Analysis](#workflow-8-crisis-anatomy---multi-framework-analysis)

---

## Workflow 1: Wage-Led vs Profit-Led Growth Regime

**Research Question:** Is the economy characterized by wage-led or profit-led growth?

**Theoretical Framework:** Post-Keynesian (Kaleckian growth theory)

**Academic Context:**
This analysis is crucial for understanding the relationship between distribution and growth, with significant policy implications. Wage-led regimes benefit from redistribution toward labor, while profit-led regimes may require different policy approaches.

### Step-by-Step Process

#### 1. Load Required Data
```
File > Load Default Datasets
→ Ensure 'macro' dataset is loaded
```

#### 2. Examine Distribution Trends
```
Data Management Tab:
→ Select 'macro' dataset
→ Review wage_share and profit_share variables
→ Note historical trends and current values
```

#### 3. Run Post-Keynesian Analysis
```
Framework Analysis Tab:
→ Select "Post Keynesian" framework
→ Click "Run Analysis"
→ Navigate to "Analysis Results" sub-tab
```

**Key Indicators to Examine:**
- Current wage share vs historical average
- Wage share trend (rising/falling)
- Capacity utilization rate
- Investment rate

#### 4. Generate Wage-Profit Visualization
```
Framework Analysis Tab > Visualizations:
→ Select "Wage vs Profit Share"
→ Click "Generate Chart"
→ Examine trends over time
```

**What to Look For:**
- Long-term trends in distribution
- Cyclical patterns
- Structural breaks
- Relationship to business cycles

#### 5. Analyze Demand Components
```
Visualizations Tab:
→ Select "Time Series Analysis"
→ Generate for variables: ['consumption', 'investment', 'gdp']
→ Examine which component drives growth
```

#### 6. Interpret Results

**Wage-Led Regime Indicators:**
- Rising wage share → higher consumption growth
- Consumption is primary growth driver
- Low capacity utilization responds to demand stimulus
- Investment follows consumption (accelerator effect)

**Profit-Led Regime Indicators:**
- Rising profit share → higher investment
- Investment is primary growth driver
- High capacity utilization
- Export competitiveness important (cost competitiveness)

#### 7. Check Capacity Utilization
```
Analysis Results:
→ Find capacity_utilization indicator
→ If <80%: Demand-constrained (supports wage-led)
→ If >85%: Possibly supply-constrained
```

#### 8. Generate Report
```
Report Generation Tab:
→ Select "Post Keynesian" framework
→ Include "Theoretical Background"
→ Include "Data Summary Statistics"
→ Generate and export report
```

### Expected Outputs

1. **Quantitative Assessment:**
   - Wage share: X%
   - Historical average: Y%
   - Trend: Rising/Falling/Stable
   - Capacity utilization: Z%

2. **Regime Classification:**
   - Wage-led or Profit-led
   - Strength of relationship
   - Policy implications

3. **Academic Output:**
   - Empirical results for paper
   - Charts for publication
   - Theoretical interpretation

### Policy Implications

**If Wage-Led:**
- Progressive redistribution supports growth
- Minimum wage increases can be expansionary
- Labor protections benefit aggregate demand
- Austerity is counterproductive

**If Profit-Led:**
- May need alternative growth model
- Export-led growth strategy
- Investment incentives
- Structural transformation needed

### Key References

- Lavoie, M., & Stockhammer, E. (2013). *Wage-led Growth: An Equitable Strategy for Economic Recovery*. Palgrave Macmillan.
- Stockhammer, E. (2011). Wage norms, capital accumulation, and unemployment. *Oxford Review of Economic Policy*, 27(2), 295-311.
- Bhaduri, A., & Marglin, S. (1990). Unemployment and the real wage. *Cambridge Journal of Economics*, 14(4), 375-393.

---

## Workflow 2: Financial Fragility Assessment (Minsky)

**Research Question:** Is financial fragility increasing according to Minsky's hypothesis?

**Theoretical Framework:** Post-Keynesian (Minsky's Financial Instability Hypothesis)

**Academic Context:**
Minsky argued that stability breeds instability through increasing leverage during good times, leading to fragile financial structures prone to crisis.

### Step-by-Step Process

#### 1. Load Financial Data
```
File > Load Default Datasets
→ Ensure 'macro' and 'crisis' datasets loaded
```

#### 2. Run Post-Keynesian Analysis
```
Framework Analysis Tab:
→ Select "Post Keynesian"
→ Run Analysis
→ Review "Financial Stability" section
```

**Key Minsky Indicators:**
- Debt-to-GDP ratio
- Debt service to income ratio
- Credit growth rate
- Asset price trends
- Leverage ratios

#### 3. Examine Sectoral Debt Accumulation
```
Visualizations:
→ Generate "Sectoral Balances (Godley)"
→ Identify which sectors are accumulating debt
```

**Minsky's Sectoral Analysis:**
- Which sectors are borrowing?
- Who is lending?
- Is debt sustainable?

#### 4. Classify Financing Regimes

**Hedge Finance:**
- Can pay interest and principal from income
- Conservative, stable
- Low debt ratios

**Speculative Finance:**
- Can pay interest but must roll over principal
- Vulnerable to interest rate changes
- Moderate debt ratios

**Ponzi Finance:**
- Cannot pay interest from income
- Relies on asset price appreciation
- High debt ratios, very fragile

#### 5. Generate Time Series
```
Create charts for:
→ Debt-to-GDP ratio over time
→ Credit growth
→ Asset prices (if available)
→ Debt service ratios
```

#### 6. Identify Minsky Moments
```
Crisis Dataset:
→ Look for sharp increases in:
   - Default rates
   - Credit spreads
   - Asset price volatility
→ Identify crisis onset points
```

#### 7. Pre-Crisis Pattern Analysis

**Warning Signs (Minsky):**
1. Prolonged period of stability
2. Rising leverage
3. Asset price booms
4. Increasing speculation
5. Financial innovation (complexity)
6. Declining lending standards

#### 8. Generate Comprehensive Report
```
Report Generation:
→ Focus on financial stability section
→ Document pre-crisis trends
→ Identify current fragility level
```

### Expected Outputs

1. **Fragility Assessment:**
   - Current debt-to-GDP: X%
   - Trend: Rising/Stable/Falling
   - Dominant financing regime
   - Fragility index

2. **Crisis Vulnerability:**
   - Distance from Minsky moment
   - Key vulnerabilities
   - Potential triggers

3. **Policy Recommendations:**
   - Financial regulation needs
   - Countercyclical measures
   - Debt restructuring considerations

### Policy Implications

- Need for financial regulation
- Countercyclical capital requirements
- Asset price monitoring
- Credit growth limits
- Public sector as stabilizer

### Key References

- Minsky, H. (1986). *Stabilizing an Unstable Economy*. Yale University Press.
- Minsky, H. (1992). The financial instability hypothesis. *The Jerome Levy Economics Institute Working Paper*, No. 74.
- Keen, S. (2011). *Debunking Economics*. Zed Books. (Chapter on Minsky)

---

## Workflow 3: Rate of Profit Trends and Crisis Tendencies

**Research Question:** Is the rate of profit falling, consistent with Marx's TRPF?

**Theoretical Framework:** Marxian Political Economy

**Academic Context:**
Marx's tendency of the rate of profit to fall (TRPF) is central to his crisis theory. While controversial, empirical analysis can illuminate long-term profitability trends.

### Step-by-Step Process

#### 1. Load Economic Data
```
Load 'macro' dataset with:
→ Profits (surplus value)
→ Capital stock (constant capital)
→ Wages (variable capital)
```

#### 2. Run Marxian Analysis
```
Framework Analysis Tab:
→ Select "Marxian"
→ Run Analysis
→ Focus on rate of profit indicators
```

**Key Marxian Indicators:**
- Rate of profit: r = s/(c+v)
- Rate of surplus value: e = s/v
- Organic composition of capital: k = c/v
- Share of wages in GDP

#### 3. Generate Rate of Profit Chart
```
Visualizations:
→ Select "Rate of Profit (Marxian)"
→ Generate chart with trend line
→ Note trend coefficient (slope)
```

**Interpretation:**
- Negative slope: Confirms TRPF
- Positive slope: Counter-tendencies dominant
- Flat: Balanced forces

#### 4. Analyze Components

**Decompose Rate of Profit:**
```
r = (s/v) / (k + 1)

Where:
s/v = rate of surplus value (exploitation)
k = organic composition of capital
```

**Check:**
- Is r falling because k rising? (labor-saving tech)
- Is r falling because s/v falling? (class struggle)
- Or both?

#### 5. Examine Counter-Tendencies

**Marx identified counter-tendencies:**
1. Increasing exploitation (rising s/v)
2. Cheapening of constant capital
3. Foreign trade
4. Expanding relative surplus population
5. Intensification of labor

**In the data, look for:**
- Productivity growth exceeding wage growth
- Unemployment trends (reserve army)
- Import price trends
- Labor intensity measures

#### 6. Historical Trend Analysis
```
Generate long-term time series:
→ Rate of profit over decades
→ Identify periods of rising/falling
→ Relate to crises and recoveries
```

#### 7. Cross-Check with Accumulation
```
Examine relationship:
→ Rate of profit vs investment rate
→ Do falling profits precede crises?
→ Do rising profits precede booms?
```

#### 8. Generate Marxian Report
```
Report Generation:
→ Select "Marxian" framework
→ Include theoretical background
→ Document empirical findings
```

### Expected Outputs

1. **Profit Rate Trends:**
   - Current rate: X%
   - Long-term trend: Falling/Rising/Stable
   - Average rate: Y%
   - Volatility: Z

2. **Crisis Tendencies:**
   - Evidence for/against TRPF
   - Dominant counter-tendencies
   - Implications for accumulation

3. **Academic Contribution:**
   - Empirical evidence on TRPF debate
   - Country-specific patterns
   - Periodization of capitalism

### Theoretical Interpretation

**If Falling:**
- Supports classical Marxian crisis theory
- Labor-saving technical change dominant
- Implies periodic crises inevitable
- Need for counter-tendencies analysis

**If Rising:**
- Counter-tendencies dominant
- May indicate:
  - Increasing exploitation
  - Cheapening of capital goods
  - Neoliberal restructuring successful (from capital's view)

**If Stable:**
- Balanced forces
- May indicate:
  - Successful crisis management
  - Institutional evolution
  - Structural transformation

### Policy Implications

**From Marxian Perspective:**
- Falling r → crisis tendencies → need for systemic change
- Rising exploitation → strengthen labor
- High unemployment → organize reserve army
- Financialization → regulate fictitious capital

### Key References

- Marx, K. (1894/1991). *Capital, Volume III*. Penguin Classics. (Part III)
- Shaikh, A. (2016). *Capitalism: Competition, Conflict, Crises*. Oxford University Press.
- Dumenil, G., & Levy, D. (2002). The profit rate: Where and how much did it fall? *Review of Radical Political Economics*, 34(4), 437-461.
- Kliman, A. (2011). *The Failure of Capitalist Production*. Pluto Press.

---

## Workflow 4: Sectoral Balance Analysis (Godley)

**Research Question:** Are sectoral financial balances sustainable?

**Theoretical Framework:** Post-Keynesian (Stock-Flow Consistent modeling)

**Academic Context:**
Wynne Godley's sectoral balance approach provides an accounting framework that must hold as identity. It's crucial for understanding macroeconomic sustainability.

### Step-by-Step Process

#### 1. Load SFC Data
```
Load 'sfc' dataset with sectoral balance data
```

#### 2. Verify Accounting Identity
```
Run Post-Keynesian Analysis:
→ Check "Sectoral Balances" section
→ Verify: Private + Government + Foreign ≈ 0
```

**Important:** This must hold as accounting identity. Deviation indicates data error.

#### 3. Generate Sectoral Balances Chart
```
Visualizations:
→ Select "Sectoral Balances (Godley)"
→ Generate chart
```

**Visual Analysis:**
- Which sectors are in surplus/deficit?
- Are positions sustainable?
- How do balances co-move?

#### 4. Identify Unsustainable Configurations

**Warning Signs:**
- Persistent private sector deficit (rare, unsustainable)
- Large foreign deficit with government deficit (twin deficits)
- Rapid changes in sector positions

#### 5. Historical Pattern Analysis
```
Examine over time:
→ Pre-crisis: What were balance positions?
→ During crisis: How did they shift?
→ Post-crisis: New equilibrium?
```

#### 6. Policy Scenario Analysis

**Functional Finance (Lerner):**
- Government balance should offset private sector desired savings
- Foreign balance determined by trade
- Identity: G = S_private + S_foreign

**Example Scenarios:**

**Scenario 1: Austerity**
```
If government balance → surplus
Then: Private + Foreign balances must → deficit
Question: Is this sustainable?
```

**Scenario 2: Export-Led Growth**
```
If foreign balance → surplus
Then: Government + Private can both → surplus
Possible: But requires competitiveness
```

**Scenario 3: Private Sector Deleveraging**
```
If private balance → surplus (saving)
Then: Government + Foreign must → deficit
Implication: Need government deficit or exports
```

#### 7. Link to Demand and Growth
```
Examine relationship between:
→ Sectoral balances and GDP growth
→ Private surplus and recession
→ Government deficit and recovery
```

#### 8. Generate SFC Report
```
Report:
→ Document current positions
→ Assess sustainability
→ Provide policy analysis
```

### Expected Outputs

1. **Current Balances:**
   - Private sector: +X% of GDP
   - Government: -Y% of GDP
   - Foreign: +Z% of GDP
   - Identity check: Sum ≈ 0

2. **Sustainability Assessment:**
   - Which positions are unsustainable?
   - What adjustments needed?
   - Crisis vulnerabilities

3. **Policy Analysis:**
   - Fiscal space available?
   - External constraint binding?
   - Sectoral debt dynamics

### Policy Implications

**Functional Finance:**
- Government should accommodate private saving desires
- Fiscal deficits often necessary and sustainable
- Focus on real resource use, not accounting

**External Constraint:**
- Foreign deficit limits may bind
- Need for export competitiveness or capital controls
- Euro zone: No currency sovereignty

**Private Sector:**
- Deleveraging requires public sector deficit
- Balance sheet recession dynamics
- Need for debt relief or jubilee

### Key References

- Godley, W., & Lavoie, M. (2007). *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth*. Palgrave Macmillan.
- Godley, W. (1999). Seven unsustainable processes. *The Jerome Levy Economics Institute Special Report*.
- Wray, L. R. (2012). *Modern Money Theory*. Palgrave Macmillan.

---

## Workflow 5: Inequality and Exploitation Analysis

**Research Question:** How has income inequality evolved, and what are the mechanisms?

**Theoretical Frameworks:** Marxian + Institutionalist

**Academic Context:**
Rising inequality is a central concern in heterodox economics, with different frameworks emphasizing exploitation (Marxian) vs power relations (Institutionalist).

### Step-by-Step Process

#### 1. Load Inequality Data
```
Load datasets:
→ 'inequality' (distributional data)
→ 'macro' (functional distribution)
```

#### 2. Marxian Analysis: Functional Distribution
```
Run Marxian Analysis:
→ Examine wage share trends
→ Calculate rate of surplus value
→ Check productivity vs wage growth gap
```

**Key Marxian Indicators:**
- Wage share: Labor's share of income
- Profit share: Capital's share
- Rate of exploitation: e = profit/wages
- Productivity-wage gap

#### 3. Institutionalist Analysis: Personal Distribution
```
Run Institutionalist Analysis:
→ Examine Gini coefficient
→ Check Palma ratio
→ Review top 10% income share
```

#### 4. Generate Inequality Visualizations
```
Create:
→ Lorenz Curve
→ Wage vs Profit Share time series
→ Income distribution histogram
```

#### 5. Calculate Multiple Inequality Measures
```
Data Management Tab:
→ Use 'inequality' dataset
→ Calculate:
   - Gini coefficient
   - Palma ratio (top 10% / bottom 40%)
   - Theil index
   - Top 1% share
```

#### 6. Analyze Mechanisms

**Marxian Mechanisms:**
1. **Exploitation:**
   - Compare productivity growth to wage growth
   - Rising gap = increasing exploitation

2. **Reserve Army:**
   - Unemployment disciplines wages
   - Check unemployment rate trends

3. **Deskilling:**
   - Technology reduces worker bargaining power
   - Check skill premium trends

**Institutionalist Mechanisms:**
1. **Power Relations:**
   - Union density trends
   - Minimum wage real value
   - Labor market regulations

2. **Financialization:**
   - Financial sector share of profits
   - CEO compensation ratios
   - Shareholder value orientation

3. **Globalization:**
   - Trade exposure
   - Capital mobility
   - Race to the bottom

#### 7. Link to Macroeconomic Outcomes
```
Examine correlations:
→ Inequality vs growth
→ Inequality vs financial fragility
→ Inequality vs demand regime
```

#### 8. Comparative Framework Report
```
Generate report including:
→ Marxian interpretation (exploitation)
→ Institutionalist interpretation (power)
→ Comparative insights
```

### Expected Outputs

1. **Distributional Trends:**
   - Functional: Wage share fell from X% to Y%
   - Personal: Gini rose from A to B
   - Mechanisms: Productivity-wage gap = Z%

2. **Theoretical Interpretation:**
   - Marxian: Increasing exploitation, weakened labor
   - Institutionalist: Shifting power, institutional change

3. **Policy Implications:**
   - Strengthen labor organization
   - Progressive taxation
   - Minimum wage adjustment
   - Financial regulation

### Key References

- Atkinson, A. B. (2015). *Inequality: What Can Be Done?* Harvard University Press.
- Piketty, T. (2014). *Capital in the Twenty-First Century*. Harvard University Press.
- Palma, J. G. (2011). Homogeneous middles vs. heterogeneous tails. *Development and Change*, 42(1), 87-153.

---

## Workflow 6: Financialization and Institutional Change

**Research Question:** How has financialization transformed the economy?

**Theoretical Framework:** Institutionalist (with Post-Keynesian insights)

### Step-by-Step Process

#### 1. Load Macro Data
```
Ensure access to:
→ Financial sector GDP share
→ Financial sector profits
→ Household debt ratios
→ Corporate debt and equity
```

#### 2. Run Institutionalist Analysis
```
Framework Analysis:
→ Select "Institutionalist"
→ Run analysis
→ Focus on financialization indicators
```

#### 3. Calculate Financialization Index
```
Components:
→ Financial sector share of GDP
→ Financial sector share of profits
→ Household debt to income ratio
→ Non-financial corporate financialization
```

#### 4. Identify Manifestations

**Veblen's Perspective:** Absentee ownership, predation

**Modern Forms:**
1. **Household:**
   - Rising debt
   - Asset price dependence
   - Pension financialization

2. **Corporate:**
   - Shareholder value maximization
   - Short-termism
   - Share buybacks over investment

3. **Government:**
   - Financial sector power
   - Austerity ideology
   - Public-private partnerships

#### 5. Analyze Consequences

**Economic Effects:**
- Lower investment in real economy
- Increased inequality
- Greater instability
- Slower growth

**Social Effects:**
- Commodification
- Individualization of risk
- Erosion of social provision

#### 6. Historical Periodization
```
Identify phases:
→ Pre-financialization (1945-1980)
→ Transition (1980-2000)
→ Mature financialization (2000+)
→ Post-crisis (2008+)
```

#### 7. Comparative Analysis
```
If panel data available:
→ Compare countries
→ Identify institutional configurations
→ Link to outcomes
```

#### 8. Generate Report
```
Document:
→ Financialization trends
→ Institutional changes
→ Economic consequences
→ Alternative institutions
```

### Expected Outputs

1. **Financialization Metrics:**
   - Financial sector GDP share: Rose from X% to Y%
   - Household debt ratio: Increased Z%
   - Shareholder payouts: Now W% of profits

2. **Institutional Analysis:**
   - Key institutional changes
   - Power shifts
   - Path dependencies

3. **Policy Alternatives:**
   - Re-regulation
   - Stakeholder governance
   - Public banking
   - Financial transaction taxes

### Key References

- Epstein, G. A. (Ed.). (2005). *Financialization and the World Economy*. Edward Elgar.
- Lazonick, W., & O'Sullivan, M. (2000). Maximizing shareholder value. *Economy and Society*, 29(1), 13-35.
- Krippner, G. R. (2011). *Capitalizing on Crisis*. Harvard University Press.

---

## Workflow 7: Comparative Political Economy Analysis

**Research Question:** How do different capitalist varieties perform?

**Theoretical Framework:** Institutionalist (Varieties of Capitalism)

### Step-by-Step Process

#### 1. Load Panel Data
```
Load 'panel' dataset with cross-country data
```

#### 2. Identify Institutional Configurations

**Hall & Soskice Classification:**
- Liberal Market Economies (LME): US, UK
- Coordinated Market Economies (CME): Germany, Sweden
- Mixed: France, Italy

**Alternative Classifications:**
- Welfare regimes (Esping-Andersen)
- Growth models (export vs consumption-led)
- Labor relations (coordinated vs fragmented)

#### 3. Run Institutionalist Analysis
```
Examine:
→ Government size
→ Union density
→ Welfare provisions
→ Financial systems
```

#### 4. Compare Outcomes Across Types

**Economic Outcomes:**
- Growth rates
- Unemployment
- Inequality
- Innovation

**Social Outcomes:**
- Poverty rates
- Social mobility
- Health indicators
- Environmental sustainability

#### 5. Analyze Institutional Complementarities

**Example: CME Complementarities**
- Patient capital + long-term employment
- Vocational training + industry coordination
- Works councils + stakeholder governance
- Incremental innovation + quality production

#### 6. Test for Convergence vs Diversity
```
Over time:
→ Are systems converging (neoliberalism)?
→ Or maintaining distinct configurations?
→ Hybridization patterns?
```

#### 7. Crisis Response Comparison
```
Using crisis dataset:
→ How did different types respond?
→ Which institutions proved resilient?
→ Path dependence vs transformation
```

#### 8. Generate Comparative Report
```
Document:
→ Institutional types
→ Performance comparison
→ Complementarities
→ Policy lessons
```

### Expected Outputs

1. **Typology:**
   - Classification of countries
   - Key institutional differences
   - Complementarities

2. **Performance Comparison:**
   - Growth: CME vs LME
   - Inequality: Lower in CME
   - Unemployment: Variable
   - Innovation: Different types

3. **Policy Lessons:**
   - No one-size-fits-all
   - Coherence matters
   - Path dependence real
   - Systemic thinking needed

### Key References

- Hall, P. A., & Soskice, D. (2001). *Varieties of Capitalism*. Oxford University Press.
- Esping-Andersen, G. (1990). *The Three Worlds of Welfare Capitalism*. Polity Press.
- Amable, B. (2003). *The Diversity of Modern Capitalism*. Oxford University Press.

---

## Workflow 8: Crisis Anatomy - Multi-Framework Analysis

**Research Question:** What caused the crisis, and what are policy implications?

**Theoretical Frameworks:** All three (PK, Marxian, Institutionalist)

### Step-by-Step Process

#### 1. Load Crisis Data
```
Datasets:
→ 'crisis' (crisis indicators)
→ 'macro' (macro variables)
→ 'sfc' (sectoral balances)
```

#### 2. Identify Crisis Period
```
Determine:
→ Pre-crisis: Expansion phase
→ Crisis onset: When/how
→ Crisis: Depth and duration
→ Recovery: Speed and character
```

#### 3. Post-Keynesian Analysis

**Minsky's FIH:**
```
Examine pre-crisis:
→ Rising debt ratios
→ Declining lending standards
→ Asset price booms
→ Increasing financial sector
→ Shift from hedge to Ponzi finance
```

**Godley's Balances:**
```
Check:
→ Unsustainable private sector deficit
→ Foreign sector imbalances
→ Government position
```

#### 4. Marxian Analysis

**Overaccumulation Crisis:**
```
Examine:
→ Rate of profit trend pre-crisis
→ Capacity utilization
→ Rising organic composition
→ Squeeze on profits
```

**Underconsumption:**
```
Check:
→ Wage share falling
→ Consumption debt-financed
→ Unsustainable household debt
```

#### 5. Institutionalist Analysis

**Institutional Changes:**
```
Identify:
→ Financial deregulation
→ Weakened labor
→ Ideology shifts
→ Power realignments
```

**Path to Crisis:**
```
Cumulative causation:
→ Deregulation → risk-taking
→ Inequality → debt
→ Financialization → instability
```

#### 6. Generate Multi-Framework Visualizations
```
Create for each framework:
→ Key indicators over crisis period
→ Before-during-after comparisons
→ Theoretical predictions vs actual
```

#### 7. Compare Framework Explanations

**Post-Keynesian:**
- Financial instability (Minsky)
- Unsustainable debt dynamics
- Emphasis: Financial sector

**Marxian:**
- Overaccumulation or underconsumption
- Profit squeeze
- Emphasis: Real economy contradictions

**Institutionalist:**
- Institutional change (deregulation)
- Power shifts
- Emphasis: Political economy

#### 8. Synthesize and Generate Report
```
Comprehensive report:
→ Each framework's explanation
→ Complementary insights
→ Contested interpretations
→ Policy implications from each view
```

### Expected Outputs

1. **Crisis Chronology:**
   - Timeline with key events
   - Leading indicators
   - Transmission mechanisms

2. **Multi-Framework Explanation:**
   - PK: Debt-deflation, financial fragility
   - Marxian: Profit squeeze, overaccumulation
   - Institutionalist: Deregulation, power shifts

3. **Policy Matrix:**

| Framework | Diagnosis | Policy Prescription |
|-----------|-----------|---------------------|
| Post-Keynesian | Financial instability | Regulation, fiscal stimulus, debt restructuring |
| Marxian | Capitalist contradiction | Strengthen labor, redistribute, consider alternatives |
| Institutionalist | Institutional failure | Re-regulate, reform governance, new institutions |

### Policy Implications

**Short-term (Crisis Response):**
- All: Massive fiscal stimulus
- PK: Central bank liquidity provision
- Marxian: Protect workers, not bankers
- Institutionalist: Institutional reform

**Long-term (Prevention):**
- PK: Financial regulation, full employment policy
- Marxian: Redistribute income, empower labor
- Institutionalist: Institutional redesign, governance reform

### Key References

- Crotty, J. (2009). Structural causes of the global financial crisis. *Cambridge Journal of Economics*, 33(4), 563-580.
- Kotz, D. M. (2009). The financial and economic crisis of 2008. *Review of Radical Political Economics*, 41(3), 305-317.
- Palley, T. I. (2009). America's exhausted paradigm. *New America Foundation*.

---

## Conclusion

These workflows demonstrate the power of pluralist economic analysis. Each framework illuminates different aspects of economic reality, and together they provide richer understanding than any single perspective.

**Best Practices for Academic Use:**

1. **Be explicit about frameworks:** State which theoretical lens you're using
2. **Acknowledge limitations:** No framework captures everything
3. **Use multiple perspectives:** Triangulate findings
4. **Connect to theory:** Ground empirical work in theoretical traditions
5. **Consider policy:** Heterodox economics is inherently policy-oriented
6. **Cite properly:** Credit theorists and methodologies

**Remember:** The goal is not to "prove" one framework correct, but to gain multidimensional insight into complex economic phenomena.

Good luck with your research!
