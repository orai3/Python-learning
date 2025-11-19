# Methodology: Unequal Exchange Framework

## Theoretical Foundations

### 1. Labor Theory of Value

The framework is grounded in the classical-Marxian labor theory of value:

- **Socially Necessary Labor Time (SNLT)**: Commodities embody labor hours required for production
- **Value vs. Price**: Commodities sell at prices of production, not labor values
- **Surplus Value**: s = Output Value - Wages
- **Rate of Exploitation**: e = s/v (surplus value / variable capital)

### 2. Unequal Exchange Mechanisms

#### 2.1 Emmanuel's Unequal Exchange (1972)

**Core Argument**: International trade transfers value from low-wage to high-wage countries even when profit rates equalize.

**Assumptions**:
- Capital mobility → equalized profit rates globally
- Labor immobility → persistent wage differentials
- Commodities sell at prices of production

**Mathematical Formulation**:

Price of production:
```
p = (1 + r)(wl + pA)
```
Where:
- p = price
- r = profit rate (equalized internationally)
- w = wage rate (differs by country)
- l = labor coefficient
- A = technical coefficients
- pA = cost of intermediate inputs

Value transfer from country S (South) to country N (North):
```
VT = L_exported × (w_N - w_S)
```
Where L_exported = labor embodied in South's exports to North

**Implementation** (emmanuel.py):
```python
def emmanuel_transfer(south_prod, north_prod, trade_value):
    labor_content = south_prod.labor_hours / south_prod.gross_output
    labor_in_trade = labor_content * trade_value
    wage_differential = north_prod.wage_rate - south_prod.wage_rate
    transfer = labor_in_trade * wage_differential
    return transfer
```

#### 2.2 Amin's Extended Model (1974)

**Extensions Beyond Emmanuel**:
1. Incorporates productivity differentials
2. Identifies super-exploitation (wages < value of labor-power)
3. Analyzes blocked development

**Productivity-Adjusted Transfer**:
```
Expected_Wage_S = Wage_N × (Productivity_S / Productivity_N)
Super_Exploitation_Gap = Expected_Wage_S - Actual_Wage_S
Total_Transfer = Emmanuel_Transfer + SE_Gap × Labor_Exported
```

**Implementation** (amin.py):
```python
def amin_transfer(south_prod, north_prod, trade_value):
    # Productivity ratio
    prod_ratio = south_prod.labor_productivity / north_prod.labor_productivity

    # Expected wage if only productivity mattered
    expected_wage = north_prod.wage_rate * prod_ratio

    # Super-exploitation gap
    se_gap = expected_wage - south_prod.wage_rate

    # Labor in exports
    labor_content = south_prod.labor_hours / south_prod.gross_output
    labor_exported = labor_content * trade_value

    # Total transfer
    return labor_exported * se_gap
```

#### 2.3 Prebisch-Singer Hypothesis (1950)

**Core Mechanism**: Terms of trade deteriorate for primary commodity exporters.

**Causes**:
1. Income elasticity: E_demand(primary) < E_demand(manufactures)
2. Market structure: Monopolistic manufactures vs competitive primary markets
3. Labor organization: Strong unions in North vs weak in South

**Terms of Trade Index**:
```
ToT = (P_exports / P_imports) × 100
```

Declining ToT means: Same export volume buys fewer imports over time.

**Income Effect**:
```
Income_Loss = Export_Volume × (ToT_base - ToT_current) / 100
```

**Implementation** (prebisch_singer.py):
```python
def calculate_terms_of_trade(export_prices, import_prices, base_year):
    # Normalize to base year = 100
    exp_normalized = (export_prices / export_prices[base_year]) × 100
    imp_normalized = (import_prices / import_prices[base_year]) × 100

    # ToT index
    tot = (exp_normalized / imp_normalized) × 100

    return tot
```

### 3. Global Value Chain Analysis

#### 3.1 Smile Curve

**Concept**: Value distribution is U-shaped across value chain:
- High value: R&D/design, marketing/brand
- Low value: Manufacturing/assembly

**Measurement**:
```
Value_Share_i = (VA_i / Σ VA_j) × 100
Smile_Intensity = (VA_upstream + VA_downstream) / (2 × VA_midstream)
```

#### 3.2 Rent Decomposition

Following Durand & Milberg (2020), rents decomposed into:

1. **Monopoly Rents**: From market power
2. **IP Rents**: From patents, trademarks, copyrights
3. **Technological Rents**: From innovation
4. **Organizational Rents**: From GVC governance

**Calculation**:
```python
def estimate_rents(segment):
    profit = segment.value_added - segment.labor_cost
    competitive_profit = segment.labor_cost × competitive_rate
    excess_profit = max(0, profit - competitive_profit)

    # Decompose
    monopoly_rent = excess_profit × segment.market_power
    ip_rent = excess_profit × segment.ip_intensity
    tech_rent = excess_profit × (1 - market_power - ip_intensity)

    return {
        'monopoly': monopoly_rent,
        'ip': ip_rent,
        'technological': tech_rent,
        'total': monopoly_rent + ip_rent + tech_rent
    }
```

### 4. Multi-Country Input-Output Analysis

#### 4.1 Leontief Framework

**Basic Equation**:
```
x = (I - A)^(-1) f
```
Where:
- x = gross output vector (n×1)
- A = technical coefficients matrix (n×n)
- f = final demand vector (n×1)
- (I - A)^(-1) = Leontief inverse

**Multi-Country Extension**:
For N countries with M sectors each:
- Dimension: (N×M) × (N×M)
- A_ij^rs = intermediate input from sector i in country r used by sector j in country s

#### 4.2 Value Added Decomposition

**Vertically Integrated Labor**:
```
l = l_d (I - A)^(-1)
```
Where:
- l = total labor per unit final demand
- l_d = direct labor coefficients

**Value Added in Final Demand**:
```
VA = v̂ (I - A)^(-1) f
```
Where v̂ = diagonal matrix of value-added coefficients

**Embodied Labor in Trade**:
```
L_exports^r = Σ_s l_r B^rs f_s
```
Where:
- l_r = labor coefficients for country r
- B^rs = Leontief inverse block for r→s
- f_s = final demand in country s

#### 4.3 GVC Participation Metrics

**Backward Participation** (foreign value in exports):
```
BP = (Foreign_VA_in_Exports) / (Gross_Exports)
```

**Forward Participation** (domestic VA in others' exports):
```
FP = (Domestic_VA_in_Foreign_Exports) / (Gross_Exports)
```

**Total GVC Participation**:
```
GVC_Part = BP + FP
```

### 5. Super-Exploitation Metrics

#### 5.1 Wage-Productivity Gap

**Measure**:
```
Gap = (Productivity_Growth_Rate - Wage_Growth_Rate)
```

If Gap > 0 persistently → increasing exploitation

**Relative Exploitation** (vs reference country):
```
RE = (Prod/Wage)_South / (Prod/Wage)_North
```

If RE > 1 → South more exploited

#### 5.2 Labor Share

**Definition**:
```
Labor_Share = (Wages / Value_Added) × 100
```

Declining labor share → increasing capital share → rising exploitation

**Marxian Rate of Exploitation**:
```
e = (Value_Added - Wages) / Wages
  = Surplus_Value / Variable_Capital
```

#### 5.3 International Value Transfer

**Productivity-Adjusted Counterfactual**:
```
Counterfactual_Wages = Wage_North × (Productivity_South / Productivity_North)
Value_Transfer = (Counterfactual_Wages - Actual_Wages) × Labor_Hours
```

### 6. Transfer Pricing & Profit Shifting

#### 6.1 Price Filter Method

Compare declared prices vs market prices:
```
Price_Gap = Declared_Price - Market_Price
Misinvoicing = |Price_Gap| if |Price_Gap/Market_Price| > threshold
```

Typical threshold: 20-25%

#### 6.2 Profit Shifting Estimation

**Country-by-Country Analysis**:
```
Expected_Profit_i = (Employees_i / Total_Employees) × Total_Profit
Excess_Profit_i = Actual_Profit_i - Expected_Profit_i
```

If country i is tax haven and Excess_Profit_i > 0 → likely profit shifting

#### 6.3 Tax Loss Calculation

```
Tax_Loss = Profit_Shifted × Corporate_Tax_Rate
```

### 7. Policy Simulations

#### 7.1 South-South Cooperation

**Key Parameters**:
- Intra-South trade share (target: 30-50%)
- Technology transfer rate (annual productivity gain)
- Collective bargaining effect (ToT improvement)

**Simulation Logic**:
```python
for year in range(T):
    # Trade reorientation
    intra_south_share = min(target, initial + year × growth_rate)

    # Productivity gains from tech transfer
    productivity += productivity × tech_transfer_rate

    # Terms of trade improvement
    tot += tot × collective_bargaining_effect

    # Value transfer reduction
    transfer_reduction = baseline_transfer × (1 - intra_south_share × effectiveness)
```

#### 7.2 Delinking

**Three Strategies**:
1. Moderate: Gradual import substitution, maintain selective integration
2. Radical: Rapid delinking, prioritize autocentric development
3. Selective: Strategic delinking in key sectors

**Trade-offs**:
- Short-term costs: Productivity hit from reduced access to imports
- Long-term gains: Retained surplus, autocentric accumulation

**Formulation**:
```
GDP_t = GDP_0 × (1 + growth_rate_t)

Where:
growth_rate_t = base_growth - productivity_penalty (if t < 5)
                + accumulation_bonus (if t ≥ 5)
```

#### 7.3 Industrial Policy

**Components**:
1. Infant industry protection
2. R&D investment
3. Skill development
4. Export diversification

**Effects**:
```
Manufacturing_Share_t = min(target, initial + t × growth_rate)
Productivity_t = Productivity_0 × (1 + skill_development_rate)^t
Value_Chain_Position_t = initial_position + t × upgrading_rate
```

### 8. Validation & Robustness

#### 8.1 Synthetic Data Generation

**Calibration to Stylized Facts**:
1. Core-periphery wage differentials: 5-10×
2. ToT deterioration: 20-30% over 1960-2020
3. Labor share decline: -10 to -15 percentage points
4. Rising financialization: Profit/IP flows increase 3-5×

**Data Generating Process**:
```python
# Wage evolution
wage_t = wage_0 × (country_factor) × (1 + growth_rate)^t

# With convergence for successful semi-periphery
if country in converging:
    wage_t += (core_wage - wage_t) × convergence_rate

# Crisis shocks
if year in crisis_years:
    wage_t × (1 + crisis_effect)
```

#### 8.2 Sensitivity Analysis

**Parameters to Test**:
- Profit rate (r): 10-20%
- Productivity differentials: 2×-5×
- Wage gaps: 5×-10×
- Policy intensities: 0.3-0.9

**Metrics**:
- Elasticity of value transfer to wage gap
- Sensitivity of policy outcomes to intensity
- Robustness to crisis shocks

## References

### Core Theoretical Works

1. Emmanuel, A. (1972). *Unequal Exchange*. Monthly Review Press.
2. Amin, S. (1974). *Accumulation on a World Scale*. Monthly Review Press.
3. Prebisch, R. (1950). "Economic Development of Latin America." *ECLA*.
4. Singer, H. (1950). "Distribution of Gains from Trade." *AER*.

### Contemporary Applications

5. Hickel, J., Sullivan, D., & Zoomkawala, H. (2021). "Plunder in the Post-Colonial Era." *New Political Economy*.
6. Cope, Z. (2019). *The Wealth of (Some) Nations*. Pluto Press.
7. Durand, C., & Milberg, W. (2020). "Intellectual Monopoly in GVCs." *RIPE*.
8. Timmer, M. et al. (2015). "An Illustrated User Guide to WIOD." *RIDE*.

### Methodological Foundations

9. Miller, R., & Blair, P. (2009). *Input-Output Analysis*. Cambridge UP.
10. Los, B., Timmer, M., & de Vries, G. (2016). "Tracing Value-Added in GVCs." *JIE*.
11. Crivelli, E., de Mooij, R., & Keen, M. (2016). "Base Erosion, Profit Shifting." *IMFSP*.

## Software Implementation Notes

### Performance Considerations

- **Matrix Operations**: Use NumPy for large-scale I-O calculations
- **Data Storage**: Pandas for time-series and panel data
- **Visualization**: Matplotlib/Seaborn for publication-quality plots

### Numerical Stability

- Check matrix invertibility before Leontief inverse
- Handle zero division in ratio calculations
- Use logarithmic transformations for growth rates
- Normalize large values to avoid floating-point errors

### Extensibility

Framework designed for:
- Integration with real datasets (WIOD, OECD, UN Comtrade)
- Additional models (Wallerstein, Frank, dependency theorists)
- Econometric estimation (GMM, panel methods)
- Machine learning for pattern recognition
