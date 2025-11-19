
# Heterodox Macro Dashboard - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Management](#data-management)
3. [Framework Analysis](#framework-analysis)
4. [Visualizations](#visualizations)
5. [Report Generation](#report-generation)
6. [Example Workflows](#example-workflows)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First Launch

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Automatic data loading:**
   - The application automatically attempts to load default datasets on startup
   - You'll see a confirmation dialog showing how many datasets were loaded
   - If datasets fail to load, check that the `../datasets/` directory exists

3. **Interface overview:**
   - **Data Management Tab**: Load, explore, and export datasets
   - **Framework Analysis Tab**: Run theoretical framework analysis
   - **Report Generation Tab**: Create comprehensive reports

### System Requirements

- **Display**: Minimum 1280x800 resolution recommended
- **Memory**: 4GB RAM minimum (8GB recommended for large datasets)
- **Storage**: 500MB free space for datasets and reports

---

## Data Management

### Loading Default Datasets

**Method 1: Automatic (Recommended)**
- Datasets load automatically on startup
- Includes: macro, inequality, SFC, crisis, and panel data

**Method 2: Manual**
1. Click `File > Load Default Datasets` (Ctrl+L)
2. Wait for confirmation dialog
3. Check "Dataset Information" section for details

### Loading Custom Datasets

1. Click `File > Load Custom Dataset`
2. Browse to your CSV file
3. Enter a name for the dataset
4. Click "Open"

**CSV Format Requirements:**
- First row must be column headers
- Date columns should be named: `date`, `quarter`, `year`, or `period`
- Numeric data should use standard formats (no currency symbols)
- Missing values can be blank or `NA`

### Exploring Data

1. Select dataset from dropdown menu
2. View information:
   - Number of rows and columns
   - Load timestamp
   - List of available variables

3. Data Preview Table:
   - Shows first 100 rows
   - All columns visible
   - Sortable by clicking column headers
   - Not editable (view only)

### Exporting Data

1. Select dataset to export
2. Click "Export Dataset"
3. Choose format:
   - CSV (recommended for Excel, R, Python)
   - Excel (.xlsx)
   - Stata (.dta)
4. Choose location and filename
5. Click "Save"

---

## Framework Analysis

### Running Analysis

#### Post-Keynesian Analysis

**Focus:** Effective demand, sectoral balances, financial fragility

1. Select "Post Keynesian" from framework dropdown
2. Click "Run Analysis"
3. View results in three tabs:

**Analysis Results Tab:**
- Current wage share and profit share
- Sectoral balance positions
- Debt-to-GDP ratios
- Capacity utilization
- Economic interpretations

**Key Indicators:**
- Wage Share: Labor's share of national income
- Sectoral Balances: Government, private, foreign
- Debt Ratios: Financial fragility measures
- Investment Rate: Capital accumulation

**Theoretical Notes:**
- Based on Kalecki, Minsky, Godley
- Emphasizes demand constraints
- Stock-flow consistency

#### Marxian Analysis

**Focus:** Class struggle, exploitation, crisis tendencies

1. Select "Marxian" from framework dropdown
2. Click "Run Analysis"

**Key Indicators:**
- Rate of Profit: Profitability of capital
- Rate of Surplus Value: Degree of exploitation
- Organic Composition: Capital intensity
- Unemployment Rate: Reserve army of labor

**Interpretations:**
- Falling profit rate: Crisis tendency
- Rising organic composition: Labor-saving tech change
- Wage/productivity gap: Increasing exploitation

**Theoretical Notes:**
- Based on Marx, Shaikh, Foley
- Emphasizes contradictions of accumulation
- Class conflict over distribution

#### Institutionalist Analysis

**Focus:** Power relations, institutions, comparative systems

1. Select "Institutionalist" from framework dropdown
2. Click "Run Analysis"

**Key Indicators:**
- Financialization Ratio: Financial sector dominance
- Inequality Measures: Power imbalances
- Government Size: Countervailing power
- Union Density: Labor organization

**Interpretations:**
- High financialization: Veblen's absentee ownership
- Rising inequality: Power imbalances
- Institutional change: Path dependence

**Theoretical Notes:**
- Based on Veblen, Galbraith, Myrdal
- Emphasizes power and provisioning
- Evolutionary change

### Comparing Frameworks

1. Click "Compare All Frameworks" button
2. View comparative report showing:
   - Each framework's key indicators
   - Different interpretations of same data
   - Complementary insights

**Use Cases:**
- Teaching pluralist economics
- Comprehensive research analysis
- Understanding multi-dimensional dynamics

---

## Visualizations

### Available Charts

#### 1. Sectoral Balances (Godley Approach)

**What it shows:**
- Government, private, and foreign sector balances
- Must sum to zero (accounting identity)

**How to interpret:**
- Government deficit = Private + Foreign surplus
- Useful for understanding who is saving/borrowing
- Key for Post-Keynesian functional finance

**Generate:**
1. Go to Visualizations tab
2. Select "Sectoral Balances (Godley)"
3. Click "Generate Chart"

#### 2. Wage vs Profit Share

**What it shows:**
- Functional income distribution over time
- Wage share and profit share trends

**How to interpret:**
- Falling wage share = profit squeeze on workers
- Distribution affects demand regime
- Key for both PK and Marxian analysis

**Generate:**
1. Select "Wage vs Profit Share"
2. Click "Generate Chart"

#### 3. Rate of Profit (Marxian)

**What it shows:**
- Profitability of capital over time
- Trend line showing tendency

**How to interpret:**
- Falling rate = Marx's crisis tendency
- Rising rate = counter-tendencies dominant
- Key for Marxian crisis theory

**Generate:**
1. Select "Rate of Profit (Marxian)"
2. Click "Generate Chart"

#### 4. Lorenz Curve

**What it shows:**
- Income/wealth inequality distribution
- Gini coefficient displayed

**How to interpret:**
- Closer to diagonal = more equal
- Further from diagonal = more unequal
- Area between curve and diagonal = Gini

**Generate:**
1. Select "Lorenz Curve"
2. Click "Generate Chart"

#### 5. Correlation Matrix

**What it shows:**
- Correlations between economic variables
- Heatmap format with values

**How to interpret:**
- Red = positive correlation
- Blue = negative correlation
- Intensity = strength

**Generate:**
1. Select "Correlation Matrix"
2. Click "Generate Chart"

#### 6. Time Series Analysis

**What it shows:**
- Multiple variables over time
- Trends and cyclical patterns

**Generate:**
1. Select "Time Series Analysis"
2. Click "Generate Chart"

### Chart Tools

All charts include:
- **Zoom**: Click and drag to zoom
- **Pan**: Hold right-click and drag
- **Home**: Reset to original view
- **Save**: Export as PNG, PDF, or SVG

---

## Report Generation

### Creating Reports

1. Navigate to "Report Generation" tab

2. **Select Frameworks:**
   - Check boxes for frameworks to include
   - Can select one, two, or all three

3. **Select Sections:**
   - Theoretical Background: Framework foundations
   - Data Summary Statistics: Dataset information
   - Visualization Descriptions: Chart interpretations

4. Click "Generate Report"

5. Review in preview window

6. Click "Export Report" to save

### Report Structure

**Header:**
- Title and timestamp
- Executive summary

**Data Sources:**
- List of datasets used
- Number of observations
- Variables included

**Framework Analysis:**
- For each selected framework:
  - Key indicators with values
  - Economic interpretations
  - Theoretical context

**Comparative Analysis:**
- Cross-framework insights
- Common indicators compared
- Complementary perspectives

**Policy Implications:**
- Framework-specific recommendations
- Based on analysis results

**Methodology:**
- Theoretical foundations
- Data sources
- Calculation methods

**Footer:**
- Tool information
- Citation suggestion

### Using Reports

**For Academic Papers:**
- Copy relevant sections
- Cite tool and data sources
- Include charts as figures

**For Teaching:**
- Use as lecture material
- Compare framework interpretations
- Discuss policy implications

**For Policy Analysis:**
- Extract key findings
- Review policy recommendations
- Consider multiple perspectives

---

## Example Workflows

### Workflow 1: Assessing Demand Regime (Wage-led vs Profit-led)

**Research Question:** Is this economy wage-led or profit-led?

**Steps:**

1. Load macro dataset

2. Run Post-Keynesian analysis
   - Check current wage share vs historical average
   - Examine capacity utilization

3. Generate "Wage vs Profit Share" chart
   - Look for trends

4. Generate "Time Series Analysis" for consumption and investment
   - See which responds more to distribution changes

5. Interpretation:
   - **Wage-led:** Rising wage share → higher consumption → higher output
   - **Profit-led:** Rising profit share → higher investment → higher output

6. Export analysis report with interpretations

**Academic Reference:**
- Lavoie, M., & Stockhammer, E. (2013). Wage-led Growth: An Equitable Strategy for Economic Recovery. Palgrave Macmillan.

### Workflow 2: Analyzing Financial Fragility

**Research Question:** Is the financial system becoming more fragile?

**Steps:**

1. Load macro and crisis datasets

2. Run Post-Keynesian analysis
   - Focus on financial stability indicators
   - Check debt-to-GDP trends

3. Generate "Sectoral Balances" chart
   - Identify which sectors are accumulating debt

4. Check Minsky fragility indicators:
   - Debt service ratios
   - Asset price growth
   - Leverage trends

5. Generate time series for:
   - Credit growth
   - Asset prices
   - Debt ratios

6. Interpretation (Minsky's stages):
   - **Hedge:** Conservative financing
   - **Speculative:** Reliance on refinancing
   - **Ponzi:** Dependence on asset appreciation

7. Export comprehensive report

**Academic Reference:**
- Minsky, H. (1986). Stabilizing an Unstable Economy. Yale University Press.

### Workflow 3: Class Conflict and Distribution

**Research Question:** How has class struggle affected distribution?

**Steps:**

1. Load macro and inequality datasets

2. Run Marxian analysis
   - Examine rate of surplus value
   - Check wage share trends
   - Review unemployment rate (reserve army)

3. Generate visualizations:
   - Wage vs Profit Share
   - Lorenz Curve
   - Rate of Profit

4. Calculate:
   - Productivity growth vs wage growth gap
   - Changes in labor's bargaining power

5. Run Institutionalist analysis
   - Check union density trends
   - Examine power relations indicators

6. Compare frameworks:
   - Marxian: Exploitation perspective
   - Institutionalist: Power relations perspective

7. Generate comparative report

**Academic Reference:**
- Shaikh, A. (2016). Capitalism: Competition, Conflict, Crises. Oxford University Press.

### Workflow 4: Comparing Welfare Regimes

**Research Question:** How do different countries' institutional configurations affect outcomes?

**Steps:**

1. Load panel dataset (cross-country data)

2. Run Institutionalist analysis
   - Identify welfare regime types
   - Compare institutional indicators

3. For each country group:
   - Calculate inequality measures
   - Check government size
   - Review economic performance

4. Generate comparative visualizations:
   - Correlation matrix by regime type
   - Distribution of outcomes

5. Interpretation:
   - Varieties of capitalism approach
   - Path dependence
   - Institutional complementarities

6. Export findings

**Academic Reference:**
- Hall, P. A., & Soskice, D. (2001). Varieties of Capitalism. Oxford University Press.

### Workflow 5: Crisis Analysis

**Research Question:** What caused the financial crisis?

**Steps:**

1. Load crisis dataset

2. Run all three frameworks:
   - **Post-Keynesian:** Minsky's financial instability
   - **Marxian:** Profit squeeze and overaccumulation
   - **Institutionalist:** Deregulation and financialization

3. Generate time series charts:
   - Before, during, and after crisis
   - Key indicators for each framework

4. Identify:
   - Pre-crisis warning signs
   - Crisis triggers
   - Recovery patterns

5. Compare framework interpretations:
   - PK: Debt accumulation and fragility
   - Marxian: Falling profit rate
   - Institutionalist: Institutional change and power shifts

6. Generate comprehensive multi-framework report

7. Draw policy implications from each perspective

**Academic References:**
- Multiple interpretations in: Crotty, J. (2009). Structural causes of the global financial crisis. Cambridge Journal of Economics.

---

## Troubleshooting

### Data Issues

**Problem:** Datasets fail to load

**Solutions:**
1. Check that `../datasets/` directory exists
2. Verify CSV files are present
3. Check file permissions (must be readable)
4. Review error messages in status bar
5. Try loading datasets manually one at a time

**Problem:** Custom CSV won't load

**Solutions:**
1. Ensure first row contains headers
2. Check for special characters in column names
3. Verify numeric columns don't contain text
4. Save CSV with UTF-8 encoding
5. Check for empty rows at beginning/end

### Analysis Issues

**Problem:** Analysis fails or returns errors

**Solutions:**
1. Ensure datasets are loaded first
2. Check that required variables exist:
   - Macro: gdp, consumption, investment, wages, profits
   - Inequality: income, wealth variables
   - SFC: balance columns
3. Verify data has sufficient observations
4. Check for missing values

**Problem:** Indicators show as "None" or "N/A"

**Solutions:**
1. Variable may not exist in dataset
2. Check variable names in Data Management tab
3. Insufficient data for calculation
4. Review data dictionary for required variables

### Visualization Issues

**Problem:** Chart doesn't display

**Solutions:**
1. Ensure analysis has been run first
2. Check that required data exists
3. Try different chart types
4. Restart application if display freezes

**Problem:** Chart is blank or shows no data

**Solutions:**
1. Verify dataset contains the required variables
2. Check for all-missing data in variables
3. Review data range (may need to filter dates)

### Performance Issues

**Problem:** Application runs slowly

**Solutions:**
1. Close other applications to free memory
2. Use smaller datasets for testing
3. Limit number of variables in correlations
4. Export and analyze subsets of data

**Problem:** Application freezes

**Solutions:**
1. Wait for long operations to complete
2. Check system resources
3. Restart application
4. Try with default datasets first

### Report Issues

**Problem:** Report generation fails

**Solutions:**
1. Ensure analysis has been run first
2. Check that frameworks are selected
3. Verify write permissions for export location
4. Try generating single-framework reports first

### General Tips

1. **Save your work:** Export reports and charts regularly
2. **Test with defaults:** Use default datasets to verify functionality
3. **Read error messages:** Status bar shows helpful information
4. **Check documentation:** README has detailed technical information
5. **Version compatibility:** Ensure all dependencies are up to date

### Getting Help

If problems persist:
1. Check GitHub issues for similar problems
2. Review README.md for technical details
3. Examine log files (if implemented)
4. Report bugs with:
   - Error messages
   - Steps to reproduce
   - Dataset characteristics
   - System information

---

## Keyboard Shortcuts

- **Ctrl+L**: Load default datasets
- **Ctrl+Q**: Quit application
- **Tab**: Navigate between form fields
- **Enter**: Activate selected button
- **Escape**: Close dialogs

---

## Best Practices

### For Research

1. **Document your workflow:**
   - Note which frameworks used
   - Record parameter choices
   - Save all charts and reports

2. **Cite properly:**
   - Tool version
   - Data sources
   - Theoretical frameworks used

3. **Validate results:**
   - Cross-check with other tools
   - Verify against published data
   - Test sensitivity to assumptions

### For Teaching

1. **Start simple:**
   - Demonstrate one framework at a time
   - Use clear, relevant examples
   - Build to comparative analysis

2. **Encourage exploration:**
   - Let students try different frameworks
   - Compare interpretations
   - Discuss policy implications

3. **Integrate with readings:**
   - Pair with primary sources
   - Connect to theoretical texts
   - Link to current events

### For Policy Analysis

1. **Use multiple frameworks:**
   - Compare perspectives
   - Identify robust findings
   - Note disagreements

2. **Consider context:**
   - Institutional specifics
   - Historical patterns
   - Structural factors

3. **Be transparent:**
   - State assumptions
   - Acknowledge limitations
   - Present alternative interpretations

---

## Updates and Support

- **Check for updates:** Review GitHub repository regularly
- **Feature requests:** Submit via GitHub issues
- **Bug reports:** Include detailed reproduction steps
- **Contributions:** Pull requests welcome

---

## Conclusion

The Heterodox Macro Dashboard is designed to make sophisticated economic analysis accessible to researchers, students, and policy analysts. By implementing multiple theoretical frameworks, it enables truly pluralist economic analysis.

Remember: Economic reality is complex and multi-faceted. No single framework captures all aspects. Use this tool to explore different perspectives and develop richer understanding.

Happy analyzing!
