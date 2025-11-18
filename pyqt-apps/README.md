# Phase 2: PyQt6 Economic Simulation Applications

This directory contains interactive PyQt6 applications implementing heterodox economic models. Each application provides real-time visualization, parameter controls, and data export capabilities for teaching and research.

## 🎯 Learning Progression

The applications are ordered by complexity:

1. **Keynesian Multiplier** - Basic PyQt concepts, simple calculations
2. **Kalecki Profits** - Multiple visualizations, scenario analysis
3. **Goodwin Cycle** - Differential equations, animation, phase diagrams
4. **Minsky Instability** - Complex simulation, state evolution
5. **Kaleckian Growth** - Regime analysis, comparative statics

## 📚 Applications Overview

### 01. Keynesian Multiplier Calculator
**File:** `01_keynesian_multiplier.py`

**Economic Theory:**
- Classic Keynesian multiplier effect
- How changes in autonomous spending affect national income
- Y = C + I + G framework
- Multiplier k = 1/(1-MPC)

**Features:**
- Real-time parameter adjustment (MPC, autonomous consumption, investment, government spending)
- Keynesian cross diagram visualization
- Equilibrium calculations
- CSV data export

**Learning Focus:**
- Basic PyQt widgets (QSpinBox, QLabel)
- Simple matplotlib integration
- Event handling in PyQt

**Run:**
```bash
python 01_keynesian_multiplier.py
```

**Key Insights:**
- Higher MPC → larger multiplier effect
- Government spending has same multiplier as private investment
- Saving paradox: More saving can reduce income

---

### 02. Kalecki Profit Equation Simulator
**File:** `02_kalecki_profits.py`

**Economic Theory:**
- Post-Keynesian profit determination
- "Workers spend what they earn, capitalists earn what they spend"
- P = I + (G-T) + (X-M) + Cₚ - Sᵥᵥ
- Class-based macroeconomics

**Features:**
- Component breakdown visualization (bar charts, waterfall charts)
- Time series tracking
- Scenario presets (austerity, fiscal stimulus, trade strategies)
- Fiscal and trade balance analysis

**Learning Focus:**
- Multiple matplotlib subplots
- Tabbed interfaces (QTabWidget)
- Combo boxes for scenarios
- More complex layouts

**Run:**
```bash
python 02_kalecki_profits.py
```

**Key Insights:**
- Profits depend on capitalist spending decisions (investment, consumption)
- Government deficits boost profits
- Trade surpluses benefit domestic profits
- Worker savings reduce profit realization

---

### 03. Goodwin Growth Cycle Model
**File:** `03_goodwin_cycle.py`

**Economic Theory:**
- Marxian/Post-Keynesian cyclical growth
- Predator-prey dynamics applied to class struggle
- Employment rate vs wage share cycles
- Reserve army of labor mechanism

**Mathematical Model:**
```
dv/dt = v(a - b·u)    [Employment dynamics]
du/dt = u(c·v - d)    [Wage share dynamics]
```

**Features:**
- ODE solver integration (scipy.integrate.odeint)
- Phase diagram with vector field
- Animated trajectory evolution
- Time series plots
- Real-time parameter sensitivity

**Learning Focus:**
- QTimer for animation
- Differential equation solving
- Vector field plotting
- Advanced matplotlib (quiver plots)
- Animation controls

**Run:**
```bash
python 03_goodwin_cycle.py
```

**Key Insights:**
- Capitalism has inherent cyclical dynamics
- High employment → wage pressure → profit squeeze → unemployment
- No stable equilibrium (unlike neoclassical models)
- Class conflict is structural feature of capitalism

---

### 04. Minsky Financial Instability Hypothesis
**File:** `04_minsky_instability.py`

**Economic Theory:**
- Post-Keynesian financial economics
- "Stability is destabilizing"
- Evolution from hedge → speculative → Ponzi financing
- Endogenous financial crises

**Borrower Classification:**
- **Hedge:** Can pay principal + interest from income
- **Speculative:** Must roll over principal
- **Ponzi:** Must borrow to pay interest

**Features:**
- Stacked area charts showing financial structure evolution
- Asset price and leverage tracking
- Crisis detection (Minsky moments)
- Scenario buttons (post-crisis, deregulation, bubble, rate shock)
- Real-time simulation with animation
- Crisis indicator/fragility index

**Learning Focus:**
- Complex state management
- Multiple coordinated visualizations
- QSlider for continuous parameters
- Stacked area charts
- Color-coded regime identification

**Run:**
```bash
python 04_minsky_instability.py
```

**Key Insights:**
- Financial crises are endogenous, not external shocks
- Long stability periods breed instability
- Deregulation accelerates shift to fragile financing
- Asset bubbles and Ponzi finance go together
- Explains 2008 financial crisis dynamics

---

### 05. Kaleckian Growth Model
**File:** `05_kaleckian_growth.py`

**Economic Theory:**
- Post-Keynesian demand-led growth
- Growth determined by investment, not savings
- Wage-led vs profit-led growth regimes
- Paradox of thrift

**Mathematical Model:**
```
Investment: g = α₀ + α₁·π + α₂·u
Saving:     s = s_π·π·u + s_ω·(1-π)·u
Equilibrium: g = s
```

**Features:**
- Equilibrium solver
- Automatic regime identification (wage-led/profit-led)
- Interactive profit share slider
- Regime sensitivity analysis
- Policy experiment buttons
- Growth vs distribution plots

**Learning Focus:**
- Analytical solutions to economic models
- Derivative calculations for regime analysis
- Twin y-axis plots
- Parameter sensitivity visualization
- Policy scenario implementation

**Run:**
```bash
python 05_kaleckian_growth.py
```

**Key Insights:**
- Growth is demand-determined, not supply-constrained
- Income distribution affects growth (not neutral)
- Wage suppression can hurt growth if economy is wage-led
- Challenges neoclassical supply-side economics
- Policy implications for austerity vs stimulus

---

## 🛠️ Installation & Requirements

### Prerequisites

```bash
pip install PyQt6 matplotlib numpy scipy
```

Or use the project's requirements.txt:
```bash
pip install -r ../requirements.txt
```

### System Requirements

- Python 3.8+
- PyQt6
- matplotlib
- numpy
- scipy (for Goodwin cycle ODE solver)

### Running Applications

Each application is standalone. Run directly:

```bash
cd pyqt-apps
python 01_keynesian_multiplier.py
```

## 🎓 Educational Use

### For Self-Study

**Week 1:** Keynesian Multiplier
- Understand basic PyQt structure
- Modify parameters, observe effects
- Export data and analyze in spreadsheet

**Week 2:** Kalecki Profits
- Explore heterodox profit theory
- Run scenario experiments
- Compare to mainstream theory

**Week 3:** Goodwin Cycle
- Learn differential equations in economics
- Experiment with parameters to change cycle amplitude
- Connect to real business cycles

**Week 4:** Minsky Instability
- Understand financial crisis dynamics
- Simulate regulatory scenarios
- Apply to historical crises (2008, 1929)

**Week 5:** Kaleckian Growth
- Explore wage-led vs profit-led debate
- Test policy interventions
- Connect to current economic policy debates

### For Teaching

Each application includes:
- **Theory box** with economic explanation
- **Visual aids** for classroom demonstration
- **Interactive controls** for live parameter changes
- **Data export** for student assignments

**Assignment Ideas:**
1. Compare multiplier effects under different MPCs
2. Analyze which policies boost Kaleckian profits
3. Identify Goodwin cycle periods in real data
4. Simulate 2008 crisis using Minsky model
5. Determine if your country is wage-led or profit-led

### For Research

**Data Export:**
All applications export to CSV format for further analysis in R, Python, Stata, etc.

**Calibration:**
Adjust parameters to match real economies:
- Use national accounts data for Kalecki model
- Estimate Phillips curve for Goodwin model
- Use financial sector data for Minsky model

**Extensions:**
Code is well-commented for modification:
- Add new scenarios
- Implement policy rules
- Connect to real data sources
- Add stochastic elements

---

## 🏗️ PyQt Architecture Notes

### Common Patterns Used

**1. Model-View Separation**
```python
class MyApp(QMainWindow):
    def __init__(self):
        # Model: economic parameters
        self.parameter1 = value1

        # View: UI components
        self.init_ui()

        # Controller: connect signals
        self.widget.valueChanged.connect(self.on_change)
```

**2. Layout Management**
- `QHBoxLayout` - horizontal layouts (left panel + right panel)
- `QVBoxLayout` - vertical layouts (stacked controls)
- `QGridLayout` - grid layouts (parameter labels + inputs)

**3. Matplotlib Integration**
```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

self.figure = Figure()
self.canvas = FigureCanvas(self.figure)
layout.addWidget(self.canvas)
```

**4. Signal-Slot Connections**
```python
self.spinbox.valueChanged.connect(self.on_parameter_change)
self.slider.valueChanged.connect(self.on_slider_change)
self.button.clicked.connect(self.on_button_click)
```

**5. Real-time Updates**
All apps recalculate and redraw on parameter changes - no "Calculate" button needed.

---

## 📖 Economic Theory References

### Keynesian Economics
- Keynes, J.M. (1936) "The General Theory of Employment, Interest and Money"

### Post-Keynesian Economics
- Kalecki, M. (1954) "Theory of Economic Dynamics"
- Lavoie, M. (2014) "Post-Keynesian Economics: New Foundations"

### Marxian Economics
- Goodwin, R.M. (1967) "A Growth Cycle" in Feinstein (ed.)

### Financial Economics
- Minsky, H. (1986) "Stabilizing an Unstable Economy"

### Growth Theory
- Hein, E. (2014) "Distribution and Growth after Keynes"

---

## 🔧 Customization Guide

### Adding New Scenarios

In any model, add to scenario list:
```python
def my_new_scenario(self):
    """Describe what this scenario does."""
    self.parameter1_spin.setValue(new_value)
    self.parameter2_spin.setValue(new_value)
    # Application auto-recalculates
```

### Modifying Visualizations

All plotting happens in `plot_*` methods:
```python
def plot_my_diagram(self):
    self.figure.clear()
    ax = self.figure.add_subplot(111)

    # Your plotting code here
    ax.plot(x, y, 'b-', linewidth=2)

    self.canvas.draw()  # Refresh display
```

### Adding New Parameters

1. Add instance variable in `__init__`
2. Add control widget in `create_control_panel`
3. Connect to `on_parameter_change`
4. Use in calculation methods

---

## 🐛 Troubleshooting

### PyQt6 Import Errors
```bash
pip install --upgrade PyQt6
```

### Matplotlib Backend Issues
If plots don't show, ensure you're using Qt5Agg backend:
```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
```

### High DPI Displays
If text/widgets appear tiny:
```python
# Add before QApplication creation
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
```

---

## 💡 Future Extensions

**Possible Additions:**

1. **Godley-Lavoie SFC Model**
   - Stock-flow consistent accounting
   - Sector financial balances
   - Matrix visualization

2. **Sraffian Input-Output Model**
   - Production prices
   - Wage-profit frontier
   - Technical change analysis

3. **Keen's Minsky Model**
   - Full dynamic Minsky model with debt
   - Debt-deflation spiral
   - Jubilee scenarios

4. **Agent-Based Models**
   - Heterogeneous agents
   - Network effects
   - Emergent phenomena

5. **Empirical Integration**
   - Load real data (FRED, OECD)
   - Parameter estimation
   - Model validation

---

## 📝 License & Attribution

These applications were generated for educational purposes in heterodox economics.

**Citation for academic use:**
```
Python Economics Toolkit (2024)
PyQt6 Applications for Heterodox Economic Models
```

**Code is provided as-is for:**
- Educational use
- Research purposes
- Teaching materials
- Further development

---

## 🤝 Contributing

To extend these applications:

1. **Keep the structure:** Model-View-Controller pattern
2. **Document theory:** Economic explanation in docstrings
3. **Comment code:** Explain both PyQt and economic logic
4. **Export data:** All models should allow CSV export
5. **Visual clarity:** Use color, labels, legends effectively

---

## 📊 Data Export Format

All applications export to CSV with:
- Header row explaining variables
- Time series or parameter sweep data
- Suitable for import to R, Stata, Excel, etc.

**Example usage:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load exported data
df = pd.read_csv('goodwin_cycle_20240101_120000.csv')

# Analyze
df.plot(x='Time', y=['EmploymentRate', 'WageShare'])
plt.show()
```

---

## 🎯 Learning Outcomes

After working through these applications, you should be able to:

**Python/PyQt Skills:**
- Build interactive GUI applications with PyQt6
- Integrate matplotlib into PyQt
- Handle user input and events
- Manage application state
- Create multi-tab interfaces
- Implement real-time updates
- Export data programmatically

**Economic Modeling:**
- Implement heterodox economic models
- Visualize economic dynamics
- Perform comparative statics
- Conduct scenario analysis
- Identify economic regimes
- Understand endogenous cycles
- Critique mainstream assumptions

**Research Skills:**
- Calibrate models to data
- Test theoretical predictions
- Generate synthetic data
- Perform sensitivity analysis
- Document methodology
- Create reproducible research tools

---

## 📞 Support & Resources

**For PyQt6 help:**
- Official docs: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- Tutorial: https://www.pythonguis.com/pyqt6-tutorial/

**For matplotlib:**
- Gallery: https://matplotlib.org/stable/gallery/
- PyQt integration: https://matplotlib.org/stable/gallery/#embedding-matplotlib-in-gui-applications

**For heterodox economics:**
- Post-Keynesian Economics Society: http://www.postkeynesian.net/
- Research on Money and Finance: https://www.soas.ac.uk/rmf/
- Institute for New Economic Thinking: https://www.ineteconomics.org/

---

**Happy modeling! 📈**

*Remember: "The ideas of economists and political philosophers, both when they are right and when they are wrong, are more powerful than is commonly understood." - J.M. Keynes*
