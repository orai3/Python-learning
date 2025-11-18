"""
Minsky Financial Instability Hypothesis - PyQt6 Application
===========================================================

Economic Theory (Heterodox - Post-Keynesian):
--------------------------------------------
Hyman Minsky (1919-1996) developed a theory explaining how financial crises
are endogenous to capitalist economies. During periods of economic stability,
firms progressively take on more risk, leading to financial fragility and
eventual crisis.

Minsky's Classification of Borrowers:
------------------------------------
1. HEDGE Units:
   - Income > Interest + Principal payments
   - Can fully service debt from current income
   - Conservative, safe financing

2. SPECULATIVE Units:
   - Income > Interest payments
   - Income < Interest + Principal
   - Must roll over debt (refinance principal)
   - Vulnerable to interest rate changes

3. PONZI Units:
   - Income < Interest payments
   - Must borrow to pay interest
   - Rely on asset price appreciation
   - Extremely fragile

The Financial Instability Hypothesis:
-----------------------------------
"Stability is destabilizing"

Phase 1 (Stability):
- Economy recovers from previous crisis
- Mostly hedge financing
- Conservative lending standards

Phase 2 (Euphoria):
- Extended period of stability
- Memories of crisis fade
- Shift to speculative financing
- Asset prices rise

Phase 3 (Mania):
- Widespread euphoria
- Ponzi financing emerges
- Asset price bubbles
- Declining lending standards

Phase 4 (Crisis):
- "Minsky Moment" - sudden reversal
- Asset prices fall
- Ponzi units insolvent
- Speculative units become Ponzi
- Credit crunch
- Recession/Depression

Relevance:
- Explains 2008 financial crisis
- Challenges efficient markets hypothesis
- Shows need for financial regulation
- Endogenous business cycles

Author: Claude AI
Target: Students/researchers of financial economics
References: Minsky, H. (1986) "Stabilizing an Unstable Economy"
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTextEdit, QGridLayout, QTabWidget,
                             QSlider, QCheckBox, QSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import csv
from datetime import datetime


class MinskyInstabilityApp(QMainWindow):
    """
    Main application for Minsky Financial Instability Hypothesis simulation.

    Features:
    - Simulate evolution of financial structure over time
    - Track distribution of hedge/speculative/Ponzi units
    - Visualize asset prices and leverage
    - Identify "Minsky moments"
    - Scenario analysis (regulation, bubbles, crashes)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Minsky Financial Instability Hypothesis")
        self.setGeometry(100, 100, 1400, 850)

        # Model parameters
        self.total_units = 1000  # Total number of borrowing units

        # Initial distribution (conservative, post-crisis)
        self.hedge_ratio = 0.80
        self.speculative_ratio = 0.18
        self.ponzi_ratio = 0.02

        # Economic parameters
        self.gdp_growth = 0.03      # Base GDP growth rate
        self.interest_rate = 0.05    # Interest rate
        self.optimism = 0.5          # Market optimism (0-1)
        self.regulation = 0.7        # Regulatory stringency (0-1)

        # Simulation parameters
        self.time_periods = 100
        self.current_period = 0

        # State variables
        self.asset_price_index = 100.0
        self.leverage_ratio = 2.0

        # History arrays
        self.history = {
            'time': [],
            'hedge': [],
            'speculative': [],
            'ponzi': [],
            'asset_price': [],
            'leverage': [],
            'gdp_growth': [],
            'crisis_indicator': []
        }

        # Animation
        self.is_animating = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.simulation_step)

        self.init_ui()
        self.reset_simulation()

    def init_ui(self):
        """Initialize the user interface."""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel: Visualizations
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, stretch=2)

    def create_control_panel(self):
        """Create the parameter control panel."""

        panel = QGroupBox("Model Controls")
        layout = QVBoxLayout()

        # Theory section
        theory_box = QGroupBox("Minsky's Theory")
        theory_layout = QVBoxLayout()
        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        theory_text.setMaximumHeight(220)
        theory_text.setHtml("""
        <h3>Financial Instability Hypothesis</h3>
        <p><b>"Stability is destabilizing"</b></p>
        <p><b>Borrower Types:</b></p>
        <ul style="margin:5px">
            <li><b style="color:green">Hedge:</b> Income covers all debt payments</li>
            <li><b style="color:orange">Speculative:</b> Must roll over principal</li>
            <li><b style="color:red">Ponzi:</b> Must borrow to pay interest</li>
        </ul>
        <p><b>The Cycle:</b></p>
        <ol style="font-size:9pt; margin:5px">
            <li>Stability → Rising confidence</li>
            <li>Risk-taking increases</li>
            <li>Shift from Hedge → Speculative → Ponzi</li>
            <li>Asset bubbles form</li>
            <li><b>Minsky Moment:</b> Crisis erupts</li>
            <li>Deleveraging → Recession</li>
        </ol>
        """)
        theory_layout.addWidget(theory_text)
        theory_box.setLayout(theory_layout)
        layout.addWidget(theory_box)

        # Parameters section
        params_box = QGroupBox("Economic Parameters")
        params_layout = QGridLayout()

        row = 0

        # GDP Growth
        params_layout.addWidget(QLabel("Base GDP Growth:"), row, 0)
        self.gdp_spin = QDoubleSpinBox()
        self.gdp_spin.setRange(-0.05, 0.10)
        self.gdp_spin.setValue(self.gdp_growth)
        self.gdp_spin.setSingleStep(0.005)
        self.gdp_spin.setDecimals(3)
        params_layout.addWidget(self.gdp_spin, row, 1)
        row += 1

        # Interest Rate
        params_layout.addWidget(QLabel("Interest Rate:"), row, 0)
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.0, 0.20)
        self.rate_spin.setValue(self.interest_rate)
        self.rate_spin.setSingleStep(0.005)
        self.rate_spin.setDecimals(3)
        params_layout.addWidget(self.rate_spin, row, 1)
        row += 1

        # Optimism
        params_layout.addWidget(QLabel("Market Optimism:"), row, 0)
        self.optimism_slider = QSlider(Qt.Orientation.Horizontal)
        self.optimism_slider.setRange(0, 100)
        self.optimism_slider.setValue(int(self.optimism * 100))
        self.optimism_slider.valueChanged.connect(self.on_slider_change)
        params_layout.addWidget(self.optimism_slider, row, 1)
        self.optimism_label = QLabel(f"{self.optimism:.2f}")
        params_layout.addWidget(self.optimism_label, row, 2)
        row += 1

        # Regulation
        params_layout.addWidget(QLabel("Regulatory Stringency:"), row, 0)
        self.regulation_slider = QSlider(Qt.Orientation.Horizontal)
        self.regulation_slider.setRange(0, 100)
        self.regulation_slider.setValue(int(self.regulation * 100))
        self.regulation_slider.valueChanged.connect(self.on_slider_change)
        params_layout.addWidget(self.regulation_slider, row, 1)
        self.regulation_label = QLabel(f"{self.regulation:.2f}")
        params_layout.addWidget(self.regulation_label, row, 2)
        row += 1

        params_box.setLayout(params_layout)
        layout.addWidget(params_box)

        # Scenario buttons
        scenario_box = QGroupBox("Scenarios")
        scenario_layout = QGridLayout()

        scenarios = [
            ("Post-Crisis Recovery", self.scenario_post_crisis),
            ("Deregulation Boom", self.scenario_deregulation),
            ("Asset Bubble", self.scenario_bubble),
            ("Rate Shock", self.scenario_rate_shock)
        ]

        for i, (name, func) in enumerate(scenarios):
            btn = QPushButton(name)
            btn.clicked.connect(func)
            scenario_layout.addWidget(btn, i // 2, i % 2)

        scenario_box.setLayout(scenario_layout)
        layout.addWidget(scenario_box)

        # Simulation controls
        sim_box = QGroupBox("Simulation")
        sim_layout = QVBoxLayout()

        self.play_button = QPushButton("▶ Start Simulation")
        self.play_button.clicked.connect(self.toggle_simulation)
        sim_layout.addWidget(self.play_button)

        self.step_button = QPushButton("Single Step")
        self.step_button.clicked.connect(self.simulation_step)
        sim_layout.addWidget(self.step_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_simulation)
        sim_layout.addWidget(self.reset_button)

        sim_box.setLayout(sim_layout)
        layout.addWidget(sim_box)

        # Status display
        status_box = QGroupBox("Current Status")
        status_layout = QVBoxLayout()

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(180)
        status_layout.addWidget(self.status_text)

        status_box.setLayout(status_layout)
        layout.addWidget(status_box)

        # Export button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        layout.addWidget(export_btn)

        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def create_visualization_panel(self):
        """Create the visualization panel."""

        tab_widget = QTabWidget()

        # Tab 1: Financial structure
        structure_tab = QWidget()
        structure_layout = QVBoxLayout()
        self.structure_figure = Figure(figsize=(10, 8))
        self.structure_canvas = FigureCanvas(self.structure_figure)
        structure_layout.addWidget(self.structure_canvas)
        structure_tab.setLayout(structure_layout)
        tab_widget.addTab(structure_tab, "Financial Structure")

        # Tab 2: Economic indicators
        indicators_tab = QWidget()
        indicators_layout = QVBoxLayout()
        self.indicators_figure = Figure(figsize=(10, 8))
        self.indicators_canvas = FigureCanvas(self.indicators_figure)
        indicators_layout.addWidget(self.indicators_canvas)
        indicators_tab.setLayout(indicators_layout)
        tab_widget.addTab(indicators_tab, "Economic Indicators")

        return tab_widget

    def on_slider_change(self):
        """Handle slider changes."""
        self.optimism = self.optimism_slider.value() / 100.0
        self.regulation = self.regulation_slider.value() / 100.0
        self.optimism_label.setText(f"{self.optimism:.2f}")
        self.regulation_label.setText(f"{self.regulation:.2f}")

    def simulation_step(self):
        """Advance simulation by one time period."""

        if self.current_period >= self.time_periods:
            if self.is_animating:
                self.toggle_simulation()
            return

        # Get current parameters
        gdp_growth = self.gdp_spin.value()
        interest_rate = self.rate_spin.value()

        # Calculate transition probabilities based on economic conditions
        # During good times (high optimism, low regulation), shift toward risky financing

        # Euphoria factor: combines optimism and economic growth
        euphoria = (self.optimism + max(0, gdp_growth) * 10) / 2.0
        euphoria = np.clip(euphoria, 0, 1)

        # Risk tolerance: inversely related to regulation
        risk_tolerance = 1.0 - self.regulation

        # Update ratios based on economic conditions
        # Hedge → Speculative transition
        hedge_to_spec = 0.02 * euphoria * risk_tolerance
        # Speculative → Ponzi transition
        spec_to_ponzi = 0.015 * euphoria * risk_tolerance

        # Crisis trigger: too many Ponzi units
        crisis_threshold = 0.25
        in_crisis = self.ponzi_ratio > crisis_threshold

        if in_crisis:
            # Crisis mode: flight to safety
            # Ponzi → default (removed)
            # Speculative → Hedge (deleveraging)
            ponzi_default = 0.10
            spec_to_hedge = 0.05

            self.ponzi_ratio = max(0.01, self.ponzi_ratio - ponzi_default)
            transfer = self.speculative_ratio * spec_to_hedge
            self.speculative_ratio -= transfer
            self.hedge_ratio += transfer

            # Asset prices crash
            self.asset_price_index *= 0.95
            # Leverage decreases
            self.leverage_ratio *= 0.98
            # GDP contracts
            gdp_growth = -0.03

            crisis_indicator = 1.0
        else:
            # Normal times: gradual shift to risky financing
            hedge_transfer = self.hedge_ratio * hedge_to_spec
            spec_transfer = self.speculative_ratio * spec_to_ponzi

            self.hedge_ratio -= hedge_transfer
            self.speculative_ratio += hedge_transfer - spec_transfer
            self.ponzi_ratio += spec_transfer

            # Asset prices rise with euphoria
            price_growth = 1.0 + 0.01 * euphoria + np.random.normal(0, 0.005)
            self.asset_price_index *= price_growth

            # Leverage increases with risk-taking
            self.leverage_ratio += 0.05 * risk_tolerance

            crisis_indicator = min(1.0, self.ponzi_ratio / crisis_threshold)

        # Normalize ratios to sum to 1
        total = self.hedge_ratio + self.speculative_ratio + self.ponzi_ratio
        self.hedge_ratio /= total
        self.speculative_ratio /= total
        self.ponzi_ratio /= total

        # Store history
        self.history['time'].append(self.current_period)
        self.history['hedge'].append(self.hedge_ratio)
        self.history['speculative'].append(self.speculative_ratio)
        self.history['ponzi'].append(self.ponzi_ratio)
        self.history['asset_price'].append(self.asset_price_index)
        self.history['leverage'].append(self.leverage_ratio)
        self.history['gdp_growth'].append(gdp_growth)
        self.history['crisis_indicator'].append(crisis_indicator)

        self.current_period += 1

        # Update displays
        self.update_all_visualizations()
        self.update_status_display()

    def update_all_visualizations(self):
        """Update all visualization plots."""
        self.plot_financial_structure()
        self.plot_economic_indicators()

    def plot_financial_structure(self):
        """Plot the evolution of financial structure."""

        self.structure_figure.clear()

        if not self.history['time']:
            return

        # Create subplots
        ax1 = self.structure_figure.add_subplot(211)
        ax2 = self.structure_figure.add_subplot(212)

        time = self.history['time']

        # Stacked area chart of borrower types
        hedge = np.array(self.history['hedge']) * 100
        spec = np.array(self.history['speculative']) * 100
        ponzi = np.array(self.history['ponzi']) * 100

        ax1.fill_between(time, 0, hedge, color='green', alpha=0.6, label='Hedge')
        ax1.fill_between(time, hedge, hedge + spec, color='orange', alpha=0.6, label='Speculative')
        ax1.fill_between(time, hedge + spec, 100, color='red', alpha=0.6, label='Ponzi')

        ax1.set_ylabel('Distribution (%)', fontsize=11)
        ax1.set_title('Evolution of Financial Structure (Minsky Cycle)', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, self.time_periods)
        ax1.set_ylim(0, 100)

        # Add phase annotations
        ponzi_pct = self.history['ponzi'][-1] * 100
        if ponzi_pct > 25:
            ax1.text(0.5, 0.95, 'CRISIS ZONE', transform=ax1.transAxes,
                    fontsize=14, color='red', weight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # Asset price and leverage
        ax2_twin = ax2.twinx()

        asset_prices = self.history['asset_price']
        leverage = self.history['leverage']

        ax2.plot(time, asset_prices, 'b-', linewidth=2, label='Asset Price Index')
        ax2_twin.plot(time, leverage, 'r--', linewidth=2, label='Leverage Ratio')

        ax2.set_xlabel('Time Period', fontsize=11)
        ax2.set_ylabel('Asset Price Index', fontsize=11, color='b')
        ax2_twin.set_ylabel('Leverage Ratio', fontsize=11, color='r')
        ax2.set_title('Asset Prices and Leverage', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        self.structure_figure.tight_layout()
        self.structure_canvas.draw()

    def plot_economic_indicators(self):
        """Plot economic indicators."""

        self.indicators_figure.clear()

        if not self.history['time']:
            return

        ax1 = self.indicators_figure.add_subplot(211)
        ax2 = self.indicators_figure.add_subplot(212)

        time = self.history['time']

        # GDP growth
        gdp_growth = np.array(self.history['gdp_growth']) * 100
        ax1.plot(time, gdp_growth, 'g-', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.fill_between(time, 0, gdp_growth, where=(gdp_growth >= 0),
                        color='green', alpha=0.3, interpolate=True)
        ax1.fill_between(time, 0, gdp_growth, where=(gdp_growth < 0),
                        color='red', alpha=0.3, interpolate=True)
        ax1.set_ylabel('GDP Growth (%)', fontsize=11)
        ax1.set_title('Economic Performance', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Crisis indicator
        crisis_ind = np.array(self.history['crisis_indicator']) * 100
        ax2.plot(time, crisis_ind, 'r-', linewidth=2.5)
        ax2.fill_between(time, 0, crisis_ind, color='red', alpha=0.3)
        ax2.axhline(y=100, color='red', linestyle='--', linewidth=1, label='Crisis Threshold')
        ax2.set_xlabel('Time Period', fontsize=11)
        ax2.set_ylabel('Financial Fragility Index', fontsize=11)
        ax2.set_title('Minsky Crisis Indicator', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 120)
        ax2.legend()

        # Shade crisis periods
        for i, crisis in enumerate(self.history['crisis_indicator']):
            if crisis >= 1.0:
                ax1.axvspan(time[i], time[i]+1, color='red', alpha=0.2)
                ax2.axvspan(time[i], time[i]+1, color='red', alpha=0.2)

        self.indicators_figure.tight_layout()
        self.indicators_canvas.draw()

    def update_status_display(self):
        """Update the status text display."""

        if not self.history['time']:
            return

        hedge_pct = self.hedge_ratio * 100
        spec_pct = self.speculative_ratio * 100
        ponzi_pct = self.ponzi_ratio * 100

        # Determine phase
        if ponzi_pct < 10 and hedge_pct > 70:
            phase = "Post-Crisis Recovery"
            phase_color = "green"
        elif ponzi_pct < 15 and spec_pct < 30:
            phase = "Stable Expansion"
            phase_color = "blue"
        elif ponzi_pct < 25:
            phase = "Euphoric Boom"
            phase_color = "orange"
        else:
            phase = "CRISIS / MINSKY MOMENT"
            phase_color = "red"

        html = f"""
        <h3>Period {self.current_period}</h3>
        <p><b>Phase: <span style="color:{phase_color}">{phase}</span></b></p>
        <hr>
        <p><b>Financial Structure:</b></p>
        <table width="100%">
        <tr><td style="color:green">Hedge:</td><td align="right"><b>{hedge_pct:.1f}%</b></td></tr>
        <tr><td style="color:orange">Speculative:</td><td align="right"><b>{spec_pct:.1f}%</b></td></tr>
        <tr><td style="color:red">Ponzi:</td><td align="right"><b>{ponzi_pct:.1f}%</b></td></tr>
        </table>
        <hr>
        <p><b>Asset Price Index:</b> {self.asset_price_index:.1f}</p>
        <p><b>Leverage Ratio:</b> {self.leverage_ratio:.2f}</p>
        """

        if ponzi_pct > 25:
            html += """<hr><p style="color:red"><b>⚠ WARNING:</b>
            Financial system highly fragile. Crisis likely!</p>"""

        self.status_text.setHtml(html)

    def toggle_simulation(self):
        """Toggle simulation animation."""
        if self.is_animating:
            self.timer.stop()
            self.is_animating = False
            self.play_button.setText("▶ Start Simulation")
        else:
            self.timer.start(100)  # 100ms per step
            self.is_animating = True
            self.play_button.setText("⏸ Pause Simulation")

    def reset_simulation(self):
        """Reset simulation to initial state."""
        if self.is_animating:
            self.toggle_simulation()

        self.current_period = 0
        self.hedge_ratio = 0.80
        self.speculative_ratio = 0.18
        self.ponzi_ratio = 0.02
        self.asset_price_index = 100.0
        self.leverage_ratio = 2.0

        self.history = {
            'time': [],
            'hedge': [],
            'speculative': [],
            'ponzi': [],
            'asset_price': [],
            'leverage': [],
            'gdp_growth': [],
            'crisis_indicator': []
        }

        self.update_all_visualizations()
        self.update_status_display()

    def scenario_post_crisis(self):
        """Apply post-crisis scenario."""
        self.reset_simulation()
        self.optimism_slider.setValue(20)
        self.regulation_slider.setValue(90)
        self.on_slider_change()

    def scenario_deregulation(self):
        """Apply deregulation boom scenario."""
        self.reset_simulation()
        self.optimism_slider.setValue(70)
        self.regulation_slider.setValue(30)
        self.on_slider_change()

    def scenario_bubble(self):
        """Apply asset bubble scenario."""
        self.reset_simulation()
        self.optimism_slider.setValue(90)
        self.regulation_slider.setValue(40)
        self.gdp_spin.setValue(0.05)
        self.on_slider_change()

    def scenario_rate_shock(self):
        """Apply interest rate shock scenario."""
        self.rate_spin.setValue(0.12)

    def export_data(self):
        """Export simulation data."""
        if not self.history['time']:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'minsky_simulation_{timestamp}.csv'

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Period', 'HedgeRatio', 'SpeculativeRatio', 'PonziRatio',
                           'AssetPrice', 'Leverage', 'GDPGrowth', 'CrisisIndicator'])

            for i in range(len(self.history['time'])):
                writer.writerow([
                    self.history['time'][i],
                    self.history['hedge'][i],
                    self.history['speculative'][i],
                    self.history['ponzi'][i],
                    self.history['asset_price'][i],
                    self.history['leverage'][i],
                    self.history['gdp_growth'][i],
                    self.history['crisis_indicator'][i]
                ])

        print(f"Data exported to {filename}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = MinskyInstabilityApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
