"""
Kalecki Profit Equation Simulator - PyQt6 Application
=====================================================

Economic Theory (Heterodox - Post-Keynesian):
--------------------------------------------
Michał Kalecki (1899-1970) was a Polish economist who developed a class-based
macroeconomic theory independently of Keynes. His famous profit equation shows
that profits are determined by capitalist spending decisions.

Kalecki's Profit Equation (simplified):
P = I + (G - T) + (X - M) + C_p - S_w

Where:
P = Gross profits
I = Investment (capitalist spending on capital goods)
G = Government spending
T = Taxes
X = Exports
M = Imports
C_p = Capitalist consumption
S_w = Worker savings

Key Insight:
"Workers spend what they earn, capitalists earn what they spend"

This means:
1. Capitalists as a class determine their own profits through investment and consumption
2. Government deficits (G - T > 0) increase profits
3. Trade surpluses (X - M > 0) increase profits
4. Worker savings reduce profits (consumption demand)

Implications:
- Profit-led vs wage-led growth debate
- Role of fiscal policy in capitalist economies
- Financialization and profit realization
- Class struggle and income distribution

Author: Claude AI
Target: Heterodox economics students/researchers
References: Kalecki (1954), "Theory of Economic Dynamics"
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTextEdit, QGridLayout, QTabWidget,
                             QSlider, QComboBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import csv
from datetime import datetime


class KaleckiProfitApp(QMainWindow):
    """
    Main application for Kalecki Profit Equation analysis.

    Features:
    - Real-time profit calculation
    - Component breakdown visualization
    - Time-series simulation
    - Scenario analysis (austerity, investment boom, etc.)
    - Data export for research
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kalecki Profit Equation Simulator")
        self.setGeometry(100, 100, 1400, 800)

        # Initialize parameters (all in billions of currency units)
        self.investment = 200.0
        self.govt_spending = 300.0
        self.taxes = 250.0
        self.exports = 150.0
        self.imports = 180.0
        self.capitalist_consumption = 100.0
        self.worker_savings = 50.0

        # For time series simulation
        self.time_periods = 20
        self.current_period = 0
        self.history = []

        self.init_ui()
        self.calculate_and_display()

    def init_ui(self):
        """Initialize the user interface."""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel: Tabs for different visualizations
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, stretch=2)

    def create_control_panel(self):
        """Create the parameter control panel."""

        panel = QGroupBox("Kalecki Model Parameters")
        layout = QVBoxLayout()

        # Theory section
        theory_box = QGroupBox("Theory")
        theory_layout = QVBoxLayout()
        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        theory_text.setMaximumHeight(200)
        theory_text.setHtml("""
        <h3>Kalecki's Profit Equation</h3>
        <p><b>P = I + (G - T) + (X - M) + C<sub>p</sub> - S<sub>w</sub></b></p>
        <p><i>"Workers spend what they earn,<br>
        capitalists earn what they spend"</i></p>
        <p><b>Key Insights:</b></p>
        <ul>
            <li>Profits determined by capitalist spending (I, C<sub>p</sub>)</li>
            <li>Government deficits boost profits</li>
            <li>Trade surpluses boost profits</li>
            <li>Worker savings reduce profit realization</li>
        </ul>
        """)
        theory_layout.addWidget(theory_text)
        theory_box.setLayout(theory_layout)
        layout.addWidget(theory_box)

        # Parameters section
        params_box = QGroupBox("Components (billions $)")
        params_layout = QGridLayout()

        row = 0

        # Investment
        params_layout.addWidget(QLabel("Investment (I):"), row, 0)
        self.i_spin = QDoubleSpinBox()
        self.i_spin.setRange(0, 1000)
        self.i_spin.setValue(self.investment)
        self.i_spin.setSingleStep(10)
        self.i_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.i_spin, row, 1)
        row += 1

        # Government spending
        params_layout.addWidget(QLabel("Gov't Spending (G):"), row, 0)
        self.g_spin = QDoubleSpinBox()
        self.g_spin.setRange(0, 1000)
        self.g_spin.setValue(self.govt_spending)
        self.g_spin.setSingleStep(10)
        self.g_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.g_spin, row, 1)
        row += 1

        # Taxes
        params_layout.addWidget(QLabel("Taxes (T):"), row, 0)
        self.t_spin = QDoubleSpinBox()
        self.t_spin.setRange(0, 1000)
        self.t_spin.setValue(self.taxes)
        self.t_spin.setSingleStep(10)
        self.t_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.t_spin, row, 1)
        row += 1

        # Exports
        params_layout.addWidget(QLabel("Exports (X):"), row, 0)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(0, 1000)
        self.x_spin.setValue(self.exports)
        self.x_spin.setSingleStep(10)
        self.x_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.x_spin, row, 1)
        row += 1

        # Imports
        params_layout.addWidget(QLabel("Imports (M):"), row, 0)
        self.m_spin = QDoubleSpinBox()
        self.m_spin.setRange(0, 1000)
        self.m_spin.setValue(self.imports)
        self.m_spin.setSingleStep(10)
        self.m_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.m_spin, row, 1)
        row += 1

        # Capitalist consumption
        params_layout.addWidget(QLabel("Capitalist Consump. (Cₚ):"), row, 0)
        self.cp_spin = QDoubleSpinBox()
        self.cp_spin.setRange(0, 500)
        self.cp_spin.setValue(self.capitalist_consumption)
        self.cp_spin.setSingleStep(10)
        self.cp_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.cp_spin, row, 1)
        row += 1

        # Worker savings
        params_layout.addWidget(QLabel("Worker Savings (Sᵥᵥ):"), row, 0)
        self.sw_spin = QDoubleSpinBox()
        self.sw_spin.setRange(0, 500)
        self.sw_spin.setValue(self.worker_savings)
        self.sw_spin.setSingleStep(10)
        self.sw_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.sw_spin, row, 1)
        row += 1

        params_box.setLayout(params_layout)
        layout.addWidget(params_box)

        # Scenario presets
        scenario_box = QGroupBox("Scenario Analysis")
        scenario_layout = QVBoxLayout()

        scenario_label = QLabel("Quick Scenarios:")
        scenario_layout.addWidget(scenario_label)

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems([
            "Select scenario...",
            "Austerity (cut G, raise T)",
            "Fiscal Stimulus (raise G)",
            "Investment Boom",
            "Trade Surplus Strategy",
            "Wage Suppression (raise Sᵥᵥ)",
            "Financialization (cut I, raise Cₚ)"
        ])
        self.scenario_combo.currentTextChanged.connect(self.apply_scenario)
        scenario_layout.addWidget(self.scenario_combo)

        scenario_box.setLayout(scenario_layout)
        layout.addWidget(scenario_box)

        # Results display
        results_box = QGroupBox("Current Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(250)
        results_layout.addWidget(self.results_text)

        results_box.setLayout(results_layout)
        layout.addWidget(results_box)

        # Export button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        layout.addWidget(export_btn)

        # Add stretch to push everything to top
        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def create_visualization_panel(self):
        """Create tabbed visualization panel."""

        tab_widget = QTabWidget()

        # Tab 1: Component breakdown
        breakdown_tab = QWidget()
        breakdown_layout = QVBoxLayout()
        self.breakdown_figure = Figure(figsize=(10, 6))
        self.breakdown_canvas = FigureCanvas(self.breakdown_figure)
        breakdown_layout.addWidget(self.breakdown_canvas)
        breakdown_tab.setLayout(breakdown_layout)
        tab_widget.addTab(breakdown_tab, "Profit Components")

        # Tab 2: Time series (if simulated)
        timeseries_tab = QWidget()
        timeseries_layout = QVBoxLayout()
        self.timeseries_figure = Figure(figsize=(10, 6))
        self.timeseries_canvas = FigureCanvas(self.timeseries_figure)
        timeseries_layout.addWidget(self.timeseries_canvas)
        timeseries_tab.setLayout(timeseries_layout)
        tab_widget.addTab(timeseries_tab, "Time Series")

        return tab_widget

    def on_parameter_change(self):
        """Handle parameter changes."""
        self.investment = self.i_spin.value()
        self.govt_spending = self.g_spin.value()
        self.taxes = self.t_spin.value()
        self.exports = self.x_spin.value()
        self.imports = self.m_spin.value()
        self.capitalist_consumption = self.cp_spin.value()
        self.worker_savings = self.sw_spin.value()

        self.calculate_and_display()

    def calculate_profits(self):
        """
        Calculate profits using Kalecki's equation.

        Returns:
            dict: Profit components and analysis
        """
        # Component calculations
        fiscal_balance = self.govt_spending - self.taxes  # Deficit = positive for profits
        trade_balance = self.exports - self.imports        # Surplus = positive for profits

        # Total profits
        profits = (self.investment +
                  fiscal_balance +
                  trade_balance +
                  self.capitalist_consumption -
                  self.worker_savings)

        # Calculate contribution percentages
        total_positive = (self.investment +
                         max(0, fiscal_balance) +
                         max(0, trade_balance) +
                         self.capitalist_consumption)

        components = {
            'profits': profits,
            'investment': self.investment,
            'fiscal_balance': fiscal_balance,
            'trade_balance': trade_balance,
            'capitalist_consumption': self.capitalist_consumption,
            'worker_savings': self.worker_savings,
            'govt_spending': self.govt_spending,
            'taxes': self.taxes,
            'exports': self.exports,
            'imports': self.imports
        }

        return components

    def calculate_and_display(self):
        """Calculate and update all displays."""

        results = self.calculate_profits()

        # Store in history for time series
        self.history.append(results.copy())
        if len(self.history) > 50:
            self.history.pop(0)

        # Update displays
        self.update_results_text(results)
        self.plot_breakdown(results)
        self.plot_timeseries()

    def update_results_text(self, results):
        """Update the results text display."""

        fiscal_status = "Deficit" if results['fiscal_balance'] > 0 else "Surplus"
        trade_status = "Surplus" if results['trade_balance'] > 0 else "Deficit"

        html = f"""
        <h3>Profit Analysis</h3>
        <p><b>Total Profits: ${results['profits']:.2f}B</b></p>
        <hr>
        <h4>Components:</h4>
        <table width="100%">
        <tr><td>Investment:</td><td align="right"><b>+${results['investment']:.2f}B</b></td></tr>
        <tr><td>Fiscal Balance (G-T):</td><td align="right">
            <b>{"+" if results['fiscal_balance'] >= 0 else ""}{results['fiscal_balance']:.2f}B</b>
            <i>({fiscal_status})</i></td></tr>
        <tr><td style="padding-left:20px">Gov't Spending:</td><td align="right">${results['govt_spending']:.2f}B</td></tr>
        <tr><td style="padding-left:20px">Taxes:</td><td align="right">-${results['taxes']:.2f}B</td></tr>
        <tr><td>Trade Balance (X-M):</td><td align="right">
            <b>{"+" if results['trade_balance'] >= 0 else ""}{results['trade_balance']:.2f}B</b>
            <i>({trade_status})</i></td></tr>
        <tr><td style="padding-left:20px">Exports:</td><td align="right">${results['exports']:.2f}B</td></tr>
        <tr><td style="padding-left:20px">Imports:</td><td align="right">-${results['imports']:.2f}B</td></tr>
        <tr><td>Capitalist Consumption:</td><td align="right"><b>+${results['capitalist_consumption']:.2f}B</b></td></tr>
        <tr><td>Worker Savings:</td><td align="right"><b>-${results['worker_savings']:.2f}B</b></td></tr>
        </table>
        <hr>
        <p><i>Interpretation: </i>"""

        if results['investment'] > results['profits'] * 0.5:
            html += "Investment is the primary source of profit realization. "
        if results['fiscal_balance'] > 0:
            html += f"Government deficit of ${results['fiscal_balance']:.1f}B supports profits. "
        elif results['fiscal_balance'] < 0:
            html += f"Government surplus of ${-results['fiscal_balance']:.1f}B reduces profits. "
        if results['trade_balance'] > 0:
            html += "Trade surplus contributes to profits."

        html += "</p>"

        self.results_text.setHtml(html)

    def plot_breakdown(self, results):
        """Plot profit component breakdown."""

        self.breakdown_figure.clear()

        # Create subplots: bar chart and waterfall
        ax1 = self.breakdown_figure.add_subplot(121)
        ax2 = self.breakdown_figure.add_subplot(122)

        # Bar chart of components
        components = ['Investment', 'Fiscal\nBalance', 'Trade\nBalance',
                     'Capitalist\nConsump.', 'Worker\nSavings']
        values = [
            results['investment'],
            results['fiscal_balance'],
            results['trade_balance'],
            results['capitalist_consumption'],
            -results['worker_savings']  # Negative because it reduces profits
        ]

        colors = ['green' if v > 0 else 'red' for v in values]

        ax1.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Contribution to Profits ($B)', fontsize=11)
        ax1.set_title('Component Contributions', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=0, labelsize=9)

        # Waterfall chart showing cumulative effect
        cumulative = 0
        bottoms = []
        heights = []
        labels = []

        for comp, val in zip(components, values):
            if val >= 0:
                bottoms.append(cumulative)
                heights.append(val)
            else:
                bottoms.append(cumulative + val)
                heights.append(-val)
            cumulative += val
            labels.append(f'{val:+.0f}')

        # Plot waterfall
        for i, (bottom, height, val) in enumerate(zip(bottoms, heights, values)):
            color = 'green' if val > 0 else 'red'
            ax2.bar(i, height, bottom=bottom, color=color, alpha=0.7, edgecolor='black')
            # Connect bars
            if i < len(bottoms) - 1:
                next_bottom = bottoms[i+1] if values[i+1] > 0 else bottoms[i+1] + heights[i+1]
                ax2.plot([i+0.4, i+0.6], [bottom+height, next_bottom],
                        'k--', linewidth=1, alpha=0.5)

        # Final profit bar
        ax2.bar(len(components), results['profits'], color='blue', alpha=0.7,
               edgecolor='black', linewidth=2)

        ax2.set_xticks(range(len(components) + 1))
        ax2.set_xticklabels(components + ['Total\nProfits'], fontsize=9)
        ax2.set_ylabel('Profits ($B)', fontsize=11)
        ax2.set_title('Waterfall: Building Up to Total Profits', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)

        self.breakdown_figure.tight_layout()
        self.breakdown_canvas.draw()

    def plot_timeseries(self):
        """Plot time series of profit evolution."""

        if len(self.history) < 2:
            return

        self.timeseries_figure.clear()
        ax = self.timeseries_figure.add_subplot(111)

        periods = range(len(self.history))
        profits = [h['profits'] for h in self.history]
        investment = [h['investment'] for h in self.history]
        fiscal_balance = [h['fiscal_balance'] for h in self.history]

        ax.plot(periods, profits, 'b-', linewidth=2, label='Total Profits', marker='o')
        ax.plot(periods, investment, 'g--', linewidth=1.5, label='Investment', alpha=0.7)
        ax.plot(periods, fiscal_balance, 'r--', linewidth=1.5, label='Fiscal Balance', alpha=0.7)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Period', fontsize=11)
        ax.set_ylabel('Value ($B)', fontsize=11)
        ax.set_title('Evolution of Profits and Key Components', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        self.timeseries_figure.tight_layout()
        self.timeseries_canvas.draw()

    def apply_scenario(self, scenario_name):
        """Apply predefined scenario."""

        if scenario_name == "Select scenario...":
            return

        # Store baseline
        baseline = {
            'I': self.investment,
            'G': self.govt_spending,
            'T': self.taxes,
            'X': self.exports,
            'M': self.imports,
            'Cp': self.capitalist_consumption,
            'Sw': self.worker_savings
        }

        if scenario_name == "Austerity (cut G, raise T)":
            self.g_spin.setValue(baseline['G'] * 0.7)
            self.t_spin.setValue(baseline['T'] * 1.2)

        elif scenario_name == "Fiscal Stimulus (raise G)":
            self.g_spin.setValue(baseline['G'] * 1.5)

        elif scenario_name == "Investment Boom":
            self.i_spin.setValue(baseline['I'] * 1.5)

        elif scenario_name == "Trade Surplus Strategy":
            self.x_spin.setValue(baseline['X'] * 1.3)
            self.m_spin.setValue(baseline['M'] * 0.8)

        elif scenario_name == "Wage Suppression (raise Sᵥᵥ)":
            self.sw_spin.setValue(baseline['Sw'] * 1.5)

        elif scenario_name == "Financialization (cut I, raise Cₚ)":
            self.i_spin.setValue(baseline['I'] * 0.7)
            self.cp_spin.setValue(baseline['Cp'] * 1.8)

        # Reset combo box
        self.scenario_combo.setCurrentIndex(0)

    def export_data(self):
        """Export current state and history to CSV."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'kalecki_profits_{timestamp}.csv'

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(['Period', 'Profits', 'Investment', 'GovtSpending', 'Taxes',
                           'FiscalBalance', 'Exports', 'Imports', 'TradeBalance',
                           'CapitalistConsumption', 'WorkerSavings'])

            # Write history
            for i, record in enumerate(self.history):
                writer.writerow([
                    i,
                    record['profits'],
                    record['investment'],
                    record['govt_spending'],
                    record['taxes'],
                    record['fiscal_balance'],
                    record['exports'],
                    record['imports'],
                    record['trade_balance'],
                    record['capitalist_consumption'],
                    record['worker_savings']
                ])

        print(f"Data exported to {filename}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = KaleckiProfitApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
