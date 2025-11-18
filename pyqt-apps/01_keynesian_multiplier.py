"""
Keynesian Multiplier Calculator - PyQt6 Application
===================================================

Economic Theory:
--------------
The Keynesian multiplier demonstrates how an initial change in autonomous spending
(e.g., government spending, investment) leads to a larger final change in national income.

Basic Model:
Y = C + I + G  (Income = Consumption + Investment + Government Spending)
C = a + b*Y    (Consumption function: autonomous consumption + marginal propensity to consume)

Solving for Y:
Y = a + b*Y + I + G
Y - b*Y = a + I + G
Y(1 - b) = a + I + G
Y = (a + I + G) / (1 - b)

The multiplier is k = 1/(1 - b) = 1/(1 - MPC) = 1/MPS

Author: Claude AI
Target: Economics students learning Keynesian fundamentals
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTextEdit, QGridLayout)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class KeynesianMultiplierApp(QMainWindow):
    """
    Main application window for Keynesian Multiplier Calculator.

    Architecture:
    - QMainWindow: Top-level window container
    - Central widget contains main layout
    - Left panel: Parameter controls
    - Right panel: Visualization and results
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keynesian Multiplier Calculator")
        self.setGeometry(100, 100, 1200, 700)

        # Initialize parameters
        self.autonomous_consumption = 100.0  # 'a' in C = a + bY
        self.mpc = 0.75                      # Marginal propensity to consume
        self.investment = 50.0               # Autonomous investment
        self.government = 50.0               # Government spending

        self.init_ui()
        self.calculate_and_plot()

    def init_ui(self):
        """Initialize the user interface components."""

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Right panel: Visualization and results
        right_panel = self.create_visualization_panel()
        main_layout.addWidget(right_panel, stretch=2)

    def create_control_panel(self):
        """Create the parameter control panel."""

        panel = QGroupBox("Model Parameters")
        layout = QGridLayout()

        # Theory explanation
        theory_label = QLabel("Keynesian Cross Model")
        theory_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(theory_label, 0, 0, 1, 2)

        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        theory_text.setMaximumHeight(150)
        theory_text.setHtml("""
        <p><b>Y = C + I + G</b></p>
        <p><b>C = a + b·Y</b></p>
        <p>Where:</p>
        <ul>
            <li>Y = National income</li>
            <li>C = Consumption</li>
            <li>a = Autonomous consumption</li>
            <li>b = Marginal propensity to consume (MPC)</li>
            <li>I = Investment</li>
            <li>G = Government spending</li>
        </ul>
        <p><b>Multiplier k = 1/(1-MPC)</b></p>
        """)
        layout.addWidget(theory_text, 1, 0, 1, 2)

        # Parameter controls
        row = 2

        # Autonomous consumption
        layout.addWidget(QLabel("Autonomous Consumption (a):"), row, 0)
        self.a_spinbox = QDoubleSpinBox()
        self.a_spinbox.setRange(0, 1000)
        self.a_spinbox.setValue(self.autonomous_consumption)
        self.a_spinbox.setSingleStep(10)
        self.a_spinbox.valueChanged.connect(self.on_parameter_change)
        layout.addWidget(self.a_spinbox, row, 1)
        row += 1

        # Marginal propensity to consume
        layout.addWidget(QLabel("MPC (b):"), row, 0)
        self.mpc_spinbox = QDoubleSpinBox()
        self.mpc_spinbox.setRange(0, 0.99)
        self.mpc_spinbox.setValue(self.mpc)
        self.mpc_spinbox.setSingleStep(0.05)
        self.mpc_spinbox.setDecimals(3)
        self.mpc_spinbox.valueChanged.connect(self.on_parameter_change)
        layout.addWidget(self.mpc_spinbox, row, 1)
        row += 1

        # Investment
        layout.addWidget(QLabel("Investment (I):"), row, 0)
        self.i_spinbox = QDoubleSpinBox()
        self.i_spinbox.setRange(0, 1000)
        self.i_spinbox.setValue(self.investment)
        self.i_spinbox.setSingleStep(10)
        self.i_spinbox.valueChanged.connect(self.on_parameter_change)
        layout.addWidget(self.i_spinbox, row, 1)
        row += 1

        # Government spending
        layout.addWidget(QLabel("Government Spending (G):"), row, 0)
        self.g_spinbox = QDoubleSpinBox()
        self.g_spinbox.setRange(0, 1000)
        self.g_spinbox.setValue(self.government)
        self.g_spinbox.setSingleStep(10)
        self.g_spinbox.valueChanged.connect(self.on_parameter_change)
        layout.addWidget(self.g_spinbox, row, 1)
        row += 1

        # Results display
        results_label = QLabel("Results")
        results_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(results_label, row, 0, 1, 2)
        row += 1

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        layout.addWidget(self.results_text, row, 0, 1, 2)
        row += 1

        # Export button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        layout.addWidget(export_btn, row, 0, 1, 2)

        panel.setLayout(layout)
        return panel

    def create_visualization_panel(self):
        """Create the visualization panel with matplotlib."""

        panel = QGroupBox("Keynesian Cross Diagram")
        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        panel.setLayout(layout)
        return panel

    def on_parameter_change(self):
        """Handle parameter changes and recalculate."""
        self.autonomous_consumption = self.a_spinbox.value()
        self.mpc = self.mpc_spinbox.value()
        self.investment = self.i_spinbox.value()
        self.government = self.g_spinbox.value()
        self.calculate_and_plot()

    def calculate_equilibrium(self):
        """
        Calculate equilibrium income and related values.

        Returns:
            dict: Contains equilibrium values and components
        """
        # Multiplier k = 1/(1-b)
        multiplier = 1 / (1 - self.mpc)

        # Autonomous spending A = a + I + G
        autonomous_spending = self.autonomous_consumption + self.investment + self.government

        # Equilibrium income Y* = k * A
        equilibrium_income = multiplier * autonomous_spending

        # Equilibrium consumption C* = a + b*Y*
        equilibrium_consumption = self.autonomous_consumption + self.mpc * equilibrium_income

        # MPS (Marginal propensity to save)
        mps = 1 - self.mpc

        return {
            'multiplier': multiplier,
            'autonomous_spending': autonomous_spending,
            'equilibrium_income': equilibrium_income,
            'equilibrium_consumption': equilibrium_consumption,
            'mpc': self.mpc,
            'mps': mps
        }

    def calculate_and_plot(self):
        """Calculate equilibrium and update visualization."""

        results = self.calculate_equilibrium()

        # Update results text
        self.update_results_display(results)

        # Update plot
        self.plot_keynesian_cross(results)

    def update_results_display(self, results):
        """Update the results text display."""

        html = f"""
        <h3>Equilibrium Analysis</h3>
        <p><b>Multiplier (k):</b> {results['multiplier']:.3f}</p>
        <p><b>MPC:</b> {results['mpc']:.3f}</p>
        <p><b>MPS:</b> {results['mps']:.3f}</p>
        <hr>
        <p><b>Autonomous Spending (A):</b> ${results['autonomous_spending']:.2f}</p>
        <p><b>Equilibrium Income (Y*):</b> ${results['equilibrium_income']:.2f}</p>
        <p><b>Equilibrium Consumption (C*):</b> ${results['equilibrium_consumption']:.2f}</p>
        <hr>
        <p><i>Interpretation:</i> A $1 increase in autonomous spending
        will increase equilibrium income by ${results['multiplier']:.2f}</p>
        """

        self.results_text.setHtml(html)

    def plot_keynesian_cross(self, results):
        """
        Plot the Keynesian cross diagram.

        The diagram shows:
        - 45-degree line (Y = AE, where aggregate expenditure equals income)
        - Aggregate expenditure line (AE = C + I + G)
        - Equilibrium point where they intersect
        """

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Generate income range
        max_y = results['equilibrium_income'] * 1.5
        y_range = np.linspace(0, max_y, 100)

        # Calculate aggregate expenditure for each income level
        # AE = C + I + G = (a + b*Y) + I + G
        ae_line = (self.autonomous_consumption + self.mpc * y_range +
                   self.investment + self.government)

        # 45-degree line (Y = AE)
        ax.plot(y_range, y_range, 'k--', label='45° line (Y = AE)', linewidth=2)

        # Aggregate expenditure line
        ax.plot(y_range, ae_line, 'b-', label='AE = C + I + G', linewidth=2)

        # Equilibrium point
        ax.plot(results['equilibrium_income'], results['equilibrium_income'],
                'ro', markersize=10, label=f'Equilibrium (Y* = {results["equilibrium_income"]:.1f})')

        # Draw dashed lines to axes
        ax.axvline(x=results['equilibrium_income'], color='r', linestyle=':', alpha=0.5)
        ax.axhline(y=results['equilibrium_income'], color='r', linestyle=':', alpha=0.5)

        # Labels and formatting
        ax.set_xlabel('Income (Y)', fontsize=12)
        ax.set_ylabel('Aggregate Expenditure (AE)', fontsize=12)
        ax.set_title('Keynesian Cross Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_y)
        ax.set_ylim(0, max_y)

        self.canvas.draw()

    def export_data(self):
        """Export calculation results to CSV file."""

        results = self.calculate_equilibrium()

        # Generate data for various income levels
        y_range = np.linspace(0, results['equilibrium_income'] * 2, 50)
        consumption = self.autonomous_consumption + self.mpc * y_range
        ae = consumption + self.investment + self.government

        # Save to CSV
        filename = 'keynesian_multiplier_data.csv'
        with open(filename, 'w') as f:
            f.write('Income,Consumption,Investment,Government,AggregateExpenditure\n')
            for i, y in enumerate(y_range):
                f.write(f'{y:.2f},{consumption[i]:.2f},{self.investment:.2f},'
                       f'{self.government:.2f},{ae[i]:.2f}\n')

        print(f"Data exported to {filename}")


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = KeynesianMultiplierApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
