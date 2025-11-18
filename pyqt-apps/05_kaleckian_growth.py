"""
Kaleckian Growth Model - PyQt6 Application
==========================================

Economic Theory (Heterodox - Post-Keynesian):
--------------------------------------------
The Kaleckian growth model is a demand-led growth model building on Kalecki's
principle of effective demand. Unlike neoclassical models where growth is
supply-determined (savings, technology), here growth is driven by demand,
particularly investment.

Basic Model:
-----------
Capacity utilization: u = Y/K (output relative to capital stock)
Profit share: π = P/Y (profits as share of output)
Wage share: ω = 1 - π

Investment function:
g = α₀ + α₁·π + α₂·u
where g = I/K (investment rate)

Saving function:
s = s_π·π·u + s_ω·(1-π)·u
where s_π > s_ω (capitalists save more than workers)

Equilibrium: g = s

Key Features:
------------
1. Demand-led growth: Investment determines savings, not vice versa
2. Paradox of thrift: Higher saving can reduce growth
3. Wage-led vs Profit-led growth:
   - Wage-led: Higher wages → more consumption → higher demand → more growth
   - Profit-led: Higher profits → more investment → more growth

The model can exhibit either regime depending on parameters:
- If consumption effect dominates: Wage-led
- If investment effect dominates: Profit-led

Policy Implications:
------------------
- Wage suppression may be counterproductive (if wage-led)
- Redistribution to workers can boost growth (if wage-led)
- Role of aggregate demand management
- Hysteresis and path dependence

Contrast with Neoclassical:
-------------------------
Neoclassical: Growth = f(Savings, Technology, Labor)
Kaleckian: Growth = f(Demand, Distribution, Animal Spirits)

Author: Claude AI
Target: Heterodox macroeconomics students/researchers
References:
- Lavoie, M. (2014) "Post-Keynesian Economics: New Foundations"
- Hein, E. (2014) "Distribution and Growth after Keynes"
"""

import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QDoubleSpinBox, QPushButton,
                             QGroupBox, QTextEdit, QGridLayout, QTabWidget,
                             QComboBox, QSlider)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import csv
from datetime import datetime


class KaleckianGrowthApp(QMainWindow):
    """
    Main application for Kaleckian demand-led growth model.

    Features:
    - Solve for equilibrium growth and utilization
    - Identify wage-led vs profit-led regimes
    - Comparative statics (policy experiments)
    - Phase diagrams
    - Sensitivity analysis
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kaleckian Growth Model")
        self.setGeometry(100, 100, 1400, 800)

        # Model parameters - Investment function: g = α₀ + α₁·π + α₂·u
        self.alpha_0 = 0.02   # Autonomous investment
        self.alpha_1 = 0.15   # Profitability effect on investment
        self.alpha_2 = 0.30   # Accelerator effect (utilization)

        # Saving rates
        self.s_profit = 0.80  # Capitalist saving rate (high)
        self.s_wage = 0.10    # Worker saving rate (low)

        # Exogenous variables
        self.profit_share = 0.35  # Initial profit share

        # Simulation parameters
        self.shock_history = []

        self.init_ui()
        self.calculate_equilibrium()

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
        theory_box = QGroupBox("Kaleckian Model")
        theory_layout = QVBoxLayout()
        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        theory_text.setMaximumHeight(220)
        theory_text.setHtml("""
        <h3>Demand-Led Growth</h3>
        <p><b>Investment:</b> g = α₀ + α₁·π + α₂·u</p>
        <p><b>Saving:</b> s = s<sub>π</sub>·π·u + s<sub>ω</sub>·(1-π)·u</p>
        <p><b>Equilibrium:</b> g = s</p>
        <hr>
        <p><b>Growth Regimes:</b></p>
        <ul style="margin:5px; font-size:9pt">
            <li><b>Wage-led:</b> ↑Wages → ↑Consumption → ↑Demand → ↑Growth</li>
            <li><b>Profit-led:</b> ↑Profits → ↑Investment → ↑Growth</li>
        </ul>
        <p><i>Which regime prevails depends on:</i><br>
        - Consumption propensity gap (s<sub>π</sub> - s<sub>ω</sub>)<br>
        - Investment sensitivity to profits (α₁)<br>
        - Accelerator strength (α₂)</p>
        """)
        theory_layout.addWidget(theory_text)
        theory_box.setLayout(theory_layout)
        layout.addWidget(theory_box)

        # Investment parameters
        inv_box = QGroupBox("Investment Function")
        inv_layout = QGridLayout()

        row = 0
        inv_layout.addWidget(QLabel("Autonomous (α₀):"), row, 0)
        self.alpha0_spin = QDoubleSpinBox()
        self.alpha0_spin.setRange(-0.05, 0.20)
        self.alpha0_spin.setValue(self.alpha_0)
        self.alpha0_spin.setSingleStep(0.01)
        self.alpha0_spin.setDecimals(3)
        self.alpha0_spin.valueChanged.connect(self.on_parameter_change)
        inv_layout.addWidget(self.alpha0_spin, row, 1)
        row += 1

        inv_layout.addWidget(QLabel("Profit sensitivity (α₁):"), row, 0)
        self.alpha1_spin = QDoubleSpinBox()
        self.alpha1_spin.setRange(0, 1.0)
        self.alpha1_spin.setValue(self.alpha_1)
        self.alpha1_spin.setSingleStep(0.05)
        self.alpha1_spin.setDecimals(3)
        self.alpha1_spin.valueChanged.connect(self.on_parameter_change)
        inv_layout.addWidget(self.alpha1_spin, row, 1)
        row += 1

        inv_layout.addWidget(QLabel("Accelerator (α₂):"), row, 0)
        self.alpha2_spin = QDoubleSpinBox()
        self.alpha2_spin.setRange(0, 2.0)
        self.alpha2_spin.setValue(self.alpha_2)
        self.alpha2_spin.setSingleStep(0.05)
        self.alpha2_spin.setDecimals(3)
        self.alpha2_spin.valueChanged.connect(self.on_parameter_change)
        inv_layout.addWidget(self.alpha2_spin, row, 1)
        row += 1

        inv_box.setLayout(inv_layout)
        layout.addWidget(inv_box)

        # Saving parameters
        save_box = QGroupBox("Saving Propensities")
        save_layout = QGridLayout()

        row = 0
        save_layout.addWidget(QLabel("Capitalist saving (s_π):"), row, 0)
        self.sprofit_spin = QDoubleSpinBox()
        self.sprofit_spin.setRange(0, 1.0)
        self.sprofit_spin.setValue(self.s_profit)
        self.sprofit_spin.setSingleStep(0.05)
        self.sprofit_spin.setDecimals(3)
        self.sprofit_spin.valueChanged.connect(self.on_parameter_change)
        save_layout.addWidget(self.sprofit_spin, row, 1)
        row += 1

        save_layout.addWidget(QLabel("Worker saving (s_ω):"), row, 0)
        self.swage_spin = QDoubleSpinBox()
        self.swage_spin.setRange(0, 0.5)
        self.swage_spin.setValue(self.s_wage)
        self.swage_spin.setSingleStep(0.05)
        self.swage_spin.setDecimals(3)
        self.swage_spin.valueChanged.connect(self.on_parameter_change)
        save_layout.addWidget(self.swage_spin, row, 1)
        row += 1

        save_box.setLayout(save_layout)
        layout.addWidget(save_box)

        # Distribution
        dist_box = QGroupBox("Income Distribution")
        dist_layout = QVBoxLayout()

        dist_label = QLabel("Profit Share (π):")
        dist_layout.addWidget(dist_label)

        self.profit_slider = QSlider(Qt.Orientation.Horizontal)
        self.profit_slider.setRange(10, 70)  # 10% to 70%
        self.profit_slider.setValue(int(self.profit_share * 100))
        self.profit_slider.valueChanged.connect(self.on_slider_change)
        dist_layout.addWidget(self.profit_slider)

        self.profit_label = QLabel(f"{self.profit_share:.2%}")
        self.profit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dist_layout.addWidget(self.profit_label)

        dist_box.setLayout(dist_layout)
        layout.addWidget(dist_box)

        # Policy experiments
        policy_box = QGroupBox("Policy Experiments")
        policy_layout = QGridLayout()

        policies = [
            ("Wage Increase", self.policy_wage_increase),
            ("Profit Increase", self.policy_profit_increase),
            ("Stimulus (↑α₀)", self.policy_stimulus),
            ("Austerity (↓α₀)", self.policy_austerity),
            ("Animal Spirits ↑", self.policy_optimism),
            ("Reset", self.reset_parameters)
        ]

        for i, (name, func) in enumerate(policies):
            btn = QPushButton(name)
            btn.clicked.connect(func)
            policy_layout.addWidget(btn, i // 2, i % 2)

        policy_box.setLayout(policy_layout)
        layout.addWidget(policy_box)

        # Results display
        results_box = QGroupBox("Equilibrium Results")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(220)
        results_layout.addWidget(self.results_text)

        results_box.setLayout(results_layout)
        layout.addWidget(results_box)

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

        # Tab 1: Equilibrium diagram
        eq_tab = QWidget()
        eq_layout = QVBoxLayout()
        self.eq_figure = Figure(figsize=(10, 8))
        self.eq_canvas = FigureCanvas(self.eq_figure)
        eq_layout.addWidget(self.eq_canvas)
        eq_tab.setLayout(eq_layout)
        tab_widget.addTab(eq_tab, "Equilibrium Diagram")

        # Tab 2: Regime analysis
        regime_tab = QWidget()
        regime_layout = QVBoxLayout()
        self.regime_figure = Figure(figsize=(10, 8))
        self.regime_canvas = FigureCanvas(self.regime_figure)
        regime_layout.addWidget(self.regime_canvas)
        regime_tab.setLayout(regime_layout)
        tab_widget.addTab(regime_tab, "Regime Analysis")

        return tab_widget

    def on_parameter_change(self):
        """Handle parameter changes."""
        self.alpha_0 = self.alpha0_spin.value()
        self.alpha_1 = self.alpha1_spin.value()
        self.alpha_2 = self.alpha2_spin.value()
        self.s_profit = self.sprofit_spin.value()
        self.s_wage = self.swage_spin.value()
        self.calculate_equilibrium()

    def on_slider_change(self):
        """Handle slider changes."""
        self.profit_share = self.profit_slider.value() / 100.0
        self.profit_label.setText(f"{self.profit_share:.2%}")
        self.calculate_equilibrium()

    def investment_rate(self, u, pi):
        """Calculate investment rate for given utilization and profit share."""
        return self.alpha_0 + self.alpha_1 * pi + self.alpha_2 * u

    def saving_rate(self, u, pi):
        """Calculate saving rate for given utilization and profit share."""
        return self.s_profit * pi * u + self.s_wage * (1 - pi) * u

    def calculate_equilibrium(self):
        """Calculate equilibrium growth rate and capacity utilization."""

        pi = self.profit_share

        # Equilibrium condition: g = s
        # α₀ + α₁·π + α₂·u = s_π·π·u + s_ω·(1-π)·u
        # α₀ + α₁·π = u·[s_π·π + s_ω·(1-π) - α₂]
        # u* = (α₀ + α₁·π) / [s_π·π + s_ω·(1-π) - α₂]

        denominator = self.s_profit * pi + self.s_wage * (1 - pi) - self.alpha_2

        if abs(denominator) < 0.001:
            # Degenerate case
            u_star = None
            g_star = None
            stable = False
        else:
            u_star = (self.alpha_0 + self.alpha_1 * pi) / denominator
            g_star = self.investment_rate(u_star, pi)
            stable = denominator > 0

        # Calculate regime (wage-led vs profit-led)
        regime = self.determine_regime()

        # Update displays
        self.update_results_display(u_star, g_star, stable, regime)
        self.plot_equilibrium(u_star, g_star, stable)
        self.plot_regime_analysis()

    def determine_regime(self):
        """
        Determine if economy is wage-led or profit-led.

        Wage-led if: dg*/dπ < 0 (growth increases when profit share falls)
        Profit-led if: dg*/dπ > 0
        """

        pi = self.profit_share
        epsilon = 0.001

        # Calculate growth at slightly higher profit share
        pi_high = pi + epsilon
        denominator_high = self.s_profit * pi_high + self.s_wage * (1 - pi_high) - self.alpha_2

        if abs(denominator_high) < 0.001:
            return "indeterminate"

        u_high = (self.alpha_0 + self.alpha_1 * pi_high) / denominator_high
        g_high = self.investment_rate(u_high, pi_high)

        # Calculate growth at slightly lower profit share
        pi_low = pi - epsilon
        denominator_low = self.s_profit * pi_low + self.s_wage * (1 - pi_low) - self.alpha_2

        if abs(denominator_low) < 0.001:
            return "indeterminate"

        u_low = (self.alpha_0 + self.alpha_1 * pi_low) / denominator_low
        g_low = self.investment_rate(u_low, pi_low)

        # Derivative approximation
        dg_dpi = (g_high - g_low) / (2 * epsilon)

        if dg_dpi > 0.001:
            return "profit-led"
        elif dg_dpi < -0.001:
            return "wage-led"
        else:
            return "neutral"

    def update_results_display(self, u_star, g_star, stable, regime):
        """Update the results text display."""

        if u_star is None or g_star is None:
            html = """
            <h3>Equilibrium Analysis</h3>
            <p style="color:red"><b>No stable equilibrium exists</b></p>
            <p>Model parameters lead to degenerate case.</p>
            """
        else:
            stability_text = "Stable" if stable else "Unstable"
            stability_color = "green" if stable else "red"

            if regime == "wage-led":
                regime_color = "blue"
                regime_text = "WAGE-LED"
                regime_explain = "Growth increases with higher wage share (lower profit share)"
            elif regime == "profit-led":
                regime_color = "darkred"
                regime_text = "PROFIT-LED"
                regime_explain = "Growth increases with higher profit share"
            else:
                regime_color = "gray"
                regime_text = "NEUTRAL"
                regime_explain = "Growth insensitive to distribution"

            html = f"""
            <h3>Equilibrium Analysis</h3>
            <p><b>Growth Rate (g*):</b> {g_star*100:.2f}%</p>
            <p><b>Capacity Utilization (u*):</b> {u_star*100:.2f}%</p>
            <p><b>Stability:</b> <span style="color:{stability_color}">{stability_text}</span></p>
            <hr>
            <h4>Income Distribution</h4>
            <p><b>Profit Share (π):</b> {self.profit_share*100:.1f}%</p>
            <p><b>Wage Share (ω):</b> {(1-self.profit_share)*100:.1f}%</p>
            <hr>
            <h4>Growth Regime</h4>
            <p><b><span style="color:{regime_color}">{regime_text}</span></b></p>
            <p><i>{regime_explain}</i></p>
            <hr>
            <p style="font-size:9pt"><i>
            Investment: g = {self.alpha_0:.3f} + {self.alpha_1:.3f}·π + {self.alpha_2:.3f}·u<br>
            Saving: s = {self.s_profit:.3f}·π·u + {self.s_wage:.3f}·(1-π)·u
            </i></p>
            """

            if not stable:
                html += """
                <hr>
                <p style="color:red"><b>⚠ Warning:</b> Unstable equilibrium.
                Small deviations will lead to explosive dynamics.</p>
                """

        self.results_text.setHtml(html)

    def plot_equilibrium(self, u_star, g_star, stable):
        """Plot investment and saving schedules."""

        self.eq_figure.clear()
        ax = self.eq_figure.add_subplot(111)

        # Generate utilization range
        u_range = np.linspace(0, 1.5, 100)

        pi = self.profit_share

        # Calculate investment and saving for each utilization level
        g_line = self.alpha_0 + self.alpha_1 * pi + self.alpha_2 * u_range
        s_line = (self.s_profit * pi + self.s_wage * (1 - pi)) * u_range

        # Plot schedules
        ax.plot(u_range, g_line, 'b-', linewidth=2.5, label='Investment (g)')
        ax.plot(u_range, s_line, 'r-', linewidth=2.5, label='Saving (s)')

        # Plot equilibrium point if it exists
        if u_star is not None and g_star is not None and 0 < u_star < 1.5:
            marker = 'o' if stable else 'x'
            color = 'green' if stable else 'red'
            label = f'Equilibrium (u*={u_star:.2f}, g*={g_star:.2f})'

            ax.plot(u_star, g_star, marker, color=color, markersize=15,
                   markeredgewidth=3, label=label, zorder=5)

            # Draw dashed lines to axes
            ax.axvline(x=u_star, color='gray', linestyle=':', alpha=0.5)
            ax.axhline(y=g_star, color='gray', linestyle=':', alpha=0.5)

        # Formatting
        ax.set_xlabel('Capacity Utilization (u)', fontsize=12)
        ax.set_ylabel('Growth Rate / Saving Rate', fontsize=12)
        ax.set_title(f'Kaleckian Growth Model (π = {pi:.2%})', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        self.eq_figure.tight_layout()
        self.eq_canvas.draw()

    def plot_regime_analysis(self):
        """Plot how growth varies with profit share (regime analysis)."""

        self.regime_figure.clear()

        # Create subplots
        ax1 = self.regime_figure.add_subplot(211)
        ax2 = self.regime_figure.add_subplot(212)

        # Range of profit shares
        pi_range = np.linspace(0.15, 0.65, 50)

        g_values = []
        u_values = []

        for pi in pi_range:
            denominator = self.s_profit * pi + self.s_wage * (1 - pi) - self.alpha_2
            if abs(denominator) > 0.001:
                u = (self.alpha_0 + self.alpha_1 * pi) / denominator
                g = self.alpha_0 + self.alpha_1 * pi + self.alpha_2 * u
                u_values.append(u)
                g_values.append(g)
            else:
                u_values.append(np.nan)
                g_values.append(np.nan)

        # Plot growth vs profit share
        ax1.plot(pi_range * 100, np.array(g_values) * 100, 'b-', linewidth=2.5)
        ax1.axvline(x=self.profit_share * 100, color='red', linestyle='--',
                   linewidth=2, label=f'Current π = {self.profit_share:.1%}')

        # Determine regime and add shading
        current_regime = self.determine_regime()
        if current_regime == "wage-led":
            ax1.text(0.05, 0.95, 'WAGE-LED REGIME', transform=ax1.transAxes,
                    fontsize=12, color='blue', weight='bold', va='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        elif current_regime == "profit-led":
            ax1.text(0.05, 0.95, 'PROFIT-LED REGIME', transform=ax1.transAxes,
                    fontsize=12, color='darkred', weight='bold', va='top',
                    bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))

        ax1.set_xlabel('Profit Share (%)', fontsize=11)
        ax1.set_ylabel('Growth Rate (%)', fontsize=11)
        ax1.set_title('Growth vs Distribution: Identifying the Regime', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # Plot utilization vs profit share
        ax2.plot(pi_range * 100, np.array(u_values) * 100, 'g-', linewidth=2.5)
        ax2.axvline(x=self.profit_share * 100, color='red', linestyle='--',
                   linewidth=2, label=f'Current π = {self.profit_share:.1%}')

        ax2.set_xlabel('Profit Share (%)', fontsize=11)
        ax2.set_ylabel('Capacity Utilization (%)', fontsize=11)
        ax2.set_title('Utilization vs Distribution', fontsize=12, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        self.regime_figure.tight_layout()
        self.regime_canvas.draw()

    def policy_wage_increase(self):
        """Simulate wage increase (decrease profit share)."""
        new_share = max(0.10, self.profit_share - 0.05)
        self.profit_slider.setValue(int(new_share * 100))

    def policy_profit_increase(self):
        """Simulate profit increase (increase profit share)."""
        new_share = min(0.70, self.profit_share + 0.05)
        self.profit_slider.setValue(int(new_share * 100))

    def policy_stimulus(self):
        """Increase autonomous investment."""
        self.alpha0_spin.setValue(self.alpha_0 + 0.01)

    def policy_austerity(self):
        """Decrease autonomous investment."""
        self.alpha0_spin.setValue(self.alpha_0 - 0.01)

    def policy_optimism(self):
        """Increase animal spirits (all investment parameters)."""
        self.alpha0_spin.setValue(min(0.20, self.alpha_0 + 0.01))
        self.alpha1_spin.setValue(min(1.0, self.alpha_1 + 0.05))
        self.alpha2_spin.setValue(min(2.0, self.alpha_2 + 0.05))

    def reset_parameters(self):
        """Reset to default parameters."""
        self.alpha0_spin.setValue(0.02)
        self.alpha1_spin.setValue(0.15)
        self.alpha2_spin.setValue(0.30)
        self.sprofit_spin.setValue(0.80)
        self.swage_spin.setValue(0.10)
        self.profit_slider.setValue(35)

    def export_data(self):
        """Export regime analysis data."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'kaleckian_growth_{timestamp}.csv'

        pi_range = np.linspace(0.10, 0.70, 100)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ProfitShare', 'WageShare', 'GrowthRate',
                           'CapacityUtilization'])

            for pi in pi_range:
                denominator = self.s_profit * pi + self.s_wage * (1 - pi) - self.alpha_2
                if abs(denominator) > 0.001:
                    u = (self.alpha_0 + self.alpha_1 * pi) / denominator
                    g = self.alpha_0 + self.alpha_1 * pi + self.alpha_2 * u

                    writer.writerow([pi, 1-pi, g, u])

        print(f"Data exported to {filename}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = KaleckianGrowthApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
