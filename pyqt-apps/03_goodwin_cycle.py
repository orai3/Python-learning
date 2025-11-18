"""
Goodwin Growth Cycle Model - PyQt6 Application
==============================================

Economic Theory (Heterodox - Marxian/Post-Keynesian):
----------------------------------------------------
Richard Goodwin (1967) developed a model of cyclical growth based on Marx's
analysis of the reserve army of labor and class conflict over income distribution.

The model uses Lotka-Volterra predator-prey dynamics to show how:
1. High employment → Workers gain bargaining power → Wages rise
2. Rising wages → Profit squeeze → Investment falls → Employment falls
3. Low employment → Wages stagnate → Profits recover → Investment rises
4. Cycle repeats

Mathematical Model:
-----------------
dv/dt = v * (ρ - (α + β) - ω/σ)     [Employment rate dynamics]
du/dt = u * (φ(v) - α)               [Wage share dynamics]

Where:
v = Employment rate (1 - unemployment rate)
u = Wage share (labor's share of income)
ρ = Growth rate of labor productivity
α = Depreciation rate
β = Population growth rate
σ = Capital-output ratio
φ(v) = Rate of wage increase (Phillips curve relationship)

Simplified form often used:
dv/dt = v * (a - b*u)
du/dt = u * (c*v - d)

This creates a closed cycle in (v, u) space, representing perpetual class struggle.

Key Insights:
- Capitalism is inherently cyclical due to class conflict
- No stable equilibrium exists (unlike neoclassical models)
- Distribution and growth are inseparable
- Reserve army of labor is functional for capitalism

Author: Claude AI
Target: Heterodox economics students/researchers
References: Goodwin, R.M. (1967) "A Growth Cycle", in Feinstein (ed.) Socialism, Capitalism and Economic Growth
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
from scipy.integrate import odeint
import csv
from datetime import datetime


class GoodwinCycleApp(QMainWindow):
    """
    Main application for Goodwin Growth Cycle simulation.

    Features:
    - Real-time simulation of the Goodwin model
    - Phase diagram showing cycles in (employment, wage share) space
    - Time series of both variables
    - Animated evolution
    - Parameter sensitivity analysis
    - Data export
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goodwin Growth Cycle Model")
        self.setGeometry(100, 100, 1400, 800)

        # Model parameters (simplified Goodwin model)
        self.a = 0.05   # Effect of wage share on employment growth (negative)
        self.b = 0.10   # Sensitivity to wage share
        self.c = 0.15   # Effect of employment on wage growth
        self.d = 0.03   # Natural rate of wage decline

        # Initial conditions
        self.v0 = 0.95  # Initial employment rate (95%)
        self.u0 = 0.65  # Initial wage share (65%)

        # Simulation parameters
        self.t_max = 200  # Time periods
        self.dt = 0.1     # Time step
        self.is_animating = False
        self.animation_step = 0

        # Results storage
        self.t = None
        self.v = None
        self.u = None

        self.init_ui()
        self.run_simulation()

        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.animation_step_forward)

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
        theory_box = QGroupBox("Economic Theory")
        theory_layout = QVBoxLayout()
        theory_text = QTextEdit()
        theory_text.setReadOnly(True)
        theory_text.setMaximumHeight(250)
        theory_text.setHtml("""
        <h3>Goodwin's Growth Cycle</h3>
        <p><b>Class Struggle as Predator-Prey Dynamics</b></p>
        <p><i>dv/dt = v(a - b·u)</i><br>
        <i>du/dt = u(c·v - d)</i></p>
        <p>where:<br>
        v = Employment rate<br>
        u = Wage share</p>
        <p><b>The Cycle:</b></p>
        <ol style="font-size:10pt">
            <li>High employment → workers strong → wages ↑</li>
            <li>Wages ↑ → profit squeeze → investment ↓</li>
            <li>Investment ↓ → employment ↓ → workers weak</li>
            <li>Workers weak → wages ↓ → profits ↑</li>
            <li>Profits ↑ → investment ↑ → employment ↑</li>
            <li>Cycle repeats...</li>
        </ol>
        <p><i>"Reserve army of labor" maintains<br>
        capitalist profitability</i></p>
        """)
        theory_layout.addWidget(theory_text)
        theory_box.setLayout(theory_layout)
        layout.addWidget(theory_box)

        # Parameters section
        params_box = QGroupBox("Model Parameters")
        params_layout = QGridLayout()

        row = 0

        # Parameter a
        params_layout.addWidget(QLabel("a (baseline employment growth):"), row, 0)
        self.a_spin = QDoubleSpinBox()
        self.a_spin.setRange(0.001, 0.5)
        self.a_spin.setValue(self.a)
        self.a_spin.setSingleStep(0.01)
        self.a_spin.setDecimals(3)
        self.a_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.a_spin, row, 1)
        row += 1

        # Parameter b
        params_layout.addWidget(QLabel("b (wage share effect):"), row, 0)
        self.b_spin = QDoubleSpinBox()
        self.b_spin.setRange(0.001, 1.0)
        self.b_spin.setValue(self.b)
        self.b_spin.setSingleStep(0.01)
        self.b_spin.setDecimals(3)
        self.b_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.b_spin, row, 1)
        row += 1

        # Parameter c
        params_layout.addWidget(QLabel("c (employment on wages):"), row, 0)
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.001, 1.0)
        self.c_spin.setValue(self.c)
        self.c_spin.setSingleStep(0.01)
        self.c_spin.setDecimals(3)
        self.c_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.c_spin, row, 1)
        row += 1

        # Parameter d
        params_layout.addWidget(QLabel("d (natural wage decline):"), row, 0)
        self.d_spin = QDoubleSpinBox()
        self.d_spin.setRange(0.001, 0.5)
        self.d_spin.setValue(self.d)
        self.d_spin.setSingleStep(0.01)
        self.d_spin.setDecimals(3)
        self.d_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.d_spin, row, 1)
        row += 1

        params_layout.addWidget(QLabel(""), row, 0)  # Spacer
        row += 1

        # Initial conditions
        params_layout.addWidget(QLabel("Initial Employment Rate (v₀):"), row, 0)
        self.v0_spin = QDoubleSpinBox()
        self.v0_spin.setRange(0.5, 1.0)
        self.v0_spin.setValue(self.v0)
        self.v0_spin.setSingleStep(0.05)
        self.v0_spin.setDecimals(3)
        self.v0_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.v0_spin, row, 1)
        row += 1

        params_layout.addWidget(QLabel("Initial Wage Share (u₀):"), row, 0)
        self.u0_spin = QDoubleSpinBox()
        self.u0_spin.setRange(0.3, 0.9)
        self.u0_spin.setValue(self.u0)
        self.u0_spin.setSingleStep(0.05)
        self.u0_spin.setDecimals(3)
        self.u0_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.u0_spin, row, 1)
        row += 1

        # Simulation time
        params_layout.addWidget(QLabel("Simulation Time:"), row, 0)
        self.tmax_spin = QSpinBox()
        self.tmax_spin.setRange(50, 500)
        self.tmax_spin.setValue(self.t_max)
        self.tmax_spin.setSingleStep(50)
        self.tmax_spin.valueChanged.connect(self.on_parameter_change)
        params_layout.addWidget(self.tmax_spin, row, 1)
        row += 1

        params_box.setLayout(params_layout)
        layout.addWidget(params_box)

        # Animation controls
        anim_box = QGroupBox("Animation")
        anim_layout = QVBoxLayout()

        self.anim_button = QPushButton("Start Animation")
        self.anim_button.clicked.connect(self.toggle_animation)
        anim_layout.addWidget(self.anim_button)

        self.reset_button = QPushButton("Reset Simulation")
        self.reset_button.clicked.connect(self.reset_simulation)
        anim_layout.addWidget(self.reset_button)

        anim_box.setLayout(anim_layout)
        layout.addWidget(anim_box)

        # Current state display
        state_box = QGroupBox("Current State")
        state_layout = QVBoxLayout()

        self.state_text = QTextEdit()
        self.state_text.setReadOnly(True)
        self.state_text.setMaximumHeight(150)
        state_layout.addWidget(self.state_text)

        state_box.setLayout(state_layout)
        layout.addWidget(state_box)

        # Export button
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        layout.addWidget(export_btn)

        layout.addStretch()

        panel.setLayout(layout)
        return panel

    def create_visualization_panel(self):
        """Create the visualization panel with tabs."""

        tab_widget = QTabWidget()

        # Tab 1: Phase diagram
        phase_tab = QWidget()
        phase_layout = QVBoxLayout()
        self.phase_figure = Figure(figsize=(10, 8))
        self.phase_canvas = FigureCanvas(self.phase_figure)
        phase_layout.addWidget(self.phase_canvas)
        phase_tab.setLayout(phase_layout)
        tab_widget.addTab(phase_tab, "Phase Diagram (u vs v)")

        # Tab 2: Time series
        timeseries_tab = QWidget()
        timeseries_layout = QVBoxLayout()
        self.timeseries_figure = Figure(figsize=(10, 8))
        self.timeseries_canvas = FigureCanvas(self.timeseries_figure)
        timeseries_layout.addWidget(self.timeseries_canvas)
        timeseries_tab.setLayout(timeseries_layout)
        tab_widget.addTab(timeseries_tab, "Time Series")

        return tab_widget

    def on_parameter_change(self):
        """Handle parameter changes."""
        self.a = self.a_spin.value()
        self.b = self.b_spin.value()
        self.c = self.c_spin.value()
        self.d = self.d_spin.value()
        self.v0 = self.v0_spin.value()
        self.u0 = self.u0_spin.value()
        self.t_max = self.tmax_spin.value()

        self.run_simulation()

    def goodwin_derivatives(self, state, t):
        """
        Calculate derivatives for the Goodwin model.

        Args:
            state: [v, u] where v=employment rate, u=wage share
            t: time (not used but required by odeint)

        Returns:
            [dv/dt, du/dt]
        """
        v, u = state

        dv_dt = v * (self.a - self.b * u)
        du_dt = u * (self.c * v - self.d)

        return [dv_dt, du_dt]

    def run_simulation(self):
        """Run the Goodwin model simulation."""

        # Time array
        self.t = np.arange(0, self.t_max, self.dt)

        # Initial state
        state0 = [self.v0, self.u0]

        # Solve ODEs
        solution = odeint(self.goodwin_derivatives, state0, self.t)

        self.v = solution[:, 0]  # Employment rate
        self.u = solution[:, 1]  # Wage share

        # Reset animation
        self.animation_step = 0

        # Update displays
        self.update_all_plots()
        self.update_state_display()

    def update_all_plots(self, animate_to_step=None):
        """Update all visualization plots."""

        # Determine how much data to show
        if animate_to_step is not None:
            end_idx = min(animate_to_step, len(self.t))
        else:
            end_idx = len(self.t)

        self.plot_phase_diagram(end_idx)
        self.plot_timeseries(end_idx)

    def plot_phase_diagram(self, end_idx):
        """Plot the phase diagram (u vs v)."""

        self.phase_figure.clear()
        ax = self.phase_figure.add_subplot(111)

        # Plot trajectory
        ax.plot(self.v[:end_idx], self.u[:end_idx], 'b-', linewidth=1.5, alpha=0.7)

        # Plot starting point
        ax.plot(self.v[0], self.u[0], 'go', markersize=12, label='Start', zorder=5)

        # Plot current/end point
        if end_idx > 1:
            ax.plot(self.v[end_idx-1], self.u[end_idx-1], 'ro',
                   markersize=12, label='Current', zorder=5)

        # Add vector field (direction field)
        if end_idx == len(self.t):  # Only show for complete simulation
            v_grid = np.linspace(0.5, 1.0, 15)
            u_grid = np.linspace(0.3, 0.9, 15)
            V_grid, U_grid = np.meshgrid(v_grid, u_grid)

            dV = V_grid * (self.a - self.b * U_grid)
            dU = U_grid * (self.c * V_grid - self.d)

            # Normalize for better visualization
            M = np.sqrt(dV**2 + dU**2)
            M[M == 0] = 1  # Avoid division by zero
            dV_norm = dV / M
            dU_norm = dU / M

            ax.quiver(V_grid, U_grid, dV_norm, dU_norm,
                     M, alpha=0.3, cmap='gray', scale=25)

        # Labels and formatting
        ax.set_xlabel('Employment Rate (v)', fontsize=12)
        ax.set_ylabel('Wage Share (u)', fontsize=12)
        ax.set_title('Goodwin Cycle: Phase Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.3, 0.9)

        # Add interpretation text
        ax.text(0.52, 0.88, 'High wages,\nHigh employment', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(0.52, 0.32, 'Low wages,\nLow employment', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        self.phase_figure.tight_layout()
        self.phase_canvas.draw()

    def plot_timeseries(self, end_idx):
        """Plot time series of employment and wage share."""

        self.timeseries_figure.clear()

        # Two subplots
        ax1 = self.timeseries_figure.add_subplot(211)
        ax2 = self.timeseries_figure.add_subplot(212)

        # Employment rate
        ax1.plot(self.t[:end_idx], self.v[:end_idx], 'b-', linewidth=2)
        ax1.set_ylabel('Employment Rate (v)', fontsize=11)
        ax1.set_title('Goodwin Cycle: Time Series', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, self.t_max)
        ax1.set_ylim(0.5, 1.0)

        # Wage share
        ax2.plot(self.t[:end_idx], self.u[:end_idx], 'r-', linewidth=2)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Wage Share (u)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, self.t_max)
        ax2.set_ylim(0.3, 0.9)

        self.timeseries_figure.tight_layout()
        self.timeseries_canvas.draw()

    def update_state_display(self):
        """Update the current state text display."""

        if self.animation_step < len(self.t):
            idx = self.animation_step
        else:
            idx = -1

        current_v = self.v[idx]
        current_u = self.u[idx]
        current_t = self.t[idx]

        # Calculate profit share
        profit_share = 1 - current_u

        # Determine phase of cycle
        if current_v > 0.9 and current_u > 0.7:
            phase = "Peak: High employment & wages → Profit squeeze imminent"
        elif current_v < 0.7 and current_u < 0.5:
            phase = "Trough: Low employment & wages → Recovery ahead"
        elif current_u > current_v:
            phase = "Rising wages, weakening employment → Contraction phase"
        else:
            phase = "Recovering employment, stable wages → Expansion phase"

        html = f"""
        <h3>Current State (t = {current_t:.1f})</h3>
        <p><b>Employment Rate:</b> {current_v:.1%}</p>
        <p><b>Unemployment Rate:</b> {(1-current_v):.1%}</p>
        <p><b>Wage Share:</b> {current_u:.1%}</p>
        <p><b>Profit Share:</b> {profit_share:.1%}</p>
        <hr>
        <p><b>Phase:</b><br><i>{phase}</i></p>
        """

        self.state_text.setHtml(html)

    def toggle_animation(self):
        """Toggle animation on/off."""

        if self.is_animating:
            self.timer.stop()
            self.is_animating = False
            self.anim_button.setText("Start Animation")
        else:
            self.timer.start(50)  # 50ms = 20 FPS
            self.is_animating = True
            self.anim_button.setText("Pause Animation")

    def animation_step_forward(self):
        """Advance animation by one step."""

        step_size = max(1, len(self.t) // 200)  # Adaptive step size
        self.animation_step += step_size

        if self.animation_step >= len(self.t):
            self.animation_step = len(self.t)
            self.toggle_animation()  # Stop at end

        self.update_all_plots(self.animation_step)
        self.update_state_display()

    def reset_simulation(self):
        """Reset the simulation to initial state."""

        self.animation_step = 0
        if self.is_animating:
            self.toggle_animation()

        self.run_simulation()

    def export_data(self):
        """Export simulation data to CSV."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'goodwin_cycle_{timestamp}.csv'

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['Time', 'EmploymentRate', 'WageShare', 'ProfitShare',
                           'UnemploymentRate'])

            # Data
            for i in range(len(self.t)):
                writer.writerow([
                    self.t[i],
                    self.v[i],
                    self.u[i],
                    1 - self.u[i],
                    1 - self.v[i]
                ])

        print(f"Data exported to {filename}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = GoodwinCycleApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
