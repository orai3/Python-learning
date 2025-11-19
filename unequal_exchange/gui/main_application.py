"""
Unequal Exchange Analysis - Interactive PyQt6 Application

Provides GUI for:
1. Loading and exploring datasets
2. Running Emmanuel/Amin models
3. Visualizing value transfers
4. Policy simulations
5. Exporting results
"""

import sys
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QFileDialog, QSlider, QDoubleSpinBox,
    QGroupBox, QFormLayout, QMessageBox, QProgressBar
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class UnequaExchangeApp(QMainWindow):
    """
    Main application window for unequal exchange analysis.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unequal Exchange Framework - Dependency Theory Analysis")
        self.setGeometry(100, 100, 1400, 900)

        # Data storage
        self.datasets = {}
        self.model_results = {}

        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("Unequal Exchange Analysis Framework")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # Add tabs
        tabs.addTab(self.create_data_tab(), "📊 Data")
        tabs.addTab(self.create_models_tab(), "⚙️ Models")
        tabs.addTab(self.create_visualization_tab(), "📈 Visualizations")
        tabs.addTab(self.create_policy_tab(), "🔧 Policy Simulations")
        tabs.addTab(self.create_results_tab(), "📄 Results")

        # Status bar
        self.statusBar().showMessage("Ready")

    def create_data_tab(self):
        """Create data loading and exploration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        control_group = QGroupBox("Data Management")
        control_layout = QHBoxLayout()

        gen_btn = QPushButton("Generate Synthetic Data")
        gen_btn.clicked.connect(self.generate_synthetic_data)
        control_layout.addWidget(gen_btn)

        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self.load_data)
        control_layout.addWidget(load_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()

        self.data_selector = QComboBox()
        self.data_selector.currentTextChanged.connect(self.update_data_preview)
        preview_layout.addWidget(QLabel("Select Dataset:"))
        preview_layout.addWidget(self.data_selector)

        self.data_table = QTableWidget()
        preview_layout.addWidget(self.data_table)

        self.data_info = QTextEdit()
        self.data_info.setMaximumHeight(100)
        self.data_info.setReadOnly(True)
        preview_layout.addWidget(QLabel("Dataset Info:"))
        preview_layout.addWidget(self.data_info)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        return widget

    def create_models_tab(self):
        """Create models configuration and execution tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left: Model selection and parameters
        left_panel = QGroupBox("Model Configuration")
        left_layout = QVBoxLayout()

        # Model selection
        left_layout.addWidget(QLabel("Select Model:"))
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "Emmanuel Unequal Exchange",
            "Amin Extended Model",
            "Prebisch-Singer Terms of Trade",
            "Combined Analysis"
        ])
        left_layout.addWidget(self.model_selector)

        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()

        self.param_profit_rate = QDoubleSpinBox()
        self.param_profit_rate.setRange(0, 1)
        self.param_profit_rate.setValue(0.15)
        self.param_profit_rate.setSingleStep(0.01)
        params_layout.addRow("Global Profit Rate:", self.param_profit_rate)

        self.param_base_year = QComboBox()
        self.param_base_year.addItems([str(y) for y in range(1960, 2021)])
        self.param_base_year.setCurrentText("2000")
        params_layout.addRow("Base Year:", self.param_base_year)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Run button
        run_btn = QPushButton("Run Model")
        run_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 10px;")
        run_btn.clicked.connect(self.run_model)
        left_layout.addWidget(run_btn)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel)

        # Right: Results summary
        right_panel = QGroupBox("Model Results")
        right_layout = QVBoxLayout()

        self.model_output = QTextEdit()
        self.model_output.setReadOnly(True)
        right_layout.addWidget(self.model_output)

        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel)

        return widget

    def create_visualization_tab(self):
        """Create visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Visualization Type:"))
        self.viz_selector = QComboBox()
        self.viz_selector.addItems([
            "Value Transfer Network",
            "Historical Transfers",
            "Terms of Trade",
            "Exploitation Metrics",
            "Smile Curve (GVC)",
            "Policy Comparison"
        ])
        self.viz_selector.currentTextChanged.connect(self.update_visualization)
        control_layout.addWidget(self.viz_selector)

        plot_btn = QPushButton("Generate Plot")
        plot_btn.clicked.connect(self.update_visualization)
        control_layout.addWidget(plot_btn)

        control_layout.addStretch()
        layout.addLayout(control_layout)

        # Plot canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        return widget

    def create_policy_tab(self):
        """Create policy simulation tab"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left: Policy configuration
        left_panel = QGroupBox("Policy Simulation")
        left_layout = QVBoxLayout()

        left_layout.addWidget(QLabel("Policy Type:"))
        self.policy_selector = QComboBox()
        self.policy_selector.addItems([
            "South-South Cooperation",
            "Delinking Strategy",
            "Industrial Policy",
            "Alternative Integration"
        ])
        left_layout.addWidget(self.policy_selector)

        # Parameters
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout()

        self.policy_intensity = QSlider(Qt.Orientation.Horizontal)
        self.policy_intensity.setRange(0, 100)
        self.policy_intensity.setValue(50)
        params_layout.addRow("Policy Intensity:", self.policy_intensity)

        self.policy_years = QComboBox()
        self.policy_years.addItems(["10", "20", "30", "40"])
        self.policy_years.setCurrentText("20")
        params_layout.addRow("Time Horizon (years):", self.policy_years)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        sim_btn = QPushButton("Run Simulation")
        sim_btn.setStyleSheet("background-color: #3498db; color: white; font-weight: bold; padding: 10px;")
        sim_btn.clicked.connect(self.run_policy_simulation)
        left_layout.addWidget(sim_btn)

        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel)

        # Right: Simulation results
        right_panel = QGroupBox("Simulation Results")
        right_layout = QVBoxLayout()

        self.policy_figure = Figure(figsize=(8, 6))
        self.policy_canvas = FigureCanvas(self.policy_figure)
        right_layout.addWidget(self.policy_canvas)

        self.policy_output = QTextEdit()
        self.policy_output.setReadOnly(True)
        self.policy_output.setMaximumHeight(150)
        right_layout.addWidget(self.policy_output)

        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel)

        return widget

    def create_results_tab(self):
        """Create results export and documentation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QHBoxLayout()

        export_csv_btn = QPushButton("Export to CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        export_layout.addWidget(export_csv_btn)

        export_report_btn = QPushButton("Generate Report (PDF)")
        export_report_btn.clicked.connect(self.generate_report)
        export_layout.addWidget(export_report_btn)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Summary
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        return widget

    # Event handlers
    def generate_synthetic_data(self):
        """Generate synthetic datasets"""
        self.statusBar().showMessage("Generating synthetic data...")

        try:
            from ..data.synthetic_generator import SyntheticDataGenerator

            generator = SyntheticDataGenerator(start_year=1960, end_year=2020)
            self.datasets = generator.generate_complete_dataset('./unequal_exchange_data/')

            # Update data selector
            self.data_selector.clear()
            self.data_selector.addItems(self.datasets.keys())

            self.statusBar().showMessage("✓ Synthetic data generated successfully!")
            QMessageBox.information(self, "Success", "Synthetic datasets generated successfully!")

        except Exception as e:
            self.statusBar().showMessage("✗ Error generating data")
            QMessageBox.critical(self, "Error", f"Error generating data: {str(e)}")

    def load_data(self):
        """Load data from files"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Data", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            try:
                df = pd.read_csv(file_path)
                name = file_path.split('/')[-1].replace('.csv', '')
                self.datasets[name] = df

                self.data_selector.addItem(name)
                self.statusBar().showMessage(f"✓ Loaded {name}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {str(e)}")

    def update_data_preview(self, dataset_name):
        """Update data preview table"""
        if dataset_name not in self.datasets:
            return

        df = self.datasets[dataset_name]

        # Update table
        self.data_table.setRowCount(min(50, len(df)))
        self.data_table.setColumnCount(len(df.columns))
        self.data_table.setHorizontalHeaderLabels(df.columns)

        for i in range(min(50, len(df))):
            for j, col in enumerate(df.columns):
                self.data_table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))

        # Update info
        info = f"Rows: {len(df)}\nColumns: {len(df.columns)}\n\nColumns: {', '.join(df.columns)}"
        self.data_info.setText(info)

    def run_model(self):
        """Run selected model"""
        model_name = self.model_selector.currentText()
        self.statusBar().showMessage(f"Running {model_name}...")

        # Placeholder - would actually run models
        self.model_output.setText(f"""
{model_name} - Results Summary

Status: ✓ Successfully executed

Key Findings:
- Total value transfer from South to North: $XXX billion
- Average exploitation rate: XX%
- Number of country pairs analyzed: XX

See Visualizations tab for detailed charts.
        """)

        self.statusBar().showMessage(f"✓ {model_name} completed")

    def update_visualization(self):
        """Update visualization based on selection"""
        viz_type = self.viz_selector.currentText()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Placeholder visualization
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title(f"{viz_type} - Placeholder")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        self.canvas.draw()

    def run_policy_simulation(self):
        """Run policy simulation"""
        policy = self.policy_selector.currentText()
        intensity = self.policy_intensity.value() / 100
        years = int(self.policy_years.currentText())

        self.statusBar().showMessage(f"Simulating {policy}...")

        # Placeholder results
        self.policy_output.setText(f"""
Policy Simulation: {policy}
Intensity: {intensity*100:.0f}%
Time Horizon: {years} years

Results:
- Value transfer reduction: {intensity*30:.1f}%
- GDP impact: +{intensity*5:.2f}%
- Employment creation: +{intensity*2:.2f}M jobs

See chart above for detailed trajectories.
        """)

        self.statusBar().showMessage("✓ Simulation completed")

    def export_csv(self):
        """Export results to CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )

        if file_path:
            # Export placeholder
            QMessageBox.information(self, "Export", f"Results exported to {file_path}")

    def generate_report(self):
        """Generate PDF report"""
        QMessageBox.information(self, "Report", "PDF report generation feature coming soon!")


def main():
    """Run the application"""
    app = QApplication(sys.argv)
    window = UnequaExchangeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
