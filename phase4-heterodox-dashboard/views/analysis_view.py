"""
Analysis View

PyQt6 interface for theoretical framework analysis and visualization.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QTextEdit, QGroupBox, QScrollArea,
                             QTabWidget, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from controllers.analysis_controller import AnalysisController
from typing import Optional, Dict


class AnalysisView(QWidget):
    """
    View for theoretical framework analysis and visualization.
    """

    def __init__(self, analysis_controller: AnalysisController, parent=None):
        """
        Initialize analysis view.

        Args:
            analysis_controller: AnalysisController instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.analysis_controller = analysis_controller
        self.current_analysis = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Theoretical Framework Analysis")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Framework selection group
        framework_group = self.create_framework_selection_group()
        layout.addWidget(framework_group)

        # Create tab widget for different views
        self.tabs = QTabWidget()

        # Analysis results tab
        self.results_tab = self.create_results_tab()
        self.tabs.addTab(self.results_tab, "Analysis Results")

        # Visualization tab
        self.viz_tab = self.create_visualization_tab()
        self.tabs.addTab(self.viz_tab, "Visualizations")

        # Theory tab
        self.theory_tab = self.create_theory_tab()
        self.tabs.addTab(self.theory_tab, "Theoretical Background")

        layout.addWidget(self.tabs)

        self.setLayout(layout)

    def create_framework_selection_group(self) -> QGroupBox:
        """Create framework selection controls."""
        group = QGroupBox("Framework Selection")
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Select Framework:"))

        self.framework_combo = QComboBox()
        frameworks = self.analysis_controller.get_available_frameworks()
        self.framework_combo.addItems([f.replace('_', ' ').title() for f in frameworks])
        layout.addWidget(self.framework_combo)

        # Analyze button
        analyze_btn = QPushButton("Run Analysis")
        analyze_btn.clicked.connect(self.run_analysis)
        analyze_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        layout.addWidget(analyze_btn)

        # Compare all button
        compare_btn = QPushButton("Compare All Frameworks")
        compare_btn.clicked.connect(self.compare_frameworks)
        layout.addWidget(compare_btn)

        layout.addStretch()

        group.setLayout(layout)
        return group

    def create_results_tab(self) -> QWidget:
        """Create the analysis results tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.results_text)

        # Export button
        export_btn = QPushButton("Export Analysis Report")
        export_btn.clicked.connect(self.export_analysis)
        layout.addWidget(export_btn)

        widget.setLayout(layout)
        return widget

    def create_visualization_tab(self) -> QWidget:
        """Create the visualization tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Chart selection
        chart_layout = QHBoxLayout()
        chart_layout.addWidget(QLabel("Select Chart:"))

        self.chart_combo = QComboBox()
        self.chart_combo.addItems([
            "Sectoral Balances (Godley)",
            "Wage vs Profit Share",
            "Rate of Profit (Marxian)",
            "Correlation Matrix",
            "Time Series Analysis",
            "Lorenz Curve"
        ])
        chart_layout.addWidget(self.chart_combo)

        generate_btn = QPushButton("Generate Chart")
        generate_btn.clicked.connect(self.generate_chart)
        chart_layout.addWidget(generate_btn)

        chart_layout.addStretch()
        layout.addLayout(chart_layout)

        # Scroll area for chart
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        self.chart_widget = QWidget()
        self.chart_layout = QVBoxLayout()
        self.chart_widget.setLayout(self.chart_layout)

        scroll.setWidget(self.chart_widget)
        layout.addWidget(scroll)

        widget.setLayout(layout)
        return widget

    def create_theory_tab(self) -> QWidget:
        """Create the theoretical background tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        self.theory_text = QTextEdit()
        self.theory_text.setReadOnly(True)
        self.theory_text.setStyleSheet("font-family: serif; font-size: 11pt;")
        layout.addWidget(self.theory_text)

        widget.setLayout(layout)
        return widget

    def run_analysis(self):
        """Run analysis with selected framework."""
        framework_name = self.framework_combo.currentText().lower().replace(' ', '_')

        try:
            # Run analysis
            self.current_analysis = self.analysis_controller.analyze_with_framework(framework_name)

            # Generate report
            report = self.analysis_controller.generate_report(framework_name)
            self.results_text.setPlainText(report)

            # Load theoretical background
            theory = self.analysis_controller.get_framework_theory(framework_name)
            self.theory_text.setPlainText(theory)

            QMessageBox.information(
                self,
                "Success",
                f"Analysis completed using {self.framework_combo.currentText()} framework."
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Analysis failed: {str(e)}"
            )

    def compare_frameworks(self):
        """Compare analysis across all frameworks."""
        try:
            results = self.analysis_controller.analyze_all_frameworks()

            # Generate comparative report
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("COMPARATIVE FRAMEWORK ANALYSIS")
            report_lines.append("=" * 80)
            report_lines.append("")

            for framework_name, analysis in results.items():
                report_lines.append(f"\n{framework_name.replace('_', ' ').upper()}")
                report_lines.append("-" * 80)

                if 'indicators' in analysis:
                    for key, value in analysis['indicators'].items():
                        if value is not None:
                            report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")

                if 'interpretation' in analysis:
                    report_lines.append("\n  Key Insights:")
                    for interp in analysis['interpretation'][:3]:  # Show first 3
                        report_lines.append(f"    • {interp}")

                report_lines.append("")

            self.results_text.setPlainText("\n".join(report_lines))

            QMessageBox.information(
                self,
                "Success",
                "Comparative analysis completed for all frameworks."
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Comparative analysis failed: {str(e)}"
            )

    def generate_chart(self):
        """Generate selected chart."""
        chart_type = self.chart_combo.currentText()

        try:
            # Clear existing chart
            for i in reversed(range(self.chart_layout.count())):
                widget = self.chart_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)

            # Generate appropriate chart
            if chart_type == "Sectoral Balances (Godley)":
                fig = self.analysis_controller.create_sectoral_balances_chart()
            elif chart_type == "Wage vs Profit Share":
                fig = self.analysis_controller.create_wage_profit_chart()
            elif chart_type == "Rate of Profit (Marxian)":
                fig = self.analysis_controller.create_rate_of_profit_chart()
            elif chart_type == "Correlation Matrix":
                fig = self.analysis_controller.create_correlation_matrix('macro')
            elif chart_type == "Lorenz Curve":
                fig = self.analysis_controller.create_lorenz_curve('inequality', 'income')
            elif chart_type == "Time Series Analysis":
                fig = self.analysis_controller.create_time_series_chart(
                    'macro',
                    ['gdp', 'consumption', 'investment'],
                    'Macroeconomic Aggregates',
                    'Value (Billions)'
                )
            else:
                return

            # Add canvas to layout
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)

            self.chart_layout.addWidget(toolbar)
            self.chart_layout.addWidget(canvas)

            canvas.draw()

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to generate chart: {str(e)}"
            )

    def export_analysis(self):
        """Export analysis results to file."""
        if not self.current_analysis:
            QMessageBox.warning(
                self,
                "Warning",
                "Please run an analysis first."
            )
            return

        framework_name = self.framework_combo.currentText().lower().replace(' ', '_')

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Analysis",
            f"{framework_name}_analysis.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            success = self.analysis_controller.export_analysis_results(
                framework_name,
                file_path
            )

            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Analysis exported to {file_path}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Failed to export analysis"
                )
