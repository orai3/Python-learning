"""
Report View

PyQt6 interface for generating comprehensive reports.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QTextEdit, QGroupBox, QCheckBox,
                             QMessageBox, QFileDialog, QScrollArea)
from PyQt6.QtCore import Qt
from controllers.analysis_controller import AnalysisController
from datetime import datetime


class ReportView(QWidget):
    """
    View for generating comprehensive academic reports.
    """

    def __init__(self, analysis_controller: AnalysisController, parent=None):
        """
        Initialize report view.

        Args:
            analysis_controller: AnalysisController instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.analysis_controller = analysis_controller
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Report Generation")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Report options group
        options_group = self.create_report_options_group()
        layout.addWidget(options_group)

        # Report preview
        preview_group = self.create_report_preview_group()
        layout.addWidget(preview_group)

        # Generate button
        button_layout = QHBoxLayout()

        generate_btn = QPushButton("Generate Report")
        generate_btn.clicked.connect(self.generate_report)
        generate_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        button_layout.addWidget(generate_btn)

        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_report)
        button_layout.addWidget(export_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_report_options_group(self) -> QGroupBox:
        """Create report options controls."""
        group = QGroupBox("Report Options")
        layout = QVBoxLayout()

        # Framework selection checkboxes
        layout.addWidget(QLabel("Include Frameworks:"))

        self.framework_checkboxes = {}
        frameworks = self.analysis_controller.get_available_frameworks()

        for framework in frameworks:
            checkbox = QCheckBox(framework.replace('_', ' ').title())
            checkbox.setChecked(True)
            self.framework_checkboxes[framework] = checkbox
            layout.addWidget(checkbox)

        # Additional sections
        layout.addWidget(QLabel("\nInclude Sections:"))

        self.include_theory = QCheckBox("Theoretical Background")
        self.include_theory.setChecked(True)
        layout.addWidget(self.include_theory)

        self.include_data_summary = QCheckBox("Data Summary Statistics")
        self.include_data_summary.setChecked(True)
        layout.addWidget(self.include_data_summary)

        self.include_visualizations = QCheckBox("Visualization Descriptions")
        self.include_visualizations.setChecked(False)
        layout.addWidget(self.include_visualizations)

        group.setLayout(layout)
        return group

    def create_report_preview_group(self) -> QGroupBox:
        """Create report preview display."""
        group = QGroupBox("Report Preview")
        layout = QVBoxLayout()

        self.report_preview = QTextEdit()
        self.report_preview.setReadOnly(True)
        self.report_preview.setStyleSheet("font-family: monospace; font-size: 10pt;")

        # Initial message
        self.report_preview.setPlainText(
            "Click 'Generate Report' to create a comprehensive analysis report.\n\n"
            "The report will include:\n"
            "- Analysis from selected frameworks\n"
            "- Key economic indicators\n"
            "- Theoretical interpretations\n"
            "- Policy implications\n"
            "- Data sources and methodology"
        )

        layout.addWidget(self.report_preview)

        group.setLayout(layout)
        return group

    def generate_report(self):
        """Generate comprehensive report."""
        try:
            report_lines = []

            # Header
            report_lines.append("=" * 90)
            report_lines.append("HETERODOX MACROECONOMIC ANALYSIS REPORT")
            report_lines.append("=" * 90)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("=" * 90)
            report_lines.append("")

            # Executive Summary
            report_lines.append("EXECUTIVE SUMMARY")
            report_lines.append("-" * 90)
            report_lines.append("This report presents a multi-framework analysis of macroeconomic data")
            report_lines.append("using heterodox economic theories. The analysis integrates Post-Keynesian,")
            report_lines.append("Marxian, and Institutionalist perspectives to provide a comprehensive")
            report_lines.append("understanding of economic dynamics.")
            report_lines.append("")

            # Data Summary (if selected)
            if self.include_data_summary.isChecked():
                report_lines.append("\nDATA SOURCES")
                report_lines.append("-" * 90)
                datasets = self.analysis_controller.data_model.get_available_datasets()
                if datasets:
                    for dataset in datasets:
                        metadata = self.analysis_controller.data_model.get_metadata(dataset)
                        if metadata:
                            report_lines.append(f"\n{dataset.upper()}")
                            report_lines.append(f"  Observations: {metadata['rows']:,}")
                            report_lines.append(f"  Variables: {len(metadata['columns'])}")
                report_lines.append("")

            # Framework Analysis
            selected_frameworks = [
                name for name, checkbox in self.framework_checkboxes.items()
                if checkbox.isChecked()
            ]

            if not selected_frameworks:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please select at least one framework."
                )
                return

            for framework_name in selected_frameworks:
                try:
                    # Get full analysis report
                    framework_report = self.analysis_controller.generate_report(framework_name)
                    report_lines.append("\n" + framework_report)

                except Exception as e:
                    report_lines.append(f"\nError analyzing {framework_name}: {str(e)}\n")

            # Comparative Analysis
            if len(selected_frameworks) > 1:
                report_lines.append("\n" + "=" * 90)
                report_lines.append("COMPARATIVE ANALYSIS")
                report_lines.append("=" * 90)
                report_lines.append("")
                report_lines.append("Cross-Framework Insights:")
                report_lines.append("-" * 90)

                # Run comparative analysis
                all_results = self.analysis_controller.analyze_all_frameworks()

                # Compare common indicators
                common_indicators = ['wage_share', 'unemployment_rate', 'gini']
                for indicator in common_indicators:
                    values = []
                    for framework in selected_frameworks:
                        if framework in all_results:
                            if 'indicators' in all_results[framework]:
                                if indicator in all_results[framework]['indicators']:
                                    value = all_results[framework]['indicators'][indicator]
                                    if value is not None:
                                        values.append((framework, value))

                    if values:
                        report_lines.append(f"\n{indicator.replace('_', ' ').title()}:")
                        for framework, value in values:
                            report_lines.append(f"  {framework.replace('_', ' ').title()}: {value:.2f}")

            # Policy Implications
            report_lines.append("\n" + "=" * 90)
            report_lines.append("POLICY IMPLICATIONS")
            report_lines.append("=" * 90)
            report_lines.append("")
            report_lines.append("Based on the multi-framework analysis, key policy considerations include:")
            report_lines.append("")

            if 'post_keynesian' in selected_frameworks:
                report_lines.append("Post-Keynesian Perspective:")
                report_lines.append("  • Active fiscal policy to maintain full employment")
                report_lines.append("  • Financial regulation to prevent systemic instability")
                report_lines.append("  • Sectoral balance monitoring for macroeconomic sustainability")
                report_lines.append("")

            if 'marxian' in selected_frameworks:
                report_lines.append("Marxian Perspective:")
                report_lines.append("  • Strengthening labor bargaining power to improve wage share")
                report_lines.append("  • Addressing tendency of profit rate to fall through public investment")
                report_lines.append("  • Reducing income inequality through progressive redistribution")
                report_lines.append("")

            if 'institutionalist' in selected_frameworks:
                report_lines.append("Institutionalist Perspective:")
                report_lines.append("  • Institutional reform to support social provisioning")
                report_lines.append("  • Regulation of financial sector dominance")
                report_lines.append("  • Comparative analysis of successful institutional configurations")
                report_lines.append("")

            # Methodology
            if self.include_theory.isChecked():
                report_lines.append("\n" + "=" * 90)
                report_lines.append("METHODOLOGY AND THEORETICAL FOUNDATIONS")
                report_lines.append("=" * 90)
                report_lines.append("")
                report_lines.append("This analysis employs a pluralist approach to economic analysis,")
                report_lines.append("drawing on multiple heterodox traditions. Each framework provides")
                report_lines.append("distinct insights into economic dynamics:")
                report_lines.append("")
                report_lines.append("• Post-Keynesian: Emphasizes effective demand, endogenous money,")
                report_lines.append("  and stock-flow consistent accounting")
                report_lines.append("• Marxian: Focuses on exploitation, class conflict, and crisis tendencies")
                report_lines.append("• Institutionalist: Highlights power relations, institutional change,")
                report_lines.append("  and comparative systems analysis")
                report_lines.append("")

            # Footer
            report_lines.append("\n" + "=" * 90)
            report_lines.append("END OF REPORT")
            report_lines.append("=" * 90)
            report_lines.append("")
            report_lines.append("Note: This analysis is generated using the Heterodox Macro Dashboard,")
            report_lines.append("an academic research tool for pluralist economic analysis.")
            report_lines.append("")

            # Display report
            full_report = "\n".join(report_lines)
            self.report_preview.setPlainText(full_report)

            QMessageBox.information(
                self,
                "Success",
                "Report generated successfully."
            )

        except Exception as e:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to generate report: {str(e)}"
            )

    def export_report(self):
        """Export report to file."""
        report_text = self.report_preview.toPlainText()

        if not report_text or report_text.startswith("Click 'Generate Report'"):
            QMessageBox.warning(
                self,
                "Warning",
                "Please generate a report first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Report",
            f"heterodox_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)

                QMessageBox.information(
                    self,
                    "Success",
                    f"Report exported to {file_path}"
                )

            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Failed to export report: {str(e)}"
                )
