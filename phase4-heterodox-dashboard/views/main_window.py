"""
Main Window

Primary application window for the Heterodox Macro Dashboard.
"""

from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QMenuBar, QMenu,
                             QMessageBox, QApplication, QFileDialog, QStatusBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from controllers.data_controller import DataController
from controllers.analysis_controller import AnalysisController
from views.data_view import DataView
from views.analysis_view import AnalysisView
from views.report_view import ReportView


class MainWindow(QMainWindow):
    """
    Main application window.

    Provides tabbed interface for:
    - Data management
    - Theoretical framework analysis
    - Report generation
    """

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        # Initialize controllers
        self.data_controller = DataController()
        self.analysis_controller = None  # Will be initialized after data is loaded

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Heterodox Macro Dashboard - Academic Research Tool")
        self.setGeometry(100, 100, 1400, 900)

        # Create menu bar
        self.create_menu_bar()

        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Initialize views (analysis controller will be created when data is loaded)
        self.data_view = DataView(self.data_controller)
        self.tabs.addTab(self.data_view, "Data Management")

        # Create placeholder tabs (will be populated after data load)
        self.analysis_view = None
        self.report_view = None

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load datasets to begin analysis.")

        # Auto-load default datasets on startup
        self.load_default_data()

    def create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_data_action = QAction("&Load Default Datasets", self)
        load_data_action.setShortcut("Ctrl+L")
        load_data_action.triggered.connect(self.load_default_data)
        file_menu.addAction(load_data_action)

        load_custom_action = QAction("Load &Custom Dataset", self)
        load_custom_action.triggered.connect(self.load_custom_data)
        file_menu.addAction(load_custom_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        pk_action = QAction("&Post-Keynesian Analysis", self)
        pk_action.triggered.connect(lambda: self.quick_analysis('post_keynesian'))
        analysis_menu.addAction(pk_action)

        marxian_action = QAction("&Marxian Analysis", self)
        marxian_action.triggered.connect(lambda: self.quick_analysis('marxian'))
        analysis_menu.addAction(marxian_action)

        inst_action = QAction("&Institutionalist Analysis", self)
        inst_action.triggered.connect(lambda: self.quick_analysis('institutionalist'))
        analysis_menu.addAction(inst_action)

        analysis_menu.addSeparator()

        compare_action = QAction("&Compare All Frameworks", self)
        compare_action.triggered.connect(self.compare_all_frameworks)
        analysis_menu.addAction(compare_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        user_guide_action = QAction("&User Guide", self)
        user_guide_action.triggered.connect(self.show_user_guide)
        help_menu.addAction(user_guide_action)

        theory_action = QAction("&Theoretical Background", self)
        theory_action.triggered.connect(self.show_theory)
        help_menu.addAction(theory_action)

    def load_default_data(self):
        """Load default datasets."""
        self.status_bar.showMessage("Loading default datasets...")
        QApplication.processEvents()

        results = self.data_controller.load_default_datasets()

        success_count = sum(1 for v in results.values() if v)

        if success_count > 0:
            # Initialize analysis controller now that we have data
            if self.analysis_controller is None:
                data_model = self.data_controller.get_data_model()
                self.analysis_controller = AnalysisController(data_model)

                # Create analysis and report views
                self.analysis_view = AnalysisView(self.analysis_controller)
                self.tabs.addTab(self.analysis_view, "Framework Analysis")

                self.report_view = ReportView(self.analysis_controller)
                self.tabs.addTab(self.report_view, "Report Generation")

            self.status_bar.showMessage(
                f"Loaded {success_count} datasets. Ready for analysis."
            )

            QMessageBox.information(
                self,
                "Data Loaded",
                f"Successfully loaded {success_count} datasets.\n\n"
                "You can now perform theoretical framework analysis."
            )
        else:
            self.status_bar.showMessage("Failed to load datasets.")
            QMessageBox.warning(
                self,
                "Load Error",
                "Failed to load default datasets. Please check data directory."
            )

    def load_custom_data(self):
        """Load custom dataset."""
        if self.data_view:
            self.tabs.setCurrentWidget(self.data_view)
            self.data_view.load_custom_dataset()

    def quick_analysis(self, framework_name: str):
        """Run quick analysis with specified framework."""
        if self.analysis_controller is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load datasets first."
            )
            return

        if self.analysis_view:
            self.tabs.setCurrentWidget(self.analysis_view)

            # Set framework and run
            index = self.analysis_view.framework_combo.findText(
                framework_name.replace('_', ' ').title()
            )
            if index >= 0:
                self.analysis_view.framework_combo.setCurrentIndex(index)
                self.analysis_view.run_analysis()

    def compare_all_frameworks(self):
        """Compare all frameworks."""
        if self.analysis_controller is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load datasets first."
            )
            return

        if self.analysis_view:
            self.tabs.setCurrentWidget(self.analysis_view)
            self.analysis_view.compare_frameworks()

    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>Heterodox Macro Dashboard</h2>
        <p><b>Version 1.0.0</b></p>

        <p>An academic research tool for pluralist economic analysis using
        heterodox theoretical frameworks.</p>

        <h3>Frameworks:</h3>
        <ul>
            <li><b>Post-Keynesian Economics</b> - Effective demand, endogenous money,
                stock-flow consistency</li>
            <li><b>Marxian Political Economy</b> - Exploitation, class struggle,
                crisis tendencies</li>
            <li><b>Institutionalist Economics</b> - Power relations, institutional change,
                comparative systems</li>
        </ul>

        <h3>Features:</h3>
        <ul>
            <li>Load and manage economic datasets</li>
            <li>Multi-framework theoretical analysis</li>
            <li>Publication-quality visualizations</li>
            <li>Comprehensive report generation</li>
        </ul>

        <p><i>Developed for academic research and teaching in heterodox economics.</i></p>
        """

        QMessageBox.about(self, "About Heterodox Macro Dashboard", about_text)

    def show_user_guide(self):
        """Show user guide."""
        guide_text = """
        HETERODOX MACRO DASHBOARD - USER GUIDE

        GETTING STARTED:
        1. Load data using File > Load Default Datasets
        2. Explore data in the Data Management tab
        3. Run framework analysis in the Framework Analysis tab
        4. Generate reports in the Report Generation tab

        DATA MANAGEMENT:
        - Load default datasets: Pre-configured economic datasets
        - Load custom CSV: Import your own data
        - Export datasets: Save processed data

        FRAMEWORK ANALYSIS:
        - Select a theoretical framework (Post-Keynesian, Marxian, Institutionalist)
        - Click "Run Analysis" to perform framework-specific analysis
        - View results in Analysis Results tab
        - Generate visualizations in Visualizations tab
        - Read theoretical background in Theory tab

        VISUALIZATIONS:
        - Sectoral Balances: Godley's stock-flow consistent approach
        - Wage vs Profit Share: Functional income distribution
        - Rate of Profit: Marxian profitability analysis
        - Lorenz Curve: Inequality visualization

        REPORT GENERATION:
        - Select frameworks to include
        - Choose additional sections
        - Generate comprehensive report
        - Export to text file

        KEYBOARD SHORTCUTS:
        - Ctrl+L: Load default datasets
        - Ctrl+Q: Quit application

        For more information, see the README and documentation files.
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("User Guide")
        msg.setText(guide_text)
        msg.setStyleSheet("QLabel{min-width: 600px; font-family: monospace;}")
        msg.exec()

    def show_theory(self):
        """Show theoretical background information."""
        theory_text = """
        HETERODOX ECONOMIC THEORIES

        This application implements three major heterodox schools of thought:

        POST-KEYNESIAN ECONOMICS:
        - Emphasizes uncertainty, historical time, and path dependence
        - Money is endogenous and credit-driven
        - Output determined by effective demand, not supply
        - Distribution affects aggregate demand and growth
        - Stock-flow consistent accounting (Godley)

        Key Figures: Kalecki, Robinson, Kaldor, Minsky, Godley, Lavoie

        MARXIAN POLITICAL ECONOMY:
        - Labor theory of value and exploitation
        - Capital accumulation driven by profit motive
        - Tendency of rate of profit to fall
        - Class struggle over distribution
        - Periodic crises inherent to capitalism

        Key Figures: Marx, Luxemburg, Sweezy, Shaikh, Foley, Harvey

        INSTITUTIONALIST ECONOMICS:
        - Institutions shape economic behavior
        - Power relations and social provisioning
        - Cumulative causation and path dependence
        - Technology and institutional co-evolution
        - Comparative systems analysis

        Key Figures: Veblen, Commons, Galbraith, Myrdal, Chang

        WHY PLURALISM?
        Each framework provides unique insights into economic dynamics.
        A pluralist approach offers richer analysis than any single perspective.
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("Theoretical Background")
        msg.setText(theory_text)
        msg.setStyleSheet("QLabel{min-width: 600px; font-family: monospace;}")
        msg.exec()
