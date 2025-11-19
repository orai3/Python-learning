"""
Institutional Political Economy Analysis Application
PyQt6 GUI for comparative political economy research

Features:
- Interactive data exploration
- PCA and cluster analysis
- Institutional complementarity testing
- Historical trajectory visualization
- Regime transition detection
- Export capabilities for academic research
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QGroupBox, QGridLayout, QTextEdit,
    QSlider, QFileDialog, QMessageBox, QSplitter, QListWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from political_economy_analysis import (
    PoliticalEconomyAnalyzer,
    RegulationSchoolAnalyzer
)


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for PyQt6"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class PoliticalEconomyApp(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Institutional Political Economy Analysis System")
        self.setGeometry(100, 100, 1400, 900)

        # Load data
        try:
            self.analyzer = PoliticalEconomyAnalyzer(
                data_path='/home/user/Python-learning/political_economy_dataset.csv'
            )
            self.df = self.analyzer.df
            self.reg_analyzer = RegulationSchoolAnalyzer(self.df)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            sys.exit(1)

        # Cache for analysis results
        self.pca_results = None
        self.cluster_results = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Header
        header = QLabel("Institutional Political Economy Quantitative Analysis System")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Info label
        info = QLabel(
            f"Dataset: {len(self.df)} observations | "
            f"{self.df['country'].nunique()} countries | "
            f"Period: {self.df['year'].min()}-{self.df['year'].max()}"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Create tabs
        self.create_overview_tab()
        self.create_pca_tab()
        self.create_cluster_tab()
        self.create_complementarity_tab()
        self.create_trajectory_tab()
        self.create_transitions_tab()
        self.create_regulation_school_tab()

    def create_overview_tab(self):
        """Overview and data exploration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls = QGroupBox("Data Selection")
        controls_layout = QGridLayout()

        controls_layout.addWidget(QLabel("Year:"), 0, 0)
        self.overview_year = QSpinBox()
        self.overview_year.setRange(self.df['year'].min(), self.df['year'].max())
        self.overview_year.setValue(self.df['year'].max())
        controls_layout.addWidget(self.overview_year, 0, 1)

        controls_layout.addWidget(QLabel("Regime Type:"), 0, 2)
        self.overview_regime = QComboBox()
        self.overview_regime.addItem("All")
        self.overview_regime.addItems(sorted(self.df['regime_type'].unique()))
        controls_layout.addWidget(self.overview_regime, 0, 3)

        controls_layout.addWidget(QLabel("Indicator:"), 1, 0)
        self.overview_indicator = QComboBox()
        indicators = [
            'neoliberalism_index', 'financialization_index',
            'power_resources_index', 'wage_share_gdp',
            'union_density', 'labor_market_coordination'
        ]
        self.overview_indicator.addItems(indicators)
        controls_layout.addWidget(self.overview_indicator, 1, 1)

        plot_btn = QPushButton("Plot Cross-Section")
        plot_btn.clicked.connect(self.plot_overview_cross_section)
        controls_layout.addWidget(plot_btn, 1, 2)

        time_plot_btn = QPushButton("Plot Time Series")
        time_plot_btn.clicked.connect(self.plot_overview_timeseries)
        controls_layout.addWidget(time_plot_btn, 1, 3)

        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        # Canvas
        self.overview_canvas = MplCanvas(self, width=10, height=6)
        self.overview_toolbar = NavigationToolbar(self.overview_canvas, self)
        layout.addWidget(self.overview_toolbar)
        layout.addWidget(self.overview_canvas)

        self.tabs.addTab(tab, "Overview")

    def create_pca_tab(self):
        """PCA analysis tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        controls = QGroupBox("PCA Settings")
        controls_layout = QGridLayout()

        controls_layout.addWidget(QLabel("Year:"), 0, 0)
        self.pca_year = QSpinBox()
        self.pca_year.setRange(self.df['year'].min(), self.df['year'].max())
        self.pca_year.setValue(self.df['year'].max())
        controls_layout.addWidget(self.pca_year, 0, 1)

        controls_layout.addWidget(QLabel("Components:"), 1, 0)
        self.pca_n_components = QSpinBox()
        self.pca_n_components.setRange(2, 10)
        self.pca_n_components.setValue(3)
        controls_layout.addWidget(self.pca_n_components, 1, 1)

        run_pca_btn = QPushButton("Run PCA")
        run_pca_btn.clicked.connect(self.run_pca_analysis)
        controls_layout.addWidget(run_pca_btn, 2, 0, 1, 2)

        controls.setLayout(controls_layout)
        left_layout.addWidget(controls)

        # Visualization options
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        biplot_btn = QPushButton("Biplot (PC1 vs PC2)")
        biplot_btn.clicked.connect(self.plot_pca_biplot)
        viz_layout.addWidget(biplot_btn)

        scree_btn = QPushButton("Scree Plot")
        scree_btn.clicked.connect(self.plot_pca_scree)
        viz_layout.addWidget(scree_btn)

        loadings_btn = QPushButton("Loadings Heatmap")
        loadings_btn.clicked.connect(self.plot_pca_loadings)
        viz_layout.addWidget(loadings_btn)

        viz_group.setLayout(viz_layout)
        left_layout.addWidget(viz_group)

        # Results text
        self.pca_results_text = QTextEdit()
        self.pca_results_text.setReadOnly(True)
        left_layout.addWidget(QLabel("Results:"))
        left_layout.addWidget(self.pca_results_text)

        left_layout.addStretch()
        left_panel.setMaximumWidth(300)
        layout.addWidget(left_panel)

        # Right panel - canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.pca_canvas = MplCanvas(self, width=8, height=6)
        self.pca_toolbar = NavigationToolbar(self.pca_canvas, self)
        right_layout.addWidget(self.pca_toolbar)
        right_layout.addWidget(self.pca_canvas)

        layout.addWidget(right_panel)

        self.tabs.addTab(tab, "PCA Analysis")

    def create_cluster_tab(self):
        """Cluster analysis tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        controls = QGroupBox("Clustering Settings")
        controls_layout = QGridLayout()

        controls_layout.addWidget(QLabel("Year:"), 0, 0)
        self.cluster_year = QSpinBox()
        self.cluster_year.setRange(self.df['year'].min(), self.df['year'].max())
        self.cluster_year.setValue(self.df['year'].max())
        controls_layout.addWidget(self.cluster_year, 0, 1)

        controls_layout.addWidget(QLabel("Method:"), 1, 0)
        self.cluster_method = QComboBox()
        self.cluster_method.addItems(['kmeans', 'hierarchical'])
        controls_layout.addWidget(self.cluster_method, 1, 1)

        controls_layout.addWidget(QLabel("N Clusters:"), 2, 0)
        self.cluster_n = QSpinBox()
        self.cluster_n.setRange(2, 10)
        self.cluster_n.setValue(4)
        controls_layout.addWidget(self.cluster_n, 2, 1)

        run_cluster_btn = QPushButton("Run Clustering")
        run_cluster_btn.clicked.connect(self.run_cluster_analysis)
        controls_layout.addWidget(run_cluster_btn, 3, 0, 1, 2)

        controls.setLayout(controls_layout)
        left_layout.addWidget(controls)

        # Visualization
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        scatter_btn = QPushButton("Cluster Scatter (PCA)")
        scatter_btn.clicked.connect(self.plot_cluster_scatter)
        viz_layout.addWidget(scatter_btn)

        dendrogram_btn = QPushButton("Dendrogram")
        dendrogram_btn.clicked.connect(self.plot_dendrogram)
        viz_layout.addWidget(dendrogram_btn)

        profile_btn = QPushButton("Cluster Profiles")
        profile_btn.clicked.connect(self.plot_cluster_profiles)
        viz_layout.addWidget(profile_btn)

        viz_group.setLayout(viz_layout)
        left_layout.addWidget(viz_group)

        # Results
        self.cluster_results_text = QTextEdit()
        self.cluster_results_text.setReadOnly(True)
        left_layout.addWidget(QLabel("Results:"))
        left_layout.addWidget(self.cluster_results_text)

        left_layout.addStretch()
        left_panel.setMaximumWidth(300)
        layout.addWidget(left_panel)

        # Canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.cluster_canvas = MplCanvas(self, width=8, height=6)
        self.cluster_toolbar = NavigationToolbar(self.cluster_canvas, self)
        right_layout.addWidget(self.cluster_toolbar)
        right_layout.addWidget(self.cluster_canvas)

        layout.addWidget(right_panel)

        self.tabs.addTab(tab, "Cluster Analysis")

    def create_complementarity_tab(self):
        """Institutional complementarity testing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls = QGroupBox("Complementarity Analysis")
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Year:"))
        self.comp_year = QSpinBox()
        self.comp_year.setRange(self.df['year'].min(), self.df['year'].max())
        self.comp_year.setValue(self.df['year'].max())
        controls_layout.addWidget(self.comp_year)

        run_comp_btn = QPushButton("Test Complementarities")
        run_comp_btn.clicked.connect(self.run_complementarity_analysis)
        controls_layout.addWidget(run_comp_btn)

        plot_btn = QPushButton("Plot Correlation Matrix")
        plot_btn.clicked.connect(self.plot_complementarity_matrix)
        controls_layout.addWidget(plot_btn)

        controls_layout.addStretch()
        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        # Results table
        layout.addWidget(QLabel("Complementarity Test Results:"))
        self.comp_table = QTableWidget()
        layout.addWidget(self.comp_table)

        # Canvas
        self.comp_canvas = MplCanvas(self, width=10, height=6)
        self.comp_toolbar = NavigationToolbar(self.comp_canvas, self)
        layout.addWidget(self.comp_toolbar)
        layout.addWidget(self.comp_canvas)

        self.tabs.addTab(tab, "Complementarities")

    def create_trajectory_tab(self):
        """Historical trajectory analysis tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        controls = QGroupBox("Trajectory Settings")
        controls_layout = QVBoxLayout()

        controls_layout.addWidget(QLabel("Select Country:"))
        self.traj_country = QComboBox()
        self.traj_country.addItems(sorted(self.df['country'].unique()))
        controls_layout.addWidget(self.traj_country)

        controls_layout.addWidget(QLabel("Select Indicators:"))
        self.traj_indicators = QListWidget()
        self.traj_indicators.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        indicators = [
            'neoliberalism_index', 'financialization_index',
            'power_resources_index', 'wage_share_gdp',
            'union_density', 'welfare_generosity'
        ]
        self.traj_indicators.addItems(indicators)
        for i in range(4):  # Select first 4 by default
            self.traj_indicators.item(i).setSelected(True)
        controls_layout.addWidget(self.traj_indicators)

        run_traj_btn = QPushButton("Analyze Trajectory")
        run_traj_btn.clicked.connect(self.run_trajectory_analysis)
        controls_layout.addWidget(run_traj_btn)

        controls.setLayout(controls_layout)
        left_layout.addWidget(controls)

        # Results
        self.traj_results_text = QTextEdit()
        self.traj_results_text.setReadOnly(True)
        left_layout.addWidget(QLabel("Trend Analysis:"))
        left_layout.addWidget(self.traj_results_text)

        left_panel.setMaximumWidth(300)
        layout.addWidget(left_panel)

        # Canvas
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.traj_canvas = MplCanvas(self, width=8, height=6)
        self.traj_toolbar = NavigationToolbar(self.traj_canvas, self)
        right_layout.addWidget(self.traj_toolbar)
        right_layout.addWidget(self.traj_canvas)

        layout.addWidget(right_panel)

        self.tabs.addTab(tab, "Historical Trajectories")

    def create_transitions_tab(self):
        """Regime transitions detection tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls = QGroupBox("Transition Detection")
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Country:"))
        self.trans_country = QComboBox()
        self.trans_country.addItem("All Countries")
        self.trans_country.addItems(sorted(self.df['country'].unique()))
        controls_layout.addWidget(self.trans_country)

        controls_layout.addWidget(QLabel("Threshold:"))
        self.trans_threshold = QSpinBox()
        self.trans_threshold.setRange(5, 50)
        self.trans_threshold.setValue(15)
        self.trans_threshold.setSuffix("%")
        controls_layout.addWidget(self.trans_threshold)

        detect_btn = QPushButton("Detect Transitions")
        detect_btn.clicked.connect(self.detect_transitions)
        controls_layout.addWidget(detect_btn)

        controls_layout.addStretch()
        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        # Results table
        layout.addWidget(QLabel("Detected Regime Transitions:"))
        self.trans_table = QTableWidget()
        layout.addWidget(self.trans_table)

        self.tabs.addTab(tab, "Regime Transitions")

    def create_regulation_school_tab(self):
        """Regulation School periodization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        controls = QGroupBox("Regulation School Analysis")
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Select Countries:"))
        self.reg_countries = QListWidget()
        self.reg_countries.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        countries = ['USA', 'Germany', 'France', 'United Kingdom', 'Japan', 'China']
        self.reg_countries.addItems(countries)
        for i in range(len(countries)):
            self.reg_countries.item(i).setSelected(True)
        controls_layout.addWidget(self.reg_countries)

        plot_btn = QPushButton("Plot Accumulation Regimes")
        plot_btn.clicked.connect(self.plot_regulation_regimes)
        controls_layout.addWidget(plot_btn)

        controls.setLayout(controls_layout)
        layout.addWidget(controls)

        # Canvas
        self.reg_canvas = MplCanvas(self, width=10, height=6)
        self.reg_toolbar = NavigationToolbar(self.reg_canvas, self)
        layout.addWidget(self.reg_toolbar)
        layout.addWidget(self.reg_canvas)

        self.tabs.addTab(tab, "Regulation School")

    # === Analysis Methods ===

    def plot_overview_cross_section(self):
        """Plot cross-sectional comparison"""
        year = self.overview_year.value()
        regime = self.overview_regime.currentText()
        indicator = self.overview_indicator.currentText()

        data = self.df[self.df['year'] == year].copy()
        if regime != "All":
            data = data[data['regime_type'] == regime]

        self.overview_canvas.axes.clear()
        data_sorted = data.sort_values(indicator, ascending=False)

        colors = {'LME': 'red', 'CME': 'blue', 'MME': 'green',
                 'Transition': 'orange', 'EAsia': 'purple',
                 'LatAm': 'brown', 'Statist': 'pink', 'Developing': 'gray'}

        bar_colors = [colors.get(r, 'black') for r in data_sorted['regime_type']]

        self.overview_canvas.axes.barh(range(len(data_sorted)), data_sorted[indicator],
                                       color=bar_colors)
        self.overview_canvas.axes.set_yticks(range(len(data_sorted)))
        self.overview_canvas.axes.set_yticklabels(data_sorted['country'], fontsize=8)
        self.overview_canvas.axes.set_xlabel(indicator.replace('_', ' ').title())
        self.overview_canvas.axes.set_title(f'{indicator} by Country ({year})')
        self.overview_canvas.axes.invert_yaxis()

        self.overview_canvas.fig.tight_layout()
        self.overview_canvas.draw()

    def plot_overview_timeseries(self):
        """Plot time series by regime type"""
        indicator = self.overview_indicator.currentText()

        self.overview_canvas.axes.clear()

        for regime in self.df['regime_type'].unique():
            regime_data = self.df[self.df['regime_type'] == regime]
            regime_avg = regime_data.groupby('year')[indicator].mean()

            self.overview_canvas.axes.plot(regime_avg.index, regime_avg.values,
                                          label=regime, linewidth=2)

        self.overview_canvas.axes.set_xlabel('Year')
        self.overview_canvas.axes.set_ylabel(indicator.replace('_', ' ').title())
        self.overview_canvas.axes.set_title(f'{indicator} Evolution by Regime Type')
        self.overview_canvas.axes.legend()
        self.overview_canvas.axes.grid(True, alpha=0.3)

        self.overview_canvas.fig.tight_layout()
        self.overview_canvas.draw()

    def run_pca_analysis(self):
        """Run PCA analysis"""
        year = self.pca_year.value()
        n_comp = self.pca_n_components.value()

        self.pca_results = self.analyzer.perform_pca(year=year, n_components=n_comp)

        # Display results
        text = f"PCA Results ({year})\n"
        text += "=" * 50 + "\n\n"
        text += f"Number of components: {self.pca_results['n_components']}\n\n"
        text += "Explained Variance:\n"
        for i, var in enumerate(self.pca_results['explained_variance']):
            text += f"  PC{i+1}: {var:.3f} ({var*100:.1f}%)\n"
        text += f"\nCumulative: {self.pca_results['cumulative_variance'][-1]:.3f} "
        text += f"({self.pca_results['cumulative_variance'][-1]*100:.1f}%)\n\n"

        text += "Top Loadings on PC1:\n"
        top_loadings = self.pca_results['loadings']['PC1'].abs().sort_values(ascending=False).head(5)
        for ind, val in top_loadings.items():
            text += f"  {ind}: {val:.3f}\n"

        self.pca_results_text.setText(text)

        # Auto-plot biplot
        self.plot_pca_biplot()

    def plot_pca_biplot(self):
        """Plot PCA biplot"""
        if self.pca_results is None:
            QMessageBox.warning(self, "Warning", "Run PCA analysis first")
            return

        self.pca_canvas.axes.clear()

        pca_scores = self.pca_results['pca_scores']
        loadings = self.pca_results['loadings']

        # Plot scores colored by regime
        colors = {'LME': 'red', 'CME': 'blue', 'MME': 'green',
                 'Transition': 'orange', 'EAsia': 'purple',
                 'LatAm': 'brown', 'Statist': 'pink', 'Developing': 'gray'}

        for regime in pca_scores['regime_type'].unique():
            regime_data = pca_scores[pca_scores['regime_type'] == regime]
            self.pca_canvas.axes.scatter(regime_data['PC1'], regime_data['PC2'],
                                        c=colors.get(regime, 'black'),
                                        label=regime, alpha=0.6, s=100)

        # Plot loadings as arrows
        scale_factor = 3
        for i, (idx, row) in enumerate(loadings.head(10).iterrows()):
            self.pca_canvas.axes.arrow(0, 0, row['PC1']*scale_factor,
                                      row['PC2']*scale_factor,
                                      head_width=0.1, head_length=0.1,
                                      fc='black', ec='black', alpha=0.5)
            self.pca_canvas.axes.text(row['PC1']*scale_factor*1.1,
                                     row['PC2']*scale_factor*1.1,
                                     idx, fontsize=8)

        self.pca_canvas.axes.set_xlabel(f"PC1 ({self.pca_results['explained_variance'][0]:.1%})")
        self.pca_canvas.axes.set_ylabel(f"PC2 ({self.pca_results['explained_variance'][1]:.1%})")
        self.pca_canvas.axes.set_title("PCA Biplot")
        self.pca_canvas.axes.legend()
        self.pca_canvas.axes.grid(True, alpha=0.3)

        self.pca_canvas.fig.tight_layout()
        self.pca_canvas.draw()

    def plot_pca_scree(self):
        """Plot scree plot"""
        if self.pca_results is None:
            QMessageBox.warning(self, "Warning", "Run PCA analysis first")
            return

        self.pca_canvas.axes.clear()

        variance = self.pca_results['explained_variance']
        cumvar = self.pca_results['cumulative_variance']

        x = range(1, len(variance) + 1)

        self.pca_canvas.axes.bar(x, variance, alpha=0.6, label='Individual')
        self.pca_canvas.axes.plot(x, cumvar, 'ro-', label='Cumulative')
        self.pca_canvas.axes.axhline(y=0.9, color='g', linestyle='--',
                                    label='90% threshold')

        self.pca_canvas.axes.set_xlabel('Principal Component')
        self.pca_canvas.axes.set_ylabel('Explained Variance')
        self.pca_canvas.axes.set_title('Scree Plot')
        self.pca_canvas.axes.legend()
        self.pca_canvas.axes.grid(True, alpha=0.3)

        self.pca_canvas.fig.tight_layout()
        self.pca_canvas.draw()

    def plot_pca_loadings(self):
        """Plot loadings heatmap"""
        if self.pca_results is None:
            QMessageBox.warning(self, "Warning", "Run PCA analysis first")
            return

        self.pca_canvas.axes.clear()

        loadings = self.pca_results['loadings']

        sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, ax=self.pca_canvas.axes, cbar_kws={'shrink': 0.8})

        self.pca_canvas.axes.set_title('PCA Loadings Heatmap')
        self.pca_canvas.axes.set_ylabel('Indicator')
        self.pca_canvas.axes.set_xlabel('Principal Component')

        self.pca_canvas.fig.tight_layout()
        self.pca_canvas.draw()

    def run_cluster_analysis(self):
        """Run cluster analysis"""
        year = self.cluster_year.value()
        method = self.cluster_method.currentText()
        n_clusters = self.cluster_n.value()

        self.cluster_results = self.analyzer.cluster_analysis(
            year=year, method=method, n_clusters=n_clusters, data_source='pca'
        )

        # Display results
        text = f"Cluster Analysis Results ({year})\n"
        text += "=" * 50 + "\n\n"
        text += f"Method: {method}\n"
        text += f"Number of clusters: {self.cluster_results['n_clusters']}\n"
        text += f"Silhouette score: {self.cluster_results['silhouette_score']:.3f}\n"
        text += f"Calinski-Harabasz: {self.cluster_results['calinski_harabasz_score']:.1f}\n\n"

        text += "Cluster Sizes:\n"
        sizes = self.cluster_results['cluster_assignments']['cluster'].value_counts().sort_index()
        for cluster, size in sizes.items():
            text += f"  Cluster {cluster}: {size} countries\n"

        text += "\nCountries by Cluster:\n"
        for cluster in sorted(self.cluster_results['cluster_assignments']['cluster'].unique()):
            countries = self.cluster_results['cluster_assignments'][
                self.cluster_results['cluster_assignments']['cluster'] == cluster
            ]['country'].tolist()
            text += f"\nCluster {cluster}:\n"
            text += "  " + ", ".join(countries) + "\n"

        self.cluster_results_text.setText(text)

        # Auto-plot
        self.plot_cluster_scatter()

    def plot_cluster_scatter(self):
        """Plot cluster scatter"""
        if self.cluster_results is None:
            QMessageBox.warning(self, "Warning", "Run cluster analysis first")
            return

        if self.pca_results is None:
            self.run_pca_analysis()

        self.cluster_canvas.axes.clear()

        pca_scores = self.pca_results['pca_scores']
        labels = self.cluster_results['labels']

        scatter = self.cluster_canvas.axes.scatter(
            pca_scores['PC1'], pca_scores['PC2'],
            c=labels, cmap='viridis', s=100, alpha=0.6
        )

        # Add country labels
        for idx, row in pca_scores.iterrows():
            self.cluster_canvas.axes.annotate(
                row['country'][:3],
                (row['PC1'], row['PC2']),
                fontsize=7, alpha=0.7
            )

        self.cluster_canvas.axes.set_xlabel(f"PC1 ({self.pca_results['explained_variance'][0]:.1%})")
        self.cluster_canvas.axes.set_ylabel(f"PC2 ({self.pca_results['explained_variance'][1]:.1%})")
        self.cluster_canvas.axes.set_title('Cluster Analysis (PCA Space)')
        self.cluster_canvas.fig.colorbar(scatter, ax=self.cluster_canvas.axes, label='Cluster')
        self.cluster_canvas.axes.grid(True, alpha=0.3)

        self.cluster_canvas.fig.tight_layout()
        self.cluster_canvas.draw()

    def plot_dendrogram(self):
        """Plot hierarchical clustering dendrogram"""
        if self.pca_results is None:
            self.run_pca_analysis()

        from scipy.cluster import hierarchy

        self.cluster_canvas.axes.clear()

        # Get PCA scores
        pca_scores = self.pca_results['pca_scores']
        X = pca_scores[[c for c in pca_scores.columns if c.startswith('PC')]].values
        labels = pca_scores['country'].values

        # Compute linkage
        Z = hierarchy.linkage(X, method='ward')

        # Plot dendrogram
        hierarchy.dendrogram(Z, labels=labels, ax=self.cluster_canvas.axes,
                           leaf_rotation=90, leaf_font_size=8)

        self.cluster_canvas.axes.set_title('Hierarchical Clustering Dendrogram')
        self.cluster_canvas.axes.set_ylabel('Distance')

        self.cluster_canvas.fig.tight_layout()
        self.cluster_canvas.draw()

    def plot_cluster_profiles(self):
        """Plot cluster profiles"""
        if self.cluster_results is None:
            QMessageBox.warning(self, "Warning", "Run cluster analysis first")
            return

        self.cluster_canvas.axes.clear()

        profiles = self.cluster_results['cluster_profiles']

        # Select subset of indicators for readability
        indicators = ['labor_market_coordination', 'union_density', 'welfare_generosity',
                     'financialization_index', 'neoliberalism_index']

        # Get data for selected indicators if available
        available = [ind for ind in indicators if ind in self.analyzer.all_indicators]

        if len(available) == 0:
            # Use PCA components instead
            profiles.plot(kind='bar', ax=self.cluster_canvas.axes)
        else:
            # Need to recompute cluster profiles with raw indicators
            year = self.cluster_year.value()
            temp_results = self.analyzer.cluster_analysis(
                year=year, method=self.cluster_method.currentText(),
                n_clusters=self.cluster_n.value(), data_source='raw',
                indicators=available
            )

            temp_results['cluster_profiles'].plot(kind='bar', ax=self.cluster_canvas.axes)

        self.cluster_canvas.axes.set_title('Cluster Profiles')
        self.cluster_canvas.axes.set_xlabel('Cluster')
        self.cluster_canvas.axes.set_ylabel('Standardized Value')
        self.cluster_canvas.axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.cluster_canvas.axes.grid(True, alpha=0.3)

        self.cluster_canvas.fig.tight_layout()
        self.cluster_canvas.draw()

    def run_complementarity_analysis(self):
        """Run complementarity testing"""
        year = self.comp_year.value()

        comp_results = self.analyzer.test_complementarities(year=year)

        # Display in table
        self.comp_table.setRowCount(len(comp_results))
        self.comp_table.setColumnCount(4)
        self.comp_table.setHorizontalHeaderLabels([
            'Institution 1', 'Institution 2', 'Correlation', 'P-value'
        ])

        for i, row in comp_results.iterrows():
            self.comp_table.setItem(i, 0, QTableWidgetItem(row['institution_1']))
            self.comp_table.setItem(i, 1, QTableWidgetItem(row['institution_2']))
            self.comp_table.setItem(i, 2, QTableWidgetItem(f"{row['overall_correlation']:.3f}"))
            self.comp_table.setItem(i, 3, QTableWidgetItem(f"{row['p_value']:.4f}"))

        self.comp_table.resizeColumnsToContents()

    def plot_complementarity_matrix(self):
        """Plot correlation matrix"""
        year = self.comp_year.value()
        data = self.df[self.df['year'] == year]

        # Select key indicators
        indicators = [
            'labor_market_coordination', 'union_density', 'employment_protection',
            'vocational_training', 'stakeholder_governance',
            'welfare_generosity', 'financialization_index',
            'neoliberalism_index', 'wage_share_gdp'
        ]

        corr_matrix = data[indicators].corr()

        self.comp_canvas.axes.clear()

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, ax=self.comp_canvas.axes,
                   square=True, linewidths=0.5)

        self.comp_canvas.axes.set_title(f'Institutional Correlation Matrix ({year})')

        self.comp_canvas.fig.tight_layout()
        self.comp_canvas.draw()

    def run_trajectory_analysis(self):
        """Run trajectory analysis for selected country"""
        country = self.traj_country.currentText()
        selected_items = self.traj_indicators.selectedItems()
        indicators = [item.text() for item in selected_items]

        if not indicators:
            QMessageBox.warning(self, "Warning", "Select at least one indicator")
            return

        traj_results = self.analyzer.trajectory_analysis(country, indicators)

        # Display trends
        text = f"Trajectory Analysis: {country}\n"
        text += f"Regime Type: {traj_results['regime']}\n"
        text += "=" * 50 + "\n\n"

        text += "Trends:\n"
        for ind, trend in traj_results['trends'].items():
            text += f"\n{ind}:\n"
            text += f"  Direction: {trend['direction']}\n"
            text += f"  Slope: {trend['slope']:.4f}\n"
            text += f"  R²: {trend['r_squared']:.3f}\n"
            text += f"  Strength: {trend['strength']:.3f}\n"

        text += "\n\nPath Dependence (autocorrelation):\n"
        for ind, autocorr in traj_results['path_dependence'].items():
            text += f"  {ind}: {autocorr:.3f}\n"

        self.traj_results_text.setText(text)

        # Plot trajectories
        self.traj_canvas.axes.clear()

        for indicator in indicators:
            years = traj_results['trajectories'][indicator]['years']
            values = traj_results['trajectories'][indicator]['values']

            self.traj_canvas.axes.plot(years, values, marker='o',
                                      label=indicator, linewidth=2)

        self.traj_canvas.axes.set_xlabel('Year')
        self.traj_canvas.axes.set_ylabel('Value')
        self.traj_canvas.axes.set_title(f'Historical Trajectory: {country}')
        self.traj_canvas.axes.legend()
        self.traj_canvas.axes.grid(True, alpha=0.3)

        self.traj_canvas.fig.tight_layout()
        self.traj_canvas.draw()

    def detect_transitions(self):
        """Detect regime transitions"""
        country_sel = self.trans_country.currentText()
        threshold = self.trans_threshold.value() / 100.0

        country = None if country_sel == "All Countries" else country_sel

        transitions = self.analyzer.detect_regime_transitions(
            country=country, threshold=threshold
        )

        # Display in table
        if len(transitions) == 0:
            QMessageBox.information(self, "Info", "No transitions detected with current threshold")
            return

        self.trans_table.setRowCount(len(transitions))
        self.trans_table.setColumnCount(6)
        self.trans_table.setHorizontalHeaderLabels([
            'Country', 'Year', 'Regime', 'Magnitude', 'Direction',
            'Neoliberalism Change'
        ])

        for i, row in transitions.iterrows():
            self.trans_table.setItem(i, 0, QTableWidgetItem(row['country']))
            self.trans_table.setItem(i, 1, QTableWidgetItem(str(int(row['year']))))
            self.trans_table.setItem(i, 2, QTableWidgetItem(row['regime_type']))
            self.trans_table.setItem(i, 3, QTableWidgetItem(f"{row['magnitude']:.3f}"))
            self.trans_table.setItem(i, 4, QTableWidgetItem(row['direction']))
            self.trans_table.setItem(i, 5, QTableWidgetItem(f"{row['neoliberalism_change']:.3f}"))

        self.trans_table.resizeColumnsToContents()

    def plot_regulation_regimes(self):
        """Plot regulation school regimes over time"""
        selected_items = self.reg_countries.selectedItems()
        countries = [item.text() for item in selected_items]

        if not countries:
            QMessageBox.warning(self, "Warning", "Select at least one country")
            return

        reg_scores = self.reg_analyzer.regime_of_accumulation_scores()
        data = reg_scores[reg_scores['country'].isin(countries)]

        self.reg_canvas.axes.clear()

        # Create subplots for each country
        n_countries = len(countries)
        fig = self.reg_canvas.fig
        fig.clear()

        for i, country in enumerate(countries):
            ax = fig.add_subplot(2, 3, i+1)

            country_data = data[data['country'] == country]

            ax.plot(country_data['year'], country_data['fordist_score'],
                   label='Fordist', linewidth=2)
            ax.plot(country_data['year'], country_data['finance_led_score'],
                   label='Finance-led', linewidth=2)
            ax.plot(country_data['year'], country_data['export_led_score'],
                   label='Export-led', linewidth=2)

            ax.set_title(country, fontsize=10)
            ax.set_xlabel('Year', fontsize=8)
            ax.set_ylabel('Score', fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Regime of Accumulation Evolution', fontsize=12)
        fig.tight_layout()
        self.reg_canvas.draw()


def main():
    """Run application"""
    app = QApplication(sys.argv)
    window = PoliticalEconomyApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
