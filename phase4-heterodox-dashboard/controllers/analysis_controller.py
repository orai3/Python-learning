"""
Analysis Controller

Manages economic analysis using different theoretical frameworks.
"""

from models.data_model import DataModel
from models.frameworks import FrameworkManager, EconomicFramework
from utils.visualizations import ChartGenerator
from utils.calculations import EconomicCalculations
from typing import Dict, List, Optional, Tuple
import pandas as pd
from matplotlib.figure import Figure


class AnalysisController:
    """
    Controller for economic analysis operations.
    Coordinates framework analysis and visualization.
    """

    def __init__(self, data_model: DataModel):
        """
        Initialize analysis controller.

        Args:
            data_model: DataModel instance with loaded data
        """
        self.data_model = data_model
        self.framework_manager = FrameworkManager()
        self.chart_generator = ChartGenerator()
        self.calculations = EconomicCalculations()

    def get_available_frameworks(self) -> List[str]:
        """Get list of available theoretical frameworks."""
        return self.framework_manager.get_available_frameworks()

    def analyze_with_framework(self, framework_name: str) -> Dict:
        """
        Perform analysis using a specific framework.

        Args:
            framework_name: Name of the framework to use

        Returns:
            Dictionary with analysis results
        """
        framework = self.framework_manager.get_framework(framework_name)

        if framework is None:
            raise ValueError(f"Unknown framework: {framework_name}")

        return framework.analyze(self.data_model)

    def analyze_all_frameworks(self) -> Dict:
        """
        Perform analysis using all frameworks.

        Returns:
            Dictionary with results from each framework
        """
        return self.framework_manager.analyze_all(self.data_model)

    def get_framework_theory(self, framework_name: str) -> str:
        """
        Get theoretical background for a framework.

        Args:
            framework_name: Name of the framework

        Returns:
            Theoretical notes as string
        """
        framework = self.framework_manager.get_framework(framework_name)

        if framework is None:
            raise ValueError(f"Unknown framework: {framework_name}")

        return framework.get_theoretical_notes()

    def create_time_series_chart(self, dataset: str, variables: List[str],
                                 title: str, ylabel: str) -> Figure:
        """
        Create time series visualization.

        Args:
            dataset: Dataset name
            variables: List of variables to plot
            title: Chart title
            ylabel: Y-axis label

        Returns:
            Matplotlib Figure
        """
        df = self.data_model.get_dataset(dataset)

        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        # Ensure we have a time index
        date_col = None
        for col in ['date', 'quarter', 'year', 'period']:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df = df.set_index(date_col)

        return self.chart_generator.plot_time_series(df, variables, title, ylabel)

    def create_sectoral_balances_chart(self, dataset: str = 'sfc') -> Figure:
        """
        Create sectoral balances visualization (Godley approach).

        Args:
            dataset: Dataset with sectoral balance data

        Returns:
            Matplotlib Figure
        """
        df = self.data_model.get_dataset(dataset)

        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        # Set time index
        date_col = None
        for col in ['date', 'quarter', 'year', 'period']:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df = df.set_index(date_col)

        return self.chart_generator.plot_sectoral_balances(df)

    def create_distribution_chart(self, dataset: str, variable: str,
                                  title: str, xlabel: str) -> Figure:
        """
        Create distribution histogram.

        Args:
            dataset: Dataset name
            variable: Variable to plot
            title: Chart title
            xlabel: X-axis label

        Returns:
            Matplotlib Figure
        """
        df = self.data_model.get_dataset(dataset)

        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        if variable not in df.columns:
            raise ValueError(f"Variable {variable} not found")

        return self.chart_generator.plot_distribution(df[variable], title, xlabel)

    def create_lorenz_curve(self, dataset: str, variable: str) -> Figure:
        """
        Create Lorenz curve for inequality analysis.

        Args:
            dataset: Dataset name
            variable: Income/wealth variable

        Returns:
            Matplotlib Figure
        """
        df = self.data_model.get_dataset(dataset)

        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        if variable not in df.columns:
            raise ValueError(f"Variable {variable} not found")

        return self.chart_generator.plot_lorenz_curve(df[variable])

    def create_wage_profit_chart(self, dataset: str = 'macro') -> Figure:
        """
        Create wage and profit share visualization.

        Args:
            dataset: Dataset name

        Returns:
            Matplotlib Figure
        """
        wage_share = self.data_model.calculate_wage_share(dataset)
        profit_share = self.data_model.calculate_profit_share(dataset)

        return self.chart_generator.plot_wage_profit_share(wage_share, profit_share)

    def create_correlation_matrix(self, dataset: str,
                                  variables: Optional[List[str]] = None) -> Figure:
        """
        Create correlation matrix heatmap.

        Args:
            dataset: Dataset name
            variables: Optional list of variables

        Returns:
            Matplotlib Figure
        """
        corr_matrix = self.data_model.get_correlation_matrix(dataset, variables)

        return self.chart_generator.plot_correlation_matrix(corr_matrix)

    def create_rate_of_profit_chart(self, dataset: str = 'macro') -> Figure:
        """
        Create rate of profit visualization (Marxian analysis).

        Args:
            dataset: Dataset name

        Returns:
            Matplotlib Figure
        """
        df = self.data_model.get_dataset(dataset)

        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        if 'profits' not in df.columns or 'capital_stock' not in df.columns:
            raise ValueError("Required variables (profits, capital_stock) not found")

        rate_of_profit = self.calculations.calculate_rate_of_profit(
            df['profits'], df['capital_stock']
        )

        return self.chart_generator.plot_rate_of_profit(rate_of_profit)

    def create_comparative_chart(self, results: Dict, indicator: str) -> Figure:
        """
        Create comparative visualization across frameworks.

        Args:
            results: Analysis results from multiple frameworks
            indicator: Indicator to compare

        Returns:
            Matplotlib Figure
        """
        return self.chart_generator.plot_comparative_frameworks(results, indicator)

    def calculate_inequality_measures(self, dataset: str, variable: str) -> Dict:
        """
        Calculate multiple inequality measures.

        Args:
            dataset: Dataset name
            variable: Income/wealth variable

        Returns:
            Dictionary with inequality measures
        """
        df = self.data_model.get_dataset(dataset)

        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        if variable not in df.columns:
            raise ValueError(f"Variable {variable} not found")

        data = df[variable].dropna().values

        results = {
            'gini': self.calculations.calculate_gini(data),
            'palma_ratio': self.calculations.calculate_palma_ratio(data),
            'theil_index': self.calculations.calculate_theil_index(data)
        }

        return results

    def generate_report(self, framework_name: str) -> str:
        """
        Generate text report for a framework analysis.

        Args:
            framework_name: Name of the framework

        Returns:
            Formatted report as string
        """
        analysis = self.analyze_with_framework(framework_name)
        framework = self.framework_manager.get_framework(framework_name)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"{framework.get_name()} Economic Analysis")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Add key indicators
        if 'indicators' in analysis and analysis['indicators']:
            report_lines.append("KEY INDICATORS:")
            report_lines.append("-" * 80)
            for key, value in analysis['indicators'].items():
                if value is not None:
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            report_lines.append("")

        # Add interpretations
        if 'interpretation' in analysis and analysis['interpretation']:
            report_lines.append("INTERPRETATION:")
            report_lines.append("-" * 80)
            for i, interp in enumerate(analysis['interpretation'], 1):
                report_lines.append(f"  {i}. {interp}")
            report_lines.append("")

        # Add distributional analysis for PK
        if 'distributional_analysis' in analysis and analysis['distributional_analysis']:
            report_lines.append("DISTRIBUTIONAL ANALYSIS:")
            report_lines.append("-" * 80)
            for key, value in analysis['distributional_analysis'].items():
                if value is not None and not isinstance(value, dict):
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            report_lines.append("")

        # Add sectoral balances for PK
        if 'sectoral_balances' in analysis and analysis['sectoral_balances']:
            report_lines.append("SECTORAL BALANCES (Godley):")
            report_lines.append("-" * 80)
            for key, value in analysis['sectoral_balances'].items():
                if value is not None and key != 'identity_check':
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            if 'identity_check' in analysis['sectoral_balances']:
                status = "✓ VERIFIED" if analysis['sectoral_balances']['identity_check'] else "✗ VIOLATION"
                report_lines.append(f"  Accounting Identity: {status}")
            report_lines.append("")

        # Add theoretical background
        report_lines.append("THEORETICAL BACKGROUND:")
        report_lines.append("-" * 80)
        report_lines.append(framework.get_theoretical_notes())
        report_lines.append("")

        report_lines.append("=" * 80)

        return "\n".join(report_lines)

    def export_analysis_results(self, framework_name: str, file_path: str) -> bool:
        """
        Export analysis results to file.

        Args:
            framework_name: Name of the framework
            file_path: Output file path

        Returns:
            True if successful
        """
        try:
            report = self.generate_report(framework_name)

            with open(file_path, 'w') as f:
                f.write(report)

            return True

        except Exception as e:
            print(f"Error exporting analysis: {str(e)}")
            return False
