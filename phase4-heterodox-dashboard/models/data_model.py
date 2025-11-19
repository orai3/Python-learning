"""
Data Model for Heterodox Macro Dashboard
Handles loading, validation, and management of economic datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class DataModel:
    """
    Central data model for loading and managing economic datasets.
    Supports multiple data sources and provides unified interface for analysis.

    References:
    - Godley, W., & Lavoie, M. (2007). Monetary Economics: An Integrated Approach
      to Credit, Money, Income, Production and Wealth. Palgrave Macmillan.
    """

    def __init__(self):
        """Initialize the data model with empty datasets."""
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        self.data_dir = Path(__file__).parent.parent.parent / "datasets"

    def load_dataset(self, dataset_name: str, file_path: Optional[str] = None) -> bool:
        """
        Load a dataset from CSV file.

        Args:
            dataset_name: Internal name for the dataset
            file_path: Path to CSV file (optional, uses default if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path is None:
                # Use default paths for known datasets
                default_files = {
                    'macro': 'macro_quarterly_data.csv',
                    'inequality': 'inequality_annual_data.csv',
                    'sfc': 'sectoral_balances_data.csv',
                    'crisis': 'financial_crisis_data.csv',
                    'panel': 'cross_country_panel_data.csv',
                    'household': 'household_microdata.csv',
                    'energy': 'energy_environment_data.csv'
                }

                if dataset_name not in default_files:
                    raise ValueError(f"Unknown dataset: {dataset_name}")

                file_path = self.data_dir / default_files[dataset_name]
            else:
                file_path = Path(file_path)

            # Load the dataset
            df = pd.read_csv(file_path)

            # Parse date columns if present
            date_columns = ['date', 'quarter', 'year', 'period']
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        pass

            self.datasets[dataset_name] = df

            # Store metadata
            self.metadata[dataset_name] = {
                'file_path': str(file_path),
                'loaded_at': datetime.now(),
                'rows': len(df),
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }

            return True

        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}")
            return False

    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a loaded dataset by name."""
        return self.datasets.get(name)

    def get_available_datasets(self) -> List[str]:
        """Get list of currently loaded datasets."""
        return list(self.datasets.keys())

    def get_metadata(self, name: str) -> Optional[Dict]:
        """Get metadata for a dataset."""
        return self.metadata.get(name)

    def get_time_series(self, dataset: str, variable: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.Series:
        """
        Extract a time series from a dataset.

        Args:
            dataset: Name of the dataset
            variable: Column name for the variable
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Pandas Series with the time series data
        """
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        if variable not in df.columns:
            raise ValueError(f"Variable {variable} not found in dataset")

        # Find date column
        date_col = None
        for col in ['date', 'quarter', 'year', 'period']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            # Return series without date filtering
            return df[variable]

        # Filter by date if specified
        filtered_df = df.copy()
        if start_date:
            filtered_df = filtered_df[filtered_df[date_col] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df[date_col] <= end_date]

        return filtered_df.set_index(date_col)[variable]

    def calculate_growth_rate(self, dataset: str, variable: str,
                             periods: int = 1) -> pd.Series:
        """
        Calculate growth rates for a variable.

        Args:
            dataset: Name of the dataset
            variable: Variable to calculate growth for
            periods: Number of periods for growth calculation

        Returns:
            Series with growth rates
        """
        series = self.get_time_series(dataset, variable)
        return series.pct_change(periods=periods) * 100

    def calculate_moving_average(self, dataset: str, variable: str,
                                 window: int = 4) -> pd.Series:
        """Calculate moving average for a variable."""
        series = self.get_time_series(dataset, variable)
        return series.rolling(window=window).mean()

    def get_sectoral_balances(self, dataset: str = 'sfc') -> pd.DataFrame:
        """
        Get sectoral balances data following Godley's approach.

        The sectoral balances identity states that the sum of balances
        across all sectors must equal zero (accounting identity).

        Returns:
            DataFrame with sectoral balance columns
        """
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        # Identify balance columns
        balance_cols = [col for col in df.columns if 'balance' in col.lower()]

        if not balance_cols:
            raise ValueError("No balance columns found in dataset")

        return df[balance_cols]

    def get_distribution_data(self, dataset: str = 'inequality') -> pd.DataFrame:
        """
        Get income/wealth distribution data.

        Returns:
            DataFrame with distribution metrics
        """
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        return df

    def calculate_wage_share(self, dataset: str = 'macro') -> pd.Series:
        """
        Calculate wage share of income (key Marxian/PK indicator).

        Wage Share = Compensation of Employees / GDP

        References:
        - Kalecki, M. (1971). Selected Essays on the Dynamics of the
          Capitalist Economy. Cambridge University Press.
        """
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        if 'compensation' in df.columns and 'gdp' in df.columns:
            return (df['compensation'] / df['gdp']) * 100
        elif 'wage_share' in df.columns:
            return df['wage_share']
        else:
            raise ValueError("Cannot calculate wage share from available data")

    def calculate_profit_share(self, dataset: str = 'macro') -> pd.Series:
        """
        Calculate profit share of income.

        Profit Share = 100 - Wage Share
        """
        wage_share = self.calculate_wage_share(dataset)
        return 100 - wage_share

    def get_financial_indicators(self, dataset: str = 'macro') -> pd.DataFrame:
        """
        Get financial sector indicators for Minsky-style analysis.

        References:
        - Minsky, H. (1986). Stabilizing an Unstable Economy. Yale University Press.
        """
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        # Identify financial columns
        financial_cols = []
        keywords = ['debt', 'credit', 'leverage', 'asset', 'financial', 'loan']

        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                financial_cols.append(col)

        if not financial_cols:
            raise ValueError("No financial indicators found")

        return df[financial_cols]

    def export_dataset(self, dataset: str, file_path: str,
                      format: str = 'csv') -> bool:
        """
        Export a dataset to file.

        Args:
            dataset: Name of the dataset
            file_path: Output file path
            format: Export format ('csv', 'excel', 'stata')

        Returns:
            True if successful
        """
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        try:
            if format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'excel':
                df.to_excel(file_path, index=False)
            elif format == 'stata':
                df.to_stata(file_path, write_index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            return True

        except Exception as e:
            print(f"Error exporting dataset: {str(e)}")
            return False

    def get_summary_statistics(self, dataset: str) -> pd.DataFrame:
        """Get summary statistics for all numeric columns in a dataset."""
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        return df.describe()

    def get_correlation_matrix(self, dataset: str,
                              variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for variables.

        Args:
            dataset: Name of the dataset
            variables: List of variables (if None, uses all numeric)

        Returns:
            Correlation matrix
        """
        df = self.datasets.get(dataset)
        if df is None:
            raise ValueError(f"Dataset {dataset} not loaded")

        if variables:
            df = df[variables]
        else:
            df = df.select_dtypes(include=[np.number])

        return df.corr()
