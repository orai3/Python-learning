"""
Data Controller

Manages data loading, validation, and transformation operations.
"""

from models.data_model import DataModel
from typing import List, Optional, Dict
import pandas as pd


class DataController:
    """
    Controller for data operations.
    Mediates between data model and views.
    """

    def __init__(self):
        """Initialize data controller with data model."""
        self.data_model = DataModel()
        self.loaded_datasets = []

    def load_default_datasets(self) -> Dict[str, bool]:
        """
        Load all default datasets.

        Returns:
            Dictionary with dataset names and load status
        """
        default_datasets = ['macro', 'inequality', 'sfc', 'crisis', 'panel']
        results = {}

        for dataset in default_datasets:
            try:
                success = self.data_model.load_dataset(dataset)
                results[dataset] = success
                if success:
                    self.loaded_datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {dataset}: {str(e)}")
                results[dataset] = False

        return results

    def load_dataset(self, dataset_name: str, file_path: Optional[str] = None) -> bool:
        """
        Load a specific dataset.

        Args:
            dataset_name: Name for the dataset
            file_path: Optional path to CSV file

        Returns:
            True if successful
        """
        success = self.data_model.load_dataset(dataset_name, file_path)

        if success and dataset_name not in self.loaded_datasets:
            self.loaded_datasets.append(dataset_name)

        return success

    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a loaded dataset."""
        return self.data_model.get_metadata(dataset_name)

    def get_available_datasets(self) -> List[str]:
        """Get list of loaded datasets."""
        return self.loaded_datasets

    def get_dataset_columns(self, dataset_name: str) -> List[str]:
        """Get column names for a dataset."""
        df = self.data_model.get_dataset(dataset_name)
        if df is not None:
            return list(df.columns)
        return []

    def get_time_series(self, dataset: str, variable: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.Series:
        """Get time series data for a variable."""
        return self.data_model.get_time_series(dataset, variable, start_date, end_date)

    def get_summary_statistics(self, dataset: str) -> pd.DataFrame:
        """Get summary statistics for a dataset."""
        return self.data_model.get_summary_statistics(dataset)

    def export_dataset(self, dataset: str, file_path: str, format: str = 'csv') -> bool:
        """Export dataset to file."""
        return self.data_model.export_dataset(dataset, file_path, format)

    def get_data_model(self) -> DataModel:
        """Get the underlying data model."""
        return self.data_model
