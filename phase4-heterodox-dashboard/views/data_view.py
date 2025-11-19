"""
Data View

PyQt6 interface for data loading and exploration.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QComboBox, QTableWidget, QTableWidgetItem,
                             QTextEdit, QGroupBox, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt
from controllers.data_controller import DataController
from typing import Optional
import pandas as pd


class DataView(QWidget):
    """
    View for data loading and exploration.
    """

    def __init__(self, data_controller: DataController, parent=None):
        """
        Initialize data view.

        Args:
            data_controller: DataController instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.data_controller = data_controller
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Data Management")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        layout.addWidget(title)

        # Dataset selection group
        dataset_group = self.create_dataset_selection_group()
        layout.addWidget(dataset_group)

        # Dataset info group
        info_group = self.create_dataset_info_group()
        layout.addWidget(info_group)

        # Data preview group
        preview_group = self.create_data_preview_group()
        layout.addWidget(preview_group)

        self.setLayout(layout)

    def create_dataset_selection_group(self) -> QGroupBox:
        """Create dataset selection controls."""
        group = QGroupBox("Dataset Selection")
        layout = QVBoxLayout()

        # Load default datasets button
        button_layout = QHBoxLayout()
        load_default_btn = QPushButton("Load Default Datasets")
        load_default_btn.clicked.connect(self.load_default_datasets)
        button_layout.addWidget(load_default_btn)

        # Load custom dataset button
        load_custom_btn = QPushButton("Load Custom CSV")
        load_custom_btn.clicked.connect(self.load_custom_dataset)
        button_layout.addWidget(load_custom_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Dataset dropdown
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Select Dataset:"))

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_selected)
        select_layout.addWidget(self.dataset_combo)

        select_layout.addStretch()
        layout.addLayout(select_layout)

        group.setLayout(layout)
        return group

    def create_dataset_info_group(self) -> QGroupBox:
        """Create dataset information display."""
        group = QGroupBox("Dataset Information")
        layout = QVBoxLayout()

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        layout.addWidget(self.info_text)

        group.setLayout(layout)
        return group

    def create_data_preview_group(self) -> QGroupBox:
        """Create data preview table."""
        group = QGroupBox("Data Preview")
        layout = QVBoxLayout()

        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        layout.addWidget(self.preview_table)

        # Export button
        export_btn = QPushButton("Export Dataset")
        export_btn.clicked.connect(self.export_dataset)
        layout.addWidget(export_btn)

        group.setLayout(layout)
        return group

    def load_default_datasets(self):
        """Load all default datasets."""
        results = self.data_controller.load_default_datasets()

        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        QMessageBox.information(
            self,
            "Load Datasets",
            f"Loaded {success_count} out of {total_count} datasets successfully."
        )

        self.update_dataset_combo()

    def load_custom_dataset(self):
        """Load a custom CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            # Ask for dataset name
            from PyQt6.QtWidgets import QInputDialog

            name, ok = QInputDialog.getText(
                self,
                "Dataset Name",
                "Enter a name for this dataset:"
            )

            if ok and name:
                success = self.data_controller.load_dataset(name, file_path)

                if success:
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Dataset '{name}' loaded successfully."
                    )
                    self.update_dataset_combo()
                else:
                    QMessageBox.warning(
                        self,
                        "Error",
                        f"Failed to load dataset from {file_path}"
                    )

    def update_dataset_combo(self):
        """Update the dataset dropdown with available datasets."""
        current_text = self.dataset_combo.currentText()

        self.dataset_combo.clear()
        datasets = self.data_controller.get_available_datasets()
        self.dataset_combo.addItems(datasets)

        # Try to restore previous selection
        if current_text:
            index = self.dataset_combo.findText(current_text)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)

    def on_dataset_selected(self, dataset_name: str):
        """Handle dataset selection."""
        if not dataset_name:
            return

        self.update_dataset_info(dataset_name)
        self.update_data_preview(dataset_name)

    def update_dataset_info(self, dataset_name: str):
        """Update the dataset information display."""
        info = self.data_controller.get_dataset_info(dataset_name)

        if info:
            info_lines = [
                f"Dataset: {dataset_name}",
                f"Rows: {info['rows']:,}",
                f"Columns: {len(info['columns'])}",
                f"Loaded: {info['loaded_at'].strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Columns:",
                ", ".join(info['columns'])
            ]

            self.info_text.setPlainText("\n".join(info_lines))

    def update_data_preview(self, dataset_name: str):
        """Update the data preview table."""
        data_model = self.data_controller.get_data_model()
        df = data_model.get_dataset(dataset_name)

        if df is None:
            return

        # Show first 100 rows
        preview_df = df.head(100)

        self.preview_table.setRowCount(len(preview_df))
        self.preview_table.setColumnCount(len(preview_df.columns))
        self.preview_table.setHorizontalHeaderLabels(list(preview_df.columns))

        for i in range(len(preview_df)):
            for j, col in enumerate(preview_df.columns):
                value = preview_df.iloc[i, j]

                # Format value
                if pd.isna(value):
                    display_value = "N/A"
                elif isinstance(value, float):
                    display_value = f"{value:.4f}"
                else:
                    display_value = str(value)

                item = QTableWidgetItem(display_value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.preview_table.setItem(i, j, item)

        self.preview_table.resizeColumnsToContents()

    def export_dataset(self):
        """Export the currently selected dataset."""
        dataset_name = self.dataset_combo.currentText()

        if not dataset_name:
            QMessageBox.warning(self, "Warning", "No dataset selected.")
            return

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Dataset",
            f"{dataset_name}.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;Stata Files (*.dta)"
        )

        if file_path:
            # Determine format from filter
            if "Excel" in selected_filter:
                format = 'excel'
            elif "Stata" in selected_filter:
                format = 'stata'
            else:
                format = 'csv'

            success = self.data_controller.export_dataset(dataset_name, file_path, format)

            if success:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Dataset exported to {file_path}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Failed to export dataset"
                )
