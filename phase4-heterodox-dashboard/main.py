#!/usr/bin/env python3
"""
Heterodox Macro Dashboard - Main Application Entry Point

A production-ready PyQt6 application for academic economic research using
heterodox theoretical frameworks.

Author: Claude Code
Date: 2025-11-19
License: MIT

Usage:
    python main.py

Requirements:
    - Python 3.8+
    - PyQt6
    - pandas, numpy, matplotlib
    - See requirements.txt for full list
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from views.main_window import MainWindow


def main():
    """
    Main application entry point.
    """
    # Create Qt application
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Heterodox Macro Dashboard")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Heterodox Economics Research")

    # Enable high DPI scaling
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Run application event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)
