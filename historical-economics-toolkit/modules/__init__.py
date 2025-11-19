"""
Historical Economics Toolkit - Main Modules
==========================================

Import all main classes and functions for convenient access.
"""

from .data_generator import HistoricalEconomicDataGenerator, generate_multi_country_dataset
from .periodization import (StructuralBreakDetector, RegimeSwitchingModel,
                            RegulationSchoolPeriodization, detect_all_breaks)
from .long_wave_analysis import (LongWaveAnalyzer, SchumpeterianCycles,
                                 TechnologyRevolutions)
from .crisis_hegemony import (CrisisAnalyzer, HegemonyAnalyzer,
                              analyze_crisis_hegemony_relationship)
from .visualization import HistoricalPlotter, create_summary_dashboard

__all__ = [
    'HistoricalEconomicDataGenerator',
    'generate_multi_country_dataset',
    'StructuralBreakDetector',
    'RegimeSwitchingModel',
    'RegulationSchoolPeriodization',
    'detect_all_breaks',
    'LongWaveAnalyzer',
    'SchumpeterianCycles',
    'TechnologyRevolutions',
    'CrisisAnalyzer',
    'HegemonyAnalyzer',
    'analyze_crisis_hegemony_relationship',
    'HistoricalPlotter',
    'create_summary_dashboard'
]
