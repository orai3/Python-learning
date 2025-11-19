"""
Advanced Heterodox Economic Research Toolkit

A comprehensive, production-quality Python toolkit for heterodox economic
research and analysis.

Version: 1.0.0
Author: Claude
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Claude"
__license__ = "MIT"

# Import key models for easy access
from .theoretical_models import (
    godley_lavoie_sfc,
    keen_minsky_model,
    sraffa_production,
    goodwin_keen_integration,
    kaleckian_structural
)

from .empirical_frameworks import (
    profit_rate_decomposition,
)

from .historical_models import (
    kalecki_models,
)

__all__ = [
    "godley_lavoie_sfc",
    "keen_minsky_model",
    "sraffa_production",
    "goodwin_keen_integration",
    "kaleckian_structural",
    "profit_rate_decomposition",
    "kalecki_models",
]
