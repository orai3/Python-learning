"""
Unequal Exchange Framework
===========================

A comprehensive computational framework for dependency theory and unequal exchange analysis.

This package implements:
- Emmanuel and Amin unequal exchange models
- Global value chain rent extraction calculations
- Terms of trade dynamics
- Transfer pricing estimation
- Intellectual property rent flows
- Super-exploitation metrics
- Multi-country input-output analysis
- Policy simulation tools

Theoretical Foundations:
- Arghiri Emmanuel: Unequal Exchange (1972)
- Samir Amin: Accumulation on a World Scale (1974)
- Andre Gunder Frank: Development of Underdevelopment
- Immanuel Wallerstein: World-Systems Analysis
- Raúl Prebisch & Hans Singer: Terms of Trade deterioration

Author: Python Economics & Modelling
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Python Economics & Modelling"

from .core.theoretical_base import (
    UnequaExchangeModel,
    LaborValueCalculator,
    ValueTransferCalculator
)

from .models.emmanuel import EmmanuelModel
from .models.amin import AminModel
from .models.prebisch_singer import PrebischSingerModel

from .analysis.value_transfers import ValueTransferAnalyzer
from .analysis.gvc_rents import GVCRentExtractor
from .analysis.super_exploitation import SuperExploitationMetrics

from .io_framework.multi_country import MultiCountryIOTable
from .policy.simulations import PolicySimulator

__all__ = [
    'UnequaExchangeModel',
    'LaborValueCalculator',
    'ValueTransferCalculator',
    'EmmanuelModel',
    'AminModel',
    'PrebischSingerModel',
    'ValueTransferAnalyzer',
    'GVCRentExtractor',
    'SuperExploitationMetrics',
    'MultiCountryIOTable',
    'PolicySimulator'
]
