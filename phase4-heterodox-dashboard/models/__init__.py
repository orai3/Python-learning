"""
Heterodox Macro Dashboard - Models Package
Data models for loading, managing, and analyzing economic data
"""

from .data_model import DataModel
from .frameworks import (
    PostKeynesianFramework,
    MarxianFramework,
    InstitutionalistFramework,
    FrameworkManager
)

__all__ = [
    'DataModel',
    'PostKeynesianFramework',
    'MarxianFramework',
    'InstitutionalistFramework',
    'FrameworkManager'
]
