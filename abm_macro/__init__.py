"""
Agent-Based Macroeconomic Model

A comprehensive ABM for heterodox macroeconomic analysis.

Features:
- 1000 heterogeneous firms with production, pricing, investment
- 5000 heterogeneous households with consumption, labor supply, portfolio choice
- Banking sector with credit creation and rationing
- Government and central bank with policy capabilities
- Emergent business cycles from micro interactions
- Wealth and income distribution evolution
- Policy experiment framework

Usage:
    >>> from abm_macro import MacroeconomyABM, ABMVisualizer
    >>> economy = MacroeconomyABM(n_firms=1000, n_households=5000, n_banks=10)
    >>> results = economy.run(n_periods=300)
    >>> viz = ABMVisualizer(economy)
    >>> viz.plot_macro_dashboard(save_path='dashboard.png')
"""

__version__ = "1.0.0"
__author__ = "Heterodox Economics ABM"

# Core components
from .base import Agent, AgentType, Market, EconomyState, calculate_gini, calculate_lorenz_curve
from .firms import Firm
from .households import Household, LaborMarket
from .banks import Bank, CreditMarket
from .government import Government, CentralBank, PolicyExperiment
from .economy import MacroeconomyABM, SimulationResults
from .visualization import ABMVisualizer, compare_policy_experiments
from .representative_agent import RepresentativeAgentModel, compare_abm_vs_representative

__all__ = [
    # Core classes
    'MacroeconomyABM',
    'SimulationResults',

    # Agents
    'Firm',
    'Household',
    'Bank',

    # Institutions
    'Government',
    'CentralBank',
    'PolicyExperiment',

    # Markets
    'LaborMarket',
    'CreditMarket',
    'Market',

    # Visualization
    'ABMVisualizer',
    'compare_policy_experiments',

    # Comparison
    'RepresentativeAgentModel',
    'compare_abm_vs_representative',

    # Utilities
    'calculate_gini',
    'calculate_lorenz_curve',
    'EconomyState',
]
