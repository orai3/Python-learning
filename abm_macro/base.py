"""
Base classes for Agent-Based Macroeconomic Model

This module provides the foundation for a heterodox ABM approach to macroeconomics,
following the tradition of Dosi et al., Dawid et al., and other complexity economists.

Key ABM Principles:
- Heterogeneous agents with bounded rationality
- Decentralized interactions and emergent macro phenomena
- Out-of-equilibrium dynamics
- Stock-flow consistency
- Network structures and contagion
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class AgentType(Enum):
    """Types of agents in the economy"""
    HOUSEHOLD = "household"
    FIRM = "firm"
    BANK = "bank"
    GOVERNMENT = "government"
    CENTRAL_BANK = "central_bank"


@dataclass
class AgentState:
    """
    Base state container for all agents.
    Ensures stock-flow consistency tracking.
    """
    agent_id: int
    agent_type: AgentType
    wealth: float = 0.0
    income: float = 0.0
    expenditure: float = 0.0
    debt: float = 0.0
    assets: Dict[str, float] = field(default_factory=dict)
    liabilities: Dict[str, float] = field(default_factory=dict)

    def net_worth(self) -> float:
        """Calculate net worth (assets - liabilities)"""
        total_assets = sum(self.assets.values()) + self.wealth
        total_liabilities = sum(self.liabilities.values()) + self.debt
        return total_assets - total_liabilities

    def flow_check(self) -> float:
        """Check flow consistency: Δwealth = income - expenditure"""
        return self.income - self.expenditure


class Agent:
    """
    Base class for all economic agents.

    Implements common functionality:
    - Unique identification
    - State management
    - Balance sheet operations
    - Memory/history tracking
    """

    def __init__(self, agent_id: int, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState(agent_id, agent_type)
        self.history: List[AgentState] = []
        self.active = True
        self.bankruptcy_flag = False

        # Behavioral parameters (heterogeneous across agents)
        self.memory_length = np.random.randint(4, 20)  # Bounded rationality
        self.expectations: Dict[str, float] = {}

    def step(self, t: int, economy_state: Dict[str, Any]):
        """
        Execute one time step of agent behavior.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement step()")

    def update_expectations(self, variable: str, value: float, adaptive_param: float = 0.3):
        """
        Adaptive expectations with bounded rationality.
        Standard heterodox assumption: agents use simple heuristics, not rational expectations.
        """
        if variable not in self.expectations:
            self.expectations[variable] = value
        else:
            # Adaptive expectations: E_t = α*actual + (1-α)*E_{t-1}
            self.expectations[variable] = (
                adaptive_param * value +
                (1 - adaptive_param) * self.expectations[variable]
            )

    def get_history(self, variable: str, length: Optional[int] = None) -> np.ndarray:
        """Get historical values of a variable"""
        if length is None:
            length = self.memory_length

        length = min(length, len(self.history))
        if length == 0:
            return np.array([])

        return np.array([getattr(h, variable, 0.0) for h in self.history[-length:]])

    def record_state(self):
        """Save current state to history"""
        import copy
        self.history.append(copy.deepcopy(self.state))

        # Limit history length to save memory
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def reset_flows(self):
        """Reset flow variables at start of period"""
        self.state.income = 0.0
        self.state.expenditure = 0.0


class Market:
    """
    Base class for market mechanisms.

    ABM markets are NOT Walrasian auctioneers - they represent:
    - Sequential, decentralized matching
    - Quantity rationing when demand ≠ supply
    - Price stickiness and adjustment dynamics
    """

    def __init__(self, name: str):
        self.name = name
        self.price_history: List[float] = []
        self.quantity_history: List[float] = []
        self.rationing_events: List[Dict] = []

    def clear(self, demand: float, supply: float, price: float) -> Dict[str, float]:
        """
        Market clearing with rationing.

        Unlike neoclassical models, ABMs allow persistent disequilibrium.
        When demand ≠ supply, quantities are rationed (not prices adjusted to clear).
        """
        quantity_traded = min(demand, supply)
        excess_demand = demand - supply

        self.price_history.append(price)
        self.quantity_history.append(quantity_traded)

        if abs(excess_demand) > 0.01 * quantity_traded:
            self.rationing_events.append({
                'time': len(self.price_history),
                'excess_demand': excess_demand,
                'price': price
            })

        return {
            'quantity': quantity_traded,
            'price': price,
            'excess_demand': excess_demand,
            'rationed': excess_demand != 0
        }


class EconomyState:
    """
    Aggregate state of the economy.
    Tracks macro variables emergent from micro interactions.
    """

    def __init__(self):
        # National accounts
        self.gdp = 0.0
        self.consumption = 0.0
        self.investment = 0.0
        self.government_spending = 0.0
        self.exports = 0.0
        self.imports = 0.0

        # Labor market
        self.employment = 0
        self.unemployment = 0
        self.labor_force = 0
        self.unemployment_rate = 0.0
        self.average_wage = 0.0

        # Distribution
        self.gini_income = 0.0
        self.gini_wealth = 0.0
        self.wage_share = 0.0
        self.profit_share = 0.0

        # Financial
        self.total_credit = 0.0
        self.total_debt = 0.0
        self.debt_to_gdp = 0.0
        self.bankruptcies = 0

        # Prices
        self.price_level = 1.0
        self.inflation = 0.0
        self.interest_rate = 0.02

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items()}


class RandomMatching:
    """
    Random matching protocol for decentralized markets.

    In ABMs, agents don't have perfect information about all trading partners.
    They search and match randomly or through networks.
    """

    @staticmethod
    def match_pairs(buyers: List[Agent], sellers: List[Agent],
                    n_matches: Optional[int] = None) -> List[tuple]:
        """
        Random bilateral matching between buyers and sellers.

        This creates realistic search frictions absent in Walrasian markets.
        """
        if n_matches is None:
            n_matches = min(len(buyers), len(sellers))

        n_matches = min(n_matches, len(buyers), len(sellers))

        shuffled_buyers = np.random.choice(buyers, size=n_matches, replace=False)
        shuffled_sellers = np.random.choice(sellers, size=n_matches, replace=False)

        return list(zip(shuffled_buyers, shuffled_sellers))

    @staticmethod
    def match_network(agents: List[Agent], network_density: float = 0.1) -> Dict[int, List[int]]:
        """
        Create network connections between agents.

        Networks matter in ABMs:
        - Information diffusion
        - Contagion (financial, expectations)
        - Trade credit chains
        """
        n = len(agents)
        adjacency = {}

        for i, agent in enumerate(agents):
            # Each agent connects to random subset of others
            n_connections = np.random.binomial(n-1, network_density)
            partners = np.random.choice(
                [a for j, a in enumerate(agents) if j != i],
                size=min(n_connections, n-1),
                replace=False
            )
            adjacency[agent.agent_id] = [p.agent_id for p in partners]

        return adjacency


def calculate_gini(values: np.ndarray) -> float:
    """
    Calculate Gini coefficient.

    Central to heterodox economics: distribution matters for macro dynamics.
    High inequality → low consumption → demand constraints (Kaleckian)
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)

    return (2 * np.sum((np.arange(1, n+1)) * sorted_values)) / (n * cumsum[-1]) - (n+1)/n


def calculate_lorenz_curve(values: np.ndarray, n_points: int = 100) -> tuple:
    """
    Calculate Lorenz curve for visualization.

    Returns (cumulative_population_share, cumulative_value_share)
    """
    if len(values) == 0:
        return np.array([0, 1]), np.array([0, 1])

    sorted_values = np.sort(values)
    cumsum = np.cumsum(sorted_values)

    # Normalize
    pop_share = np.linspace(0, 1, len(sorted_values))
    value_share = cumsum / cumsum[-1]

    return pop_share, value_share
