"""
Main Simulation Engine for ABM Macroeconomic Model

Coordinates all agents and markets through time.

Sequence of events each period:
1. Government/CB policy decisions
2. Firms: production planning, pricing, investment
3. Labor market: matching
4. Credit market: loan applications
5. Goods market: consumption
6. Accounting: profits, income, wealth updates
7. Bankruptcies and exits
8. Data collection

References:
- Dosi et al. (2010, 2013): Keynes meets Schumpeter
- Dawid et al. (2019): Eurace@Unibi model
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import copy

from .base import Agent, EconomyState, Market, calculate_gini
from .firms import Firm
from .households import Household, LaborMarket
from .banks import Bank, CreditMarket
from .government import Government, CentralBank, PolicyExperiment


@dataclass
class SimulationResults:
    """Container for simulation results"""
    time_series: Dict[str, List[float]]
    final_state: EconomyState
    firms_history: List[Dict[str, Any]]
    households_history: List[Dict[str, Any]]
    banks_history: List[Dict[str, Any]]


class MacroeconomyABM:
    """
    Agent-Based Macroeconomic Model.

    Main simulation engine coordinating all agents and markets.
    """

    def __init__(self,
                 n_firms: int = 1000,
                 n_households: int = 5000,
                 n_banks: int = 10,
                 random_seed: Optional[int] = None):
        """
        Initialize economy with heterogeneous agents.

        Args:
            n_firms: Number of firms
            n_households: Number of households
            n_banks: Number of banks
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.n_firms = n_firms
        self.n_households = n_households
        self.n_banks = n_banks

        # Time
        self.t = 0
        self.max_time = 300

        # Agents
        self.firms: List[Firm] = []
        self.households: List[Household] = []
        self.banks: List[Bank] = []

        # Institutions
        self.government = Government()
        self.central_bank = CentralBank()

        # Markets
        self.labor_market = LaborMarket()
        self.credit_market = None  # Will initialize after banks created
        self.goods_market = Market("goods")

        # Economy state
        self.state = EconomyState()
        self.state_history: List[EconomyState] = []

        # Time series data
        self.time_series: Dict[str, List[float]] = {
            'gdp': [], 'consumption': [], 'investment': [],
            'unemployment_rate': [], 'inflation': [],
            'wage_share': [], 'profit_share': [],
            'gini_income': [], 'gini_wealth': [],
            'total_credit': [], 'interest_rate': [],
            'bankruptcies': [], 'credit_rationing_rate': [],
            'government_deficit': [], 'public_debt': []
        }

        # Initialize agents
        self._initialize_agents()

        print(f"✓ Economy initialized: {n_firms} firms, {n_households} households, {n_banks} banks")

    def _initialize_agents(self):
        """Create all agents with heterogeneous characteristics"""

        # Create firms with size distribution
        # Empirical fact: firm size follows power law
        sizes = np.random.pareto(1.5, self.n_firms) + 1  # Pareto distribution
        sizes = sizes / sizes.mean() * 100  # Normalize

        for i in range(self.n_firms):
            firm = Firm(i, initial_capital=sizes[i])
            self.firms.append(firm)

        # Create households with wealth distribution
        # Initialize with some inequality (Gini ~ 0.6-0.7)
        wealth_dist = np.random.lognormal(2.0, 1.2, self.n_households)
        wealth_dist = wealth_dist / wealth_dist.mean() * 10  # Normalize

        for i in range(self.n_households):
            household = Household(i, initial_wealth=wealth_dist[i])
            self.households.append(household)

        # Create banks
        for i in range(self.n_banks):
            bank = Bank(i, initial_capital=200.0)
            self.banks.append(bank)

        # Initialize credit market
        self.credit_market = CreditMarket(self.banks)

    def step(self):
        """
        Execute one time step of the simulation.

        This is the core ABM time-stepping logic.
        """
        self.t += 1

        # Store previous state for growth calculations
        prev_gdp = self.state.gdp if self.t > 1 else 100.0

        # --- Phase 1: Policy Decisions ---
        self.central_bank.set_interest_rate(self.state.to_dict())
        self.government.decide_spending(self.state.to_dict())

        # Check for crisis and respond
        if self.central_bank.detect_crisis(self.state.to_dict(), self.banks):
            self.central_bank.crisis_response(self.banks)

        # --- Phase 2: Firm Decisions ---
        for firm in self.firms:
            if firm.active:
                firm.step(self.t, self.state.to_dict())

        # --- Phase 3: Household Decisions ---
        for household in self.households:
            household.step(self.t, self.state.to_dict())

        # --- Phase 4: Labor Market Clearing ---
        labor_stats = self.labor_market.clear(
            [f for f in self.firms if f.active],
            self.households
        )

        # Update firms' production with hired labor
        for firm in self.firms:
            if firm.active:
                firm.produce(firm.labor)

        # Firms pay wages
        for firm in self.firms:
            if firm.active and len(firm.workers) > 0:
                wage_bill = firm.pay_wages()
                # Distribute wages to workers
                for worker_id in firm.workers:
                    household = self.households[worker_id]
                    household.receive_labor_income(firm.wage_offered)

        # --- Phase 5: Government Transfers ---
        self.government.pay_transfers(self.households, self.state.to_dict())

        # --- Phase 6: Credit Market ---
        credit_stats = self.credit_market.process_loan_applications(
            [f for f in self.firms if f.active],
            self.central_bank.interest_rate
        )

        # --- Phase 7: Goods Market ---
        # Households decide consumption
        for household in self.households:
            household.decide_consumption(self.state.to_dict())

        # Match consumers with producers (simplified: random matching)
        total_consumption_demand = sum(h.consumption for h in self.households)
        total_supply = sum(f.inventories for f in self.firms if f.active)

        # Aggregate price level
        prices = [f.price for f in self.firms if f.active and f.output > 0]
        self.state.price_level = np.mean(prices) if len(prices) > 0 else 1.0

        # Calculate how much each firm sells (proportional to inventory)
        if total_supply > 0:
            for firm in self.firms:
                if firm.active and firm.inventories > 0:
                    firm_share = firm.inventories / total_supply
                    demand_for_firm = total_consumption_demand * firm_share
                    firm.sell_goods(demand_for_firm / self.state.price_level)

        # Households execute consumption
        for household in self.households:
            # Simplified: assume they get goods at average price
            quantity = household.consumption / self.state.price_level
            household.execute_consumption(quantity, self.state.price_level)

        # Government spending (adds to demand)
        govt_consumption = self.government.spending / self.state.price_level
        # Distribute to firms
        if len(self.firms) > 0:
            for firm in self.firms:
                if firm.active:
                    firm.sell_goods(govt_consumption / len([f for f in self.firms if f.active]))

        # --- Phase 8: Financial Accounting ---
        # Firms calculate profits
        for firm in self.firms:
            if firm.active:
                firm.calculate_profits()
                # Pay debt service
                can_pay = firm.pay_debt_service(self.central_bank.interest_rate)
                if not can_pay:
                    firm.bankruptcy_flag = True

        # Banks collect interest and handle defaults
        for bank in self.banks:
            bank.collect_interest(self.central_bank.interest_rate)
            bank.pay_deposit_interest(self.central_bank.interest_rate)
            bank.calculate_profits()

        # Households receive capital income (simplified: bank dividends + firm dividends)
        total_bank_profits = sum(b.profits * 0.1 for b in self.banks)  # 10% paid out
        total_firm_profits = sum(f.profits * 0.3 for f in self.firms if f.active and f.profits > 0)

        # Distribute to wealthy households (top 20%)
        wealthy_households = sorted(self.households, key=lambda h: h.state.wealth, reverse=True)
        top_20pct = wealthy_households[:int(0.2 * len(self.households))]

        capital_income_per_wealthy = (total_bank_profits + total_firm_profits) / len(top_20pct)
        for household in top_20pct:
            household.receive_capital_income(capital_income_per_wealthy)

        # Government collects taxes
        self.government.collect_taxes(self.households, self.firms)
        self.government.update_budget()

        # --- Phase 9: Bankruptcies and Exits ---
        bankrupt_firms = [f for f in self.firms if f.bankruptcy_flag]
        n_bankruptcies = len(bankrupt_firms)

        # Handle firm bankruptcies
        for firm in bankrupt_firms:
            # Workers lose jobs
            for worker_id in firm.workers:
                self.households[worker_id].lose_job()

            # Banks suffer losses
            self.credit_market.handle_defaults([firm])

            # Firm exits
            firm.handle_bankruptcy()

        # Firm entry (replace bankrupted firms)
        # New firms enter with small size
        for i in range(n_bankruptcies):
            if len([f for f in self.firms if f.active]) < self.n_firms:
                # Reactivate bankrupt firm as new entrant
                bankrupt_firms[i].active = True
                bankrupt_firms[i].capital = np.random.uniform(20, 50)
                bankrupt_firms[i].state.debt = 0.0
                bankrupt_firms[i].bankruptcy_flag = False
                bankrupt_firms[i].profits = 0.0
                bankrupt_firms[i].inventories = 0.0

        # --- Phase 10: Data Collection ---
        self.collect_aggregate_statistics()

        # Record state
        for firm in self.firms:
            if firm.active:
                firm.record_state()
        for household in self.households:
            household.record_state()

        self.state_history.append(copy.deepcopy(self.state))

    def collect_aggregate_statistics(self):
        """
        Compute aggregate statistics from micro data.

        This is key to ABM: macro emerges from micro!
        """
        active_firms = [f for f in self.firms if f.active]

        # National accounts
        self.state.gdp = (
            sum(h.consumption for h in self.households) +
            sum(f.actual_investment for f in active_firms) +
            self.government.spending
        )

        self.state.consumption = sum(h.consumption for h in self.households)
        self.state.investment = sum(f.actual_investment for f in active_firms)
        self.state.government_spending = self.government.spending

        # Labor market
        self.state.employment = sum(1 for h in self.households if h.employed)
        self.state.unemployment = len(self.households) - self.state.employment
        self.state.labor_force = len(self.households)
        self.state.unemployment_rate = self.state.unemployment / self.state.labor_force

        wages = [h.wage for h in self.households if h.employed and h.wage > 0]
        self.state.average_wage = np.mean(wages) if len(wages) > 0 else 1.0

        # Distribution
        incomes = np.array([h.total_income for h in self.households])
        wealths = np.array([h.state.wealth for h in self.households])

        self.state.gini_income = calculate_gini(incomes)
        self.state.gini_wealth = calculate_gini(wealths)

        # Wage share (labor compensation / GDP)
        total_wages = sum(h.labor_income for h in self.households)
        if self.state.gdp > 0:
            self.state.wage_share = total_wages / self.state.gdp
            self.state.profit_share = 1 - self.state.wage_share
        else:
            self.state.wage_share = 0.5
            self.state.profit_share = 0.5

        # Financial
        self.state.total_credit = sum(sum(b.loans.values()) for b in self.banks)
        self.state.total_debt = sum(f.state.debt for f in active_firms)
        if self.state.gdp > 0:
            self.state.debt_to_gdp = self.state.total_debt / self.state.gdp
        else:
            self.state.debt_to_gdp = 0.0

        self.state.bankruptcies = sum(1 for f in self.firms if f.bankruptcy_flag)

        # Prices
        if len(self.state_history) > 0:
            prev_price = self.state_history[-1].price_level
            if prev_price > 0:
                self.state.inflation = (self.state.price_level - prev_price) / prev_price
            else:
                self.state.inflation = 0.0
        else:
            self.state.inflation = 0.0

        self.state.interest_rate = self.central_bank.interest_rate

        # Update time series
        self.time_series['gdp'].append(self.state.gdp)
        self.time_series['consumption'].append(self.state.consumption)
        self.time_series['investment'].append(self.state.investment)
        self.time_series['unemployment_rate'].append(self.state.unemployment_rate)
        self.time_series['inflation'].append(self.state.inflation)
        self.time_series['wage_share'].append(self.state.wage_share)
        self.time_series['profit_share'].append(self.state.profit_share)
        self.time_series['gini_income'].append(self.state.gini_income)
        self.time_series['gini_wealth'].append(self.state.gini_wealth)
        self.time_series['total_credit'].append(self.state.total_credit)
        self.time_series['interest_rate'].append(self.state.interest_rate)
        self.time_series['bankruptcies'].append(self.state.bankruptcies)
        self.time_series['credit_rationing_rate'].append(self.credit_market.rationing_rate)
        self.time_series['government_deficit'].append(self.government.budget_deficit)
        self.time_series['public_debt'].append(self.government.public_debt)

    def run(self, n_periods: int = 300, policy_experiment: Optional[PolicyExperiment] = None):
        """
        Run simulation for n_periods.

        Args:
            n_periods: Number of time steps
            policy_experiment: Optional policy experiment to apply
        """
        print(f"\n🚀 Starting ABM simulation for {n_periods} periods...")

        for t in range(n_periods):
            # Apply policy experiment if any
            if policy_experiment is not None:
                interventions = policy_experiment.apply_interventions(
                    t, self.government, self.central_bank
                )
                if len(interventions) > 0:
                    print(f"\n⚡ Policy intervention at t={t}:")
                    for intervention in interventions:
                        print(f"   {intervention}")

            # Execute time step
            self.step()

            # Print progress
            if (t + 1) % 50 == 0:
                print(f"  t={t+1}/{n_periods} | GDP={self.state.gdp:.1f} | "
                      f"Unemp={self.state.unemployment_rate:.1%} | "
                      f"Gini={self.state.gini_wealth:.2f}")

        print(f"\n✓ Simulation complete!")
        print(f"  Final GDP: {self.state.gdp:.1f}")
        print(f"  Final unemployment: {self.state.unemployment_rate:.1%}")
        print(f"  Final wealth Gini: {self.state.gini_wealth:.2f}")
        print(f"  Final debt/GDP: {self.state.debt_to_gdp:.1%}")

        return self.get_results()

    def get_results(self) -> SimulationResults:
        """Return simulation results"""
        return SimulationResults(
            time_series=self.time_series,
            final_state=self.state,
            firms_history=[],
            households_history=[],
            banks_history=[]
        )

    def get_distributional_data(self) -> Dict[str, np.ndarray]:
        """
        Extract distributional data for analysis.

        Returns wealth and income distributions.
        """
        return {
            'wealth': np.array([h.state.wealth for h in self.households]),
            'income': np.array([h.total_income for h in self.households]),
            'consumption': np.array([h.consumption for h in self.households]),
            'wealth_class': [h.wealth_class for h in self.households]
        }

    def get_firm_distribution(self) -> Dict[str, np.ndarray]:
        """Extract firm size and performance distribution"""
        active_firms = [f for f in self.firms if f.active]

        return {
            'capital': np.array([f.capital for f in active_firms]),
            'output': np.array([f.output for f in active_firms]),
            'employment': np.array([f.labor for f in active_firms]),
            'debt': np.array([f.state.debt for f in active_firms]),
            'profits': np.array([f.profits for f in active_firms])
        }
