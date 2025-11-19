"""
Firm Agents for ABM Macroeconomic Model

Implements heterogeneous firms with:
- Post-Keynesian markup pricing (not marginal cost = price)
- Investment based on profitability and animal spirits (not intertemporal optimization)
- Adaptive expectations (not rational expectations)
- Quantity adjustment to demand shocks (Keynesian)
- Credit-financed investment (Minskyan)

References:
- Kalecki (1971): markup pricing, degree of monopoly
- Minsky (1986): financial instability, investment finance
- Dosi et al. (2010): evolutionary ABM with Schumpeterian competition
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base import Agent, AgentType


class Firm(Agent):
    """
    Heterogeneous firm agent with realistic micro-behavior.

    Key features:
    - Production function: Y = A * L^alpha * K^(1-alpha)
    - Markup pricing: P = (1 + markup) * unit_cost
    - Investment: I = f(profits, animal_spirits, debt_burden)
    - Employment: L = f(expected_demand, productivity)
    """

    def __init__(self, firm_id: int, initial_capital: float = 100.0):
        super().__init__(firm_id, AgentType.FIRM)

        # Production technology (heterogeneous)
        self.capital = initial_capital
        self.labor = 0.0
        self.productivity = np.random.lognormal(0.0, 0.3)  # TFP: log-normal distribution
        self.alpha = 0.7  # Labor share in production (Cobb-Douglas)

        # Pricing behavior (markup varies by firm - degree of monopoly)
        self.markup = np.random.uniform(0.1, 0.4)  # Kaleckian markup
        self.price = 1.0
        self.unit_cost = 0.0

        # Production and sales
        self.output = 0.0
        self.expected_demand = 50.0
        self.actual_sales = 0.0
        self.inventories = 0.0
        self.target_inventory_ratio = np.random.uniform(0.05, 0.15)

        # Financial variables
        self.revenue = 0.0
        self.profits = 0.0
        self.retained_earnings = 0.0
        self.desired_investment = 0.0
        self.actual_investment = 0.0
        self.loan_request = 0.0
        self.loan_approved = 0.0
        self.credit_rationed = False

        # Employment
        self.labor_demand = 0.0
        self.workers: List[int] = []  # IDs of employed households
        self.wage_offered = np.random.lognormal(np.log(1.0), 0.1)  # Heterogeneous wages

        # Investment behavior parameters
        self.animal_spirits = np.random.uniform(0.5, 1.5)  # Keynes: psychological factor
        self.target_capacity_utilization = 0.85
        self.depreciation_rate = 0.05

        # Adaptive expectation parameters
        self.demand_expectation_speed = np.random.uniform(0.2, 0.4)

        # Financial fragility (Minsky)
        self.debt_service = 0.0
        self.financial_position = "hedge"  # hedge, speculative, or ponzi

    def step(self, t: int, economy_state: Dict[str, Any]):
        """
        Firm's decision sequence (sequential, not simultaneous optimization):

        1. Update expectations based on past sales
        2. Decide production level
        3. Demand labor
        4. Produce
        5. Set price (markup rule)
        6. Decide investment (profitability + animal spirits)
        7. Seek credit if needed
        8. Update financial position
        """
        self.reset_flows()

        # 1. Update demand expectations (adaptive, bounded rationality)
        if t > 0:
            self.update_demand_expectations()

        # 2. Plan production
        self.plan_production()

        # 3. Determine labor needs
        self.determine_labor_demand()

        # 4. Calculate costs (will be finalized after wage bill paid)
        self.calculate_costs(economy_state.get('average_wage', self.wage_offered))

        # 5. Set price using markup rule (Post-Keynesian)
        self.set_price_markup()

        # 6. Make investment decision
        self.decide_investment()

        # 7. Determine financing needs
        self.determine_financing_needs()

    def update_demand_expectations(self):
        """
        Adaptive expectations for demand.

        NOT rational expectations - firms use simple heuristics based on recent sales.
        Heterodox principle: bounded rationality, satisficing not optimizing.
        """
        # Adaptive expectations: weight recent sales more heavily
        past_sales = self.get_history('actual_sales', length=4)

        if len(past_sales) > 0:
            # Exponentially weighted moving average
            weights = np.exp(np.arange(len(past_sales)) / 2)
            weights = weights / weights.sum()
            sales_trend = np.sum(past_sales * weights)

            # Update expectations adaptively
            self.expected_demand = (
                self.demand_expectation_speed * sales_trend +
                (1 - self.demand_expectation_speed) * self.expected_demand
            )

            # Add inventory adjustment
            if self.inventories > self.target_inventory_ratio * self.output:
                self.expected_demand *= 0.95  # Reduce production if excess inventory
            elif self.inventories < 0.5 * self.target_inventory_ratio * self.output:
                self.expected_demand *= 1.05  # Increase if inventory too low

    def plan_production(self):
        """
        Production planning based on expected demand, not market clearing.

        Keynesian principle: firms respond to quantity signals, not just prices.
        """
        # Target production includes expected sales plus inventory buffer
        target_output = self.expected_demand * (1 + self.target_inventory_ratio)

        # Capacity constraint (capital stock limits output)
        max_output = self.calculate_potential_output()

        self.output = min(target_output, max_output)

    def calculate_potential_output(self) -> float:
        """
        Potential output given capital stock.

        Production function: Y = A * L^alpha * K^(1-alpha)
        Assumes labor is available (relaxed for capacity calculation)
        """
        if self.capital <= 0:
            return 0.0

        # Optimal labor given capital (FOC when not labor-constrained)
        optimal_labor = (self.alpha * self.productivity / self.wage_offered) ** (1 / (1 - self.alpha)) * self.capital

        return self.productivity * (optimal_labor ** self.alpha) * (self.capital ** (1 - self.alpha))

    def determine_labor_demand(self):
        """
        Labor demand derived from production plan.

        Inverts production function: L = (Y / (A * K^(1-alpha)))^(1/alpha)
        """
        if self.capital <= 0 or self.output <= 0:
            self.labor_demand = 0.0
            return

        # Required labor for planned output
        self.labor_demand = (
            self.output / (self.productivity * (self.capital ** (1 - self.alpha)))
        ) ** (1 / self.alpha)

        self.labor_demand = max(0.0, self.labor_demand)

    def produce(self, labor_hired: float):
        """
        Actual production given labor hired.

        May be less than planned if labor market is tight (rationing).
        """
        self.labor = labor_hired

        if self.capital <= 0 or self.labor <= 0:
            self.output = 0.0
        else:
            self.output = self.productivity * (self.labor ** self.alpha) * (self.capital ** (1 - self.alpha))

        # Add production to inventories
        self.inventories += self.output

    def calculate_costs(self, average_wage: float):
        """
        Calculate unit costs for markup pricing.

        Cost = wage_bill + capital_costs + interest_on_debt
        """
        # Wage bill
        wage_bill = self.labor * self.wage_offered

        # Capital costs (depreciation)
        capital_cost = self.depreciation_rate * self.capital

        # Interest payments
        interest_payment = self.state.debt * economy_state.get('interest_rate', 0.02)

        total_cost = wage_bill + capital_cost + interest_payment

        # Unit cost
        if self.output > 0:
            self.unit_cost = total_cost / self.output
        else:
            self.unit_cost = wage_bill / max(self.expected_demand, 1.0)

    def set_price_markup(self):
        """
        Markup pricing rule (Post-Keynesian).

        Price = (1 + markup) * unit_cost

        NOT marginal cost pricing. Markup reflects:
        - Degree of monopoly (Kalecki)
        - Target profit rate
        - Conventional pricing norms

        Prices are sticky - don't adjust instantly to demand/supply.
        """
        target_price = (1 + self.markup) * self.unit_cost

        # Price stickiness: gradual adjustment
        price_adjustment_speed = 0.3
        self.price = price_adjustment_speed * target_price + (1 - price_adjustment_speed) * self.price

        # Minimum price floor
        self.price = max(self.price, self.unit_cost * 1.05)

    def sell_goods(self, quantity_demanded: float) -> Dict[str, float]:
        """
        Sell goods to consumers.

        Returns: {quantity_sold, revenue, inventory_change}
        """
        # Can only sell what's in inventory
        quantity_sold = min(quantity_demanded, self.inventories)

        self.actual_sales = quantity_sold
        self.revenue = quantity_sold * self.price
        self.state.income += self.revenue

        self.inventories -= quantity_sold

        return {
            'quantity': quantity_sold,
            'price': self.price,
            'revenue': self.revenue
        }

    def decide_investment(self):
        """
        Investment decision (Post-Keynesian/Kaleckian/Minskyan).

        I = f(profitability, animal_spirits, capacity_utilization, debt_burden)

        NOT intertemporal optimization with perfect foresight.
        Investment responds to:
        1. Current/recent profitability (backward-looking)
        2. Animal spirits (psychological, exogenous)
        3. Capacity constraints
        4. Financial fragility (Minsky)
        """
        # Profit rate
        if self.capital > 0:
            profit_rate = self.profits / self.capital
        else:
            profit_rate = 0.0

        # Capacity utilization
        potential_output = self.calculate_potential_output()
        if potential_output > 0:
            capacity_utilization = self.output / potential_output
        else:
            capacity_utilization = 0.0

        # Financial pressure (Minsky: debt service burden)
        if self.revenue > 0:
            debt_burden = self.debt_service / self.revenue
        else:
            debt_burden = 0.0

        # Investment function (Kalecki-style)
        # I/K = a0 + a1*profit_rate + a2*animal_spirits + a3*capacity_util - a4*debt_burden

        investment_rate = (
            0.05 +  # Baseline
            0.3 * profit_rate +  # Profitability effect
            0.1 * self.animal_spirits +  # Animal spirits
            0.2 * max(0, capacity_utilization - self.target_capacity_utilization) -  # Capacity pressure
            0.3 * debt_burden  # Financial fragility
        )

        investment_rate = max(0.0, investment_rate)  # No negative gross investment

        self.desired_investment = investment_rate * self.capital

        # Replace depreciated capital
        self.desired_investment += self.depreciation_rate * self.capital

    def determine_financing_needs(self):
        """
        Determine how to finance investment.

        Pecking order (Minsky):
        1. Internal funds (retained earnings)
        2. Bank loans (external finance)

        This creates endogenous financial cycles.
        """
        available_internal = max(0.0, self.retained_earnings)

        if self.desired_investment <= available_internal:
            # Hedge finance: can fund investment from internal sources
            self.loan_request = 0.0
            self.financial_position = "hedge"
        else:
            # Need external finance
            self.loan_request = self.desired_investment - available_internal

            # Check if this creates financial fragility
            future_debt_service = (self.state.debt + self.loan_request) * 0.05  # Assume 5% rate

            if future_debt_service < 0.3 * self.revenue:
                self.financial_position = "hedge"
            elif future_debt_service < 0.7 * self.revenue:
                self.financial_position = "speculative"  # Can pay interest but not principal
            else:
                self.financial_position = "ponzi"  # Need to borrow to pay interest

    def execute_investment(self, loan_approved: float):
        """
        Execute investment given credit availability.

        Credit rationing (Stiglitz): banks may not approve full loan request.
        This creates pro-cyclical investment dynamics.
        """
        self.loan_approved = loan_approved
        self.credit_rationed = (loan_approved < self.loan_request * 0.95)

        # Actual investment constrained by finance
        available_finance = max(0.0, self.retained_earnings) + loan_approved

        self.actual_investment = min(self.desired_investment, available_finance)

        # Update capital stock
        self.capital += self.actual_investment
        self.capital -= self.depreciation_rate * self.capital  # Depreciation

        # Update debt
        if loan_approved > 0:
            self.state.debt += loan_approved

        # Use up retained earnings
        self.retained_earnings -= min(self.retained_earnings, self.actual_investment)

        self.state.expenditure += self.actual_investment

    def pay_wages(self):
        """Pay wages to workers"""
        wage_bill = self.labor * self.wage_offered
        self.state.expenditure += wage_bill
        return wage_bill

    def pay_debt_service(self, interest_rate: float):
        """
        Pay interest on debt.

        If cannot pay → bankruptcy risk (Minskyan crisis)
        """
        self.debt_service = self.state.debt * interest_rate

        if self.debt_service > 0:
            if self.revenue >= self.debt_service:
                # Can service debt
                self.state.expenditure += self.debt_service
                return True
            else:
                # Cannot service debt → financial distress
                self.bankruptcy_flag = True
                return False
        return True

    def calculate_profits(self):
        """
        Calculate profits.

        Π = Revenue - Costs
        """
        total_costs = self.labor * self.wage_offered + self.debt_service + self.depreciation_rate * self.capital

        self.profits = self.revenue - total_costs
        self.retained_earnings += self.profits * 0.7  # 70% retention ratio

        return self.profits

    def handle_bankruptcy(self):
        """
        Bankruptcy procedure.

        Simplified: firm exits, capital destroyed, debt written off.
        Creates contagion through banking system.
        """
        self.active = False
        self.capital = 0.0
        self.labor = 0.0
        self.workers = []
        # Debt is written off - creates losses for banks

    def get_state_dict(self) -> Dict[str, Any]:
        """Return current state for logging"""
        return {
            'id': self.agent_id,
            'capital': self.capital,
            'output': self.output,
            'price': self.price,
            'labor': self.labor,
            'profits': self.profits,
            'debt': self.state.debt,
            'investment': self.actual_investment,
            'sales': self.actual_sales,
            'inventories': self.inventories,
            'financial_position': self.financial_position,
            'bankruptcy': self.bankruptcy_flag
        }
