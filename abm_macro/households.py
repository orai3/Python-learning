"""
Household Agents for ABM Macroeconomic Model

Implements heterogeneous households with:
- Consumption based on income and wealth (Keynesian/Kaleckian)
- Labor supply and job search
- Portfolio choice (deposits, assets)
- Heterogeneous consumption propensities (inequality matters!)

References:
- Keynes (1936): consumption function, marginal propensity to consume
- Godley & Lavoie (2007): stock-flow consistent household behavior
- Bowles & Park (2005): inequality and demand
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base import Agent, AgentType


class Household(Agent):
    """
    Heterogeneous household agent.

    Key heterodox features:
    - Consumption driven by CURRENT income (not lifetime optimization)
    - Marginal propensity to consume varies by wealth (inequality matters)
    - Job search subject to frictions (not instant market clearing)
    - Class matters: workers vs capital owners have different behavior
    """

    def __init__(self, household_id: int, initial_wealth: float = 10.0):
        super().__init__(household_id, AgentType.HOUSEHOLD)

        # Wealth and income
        self.state.wealth = initial_wealth
        self.labor_income = 0.0
        self.capital_income = 0.0  # Dividends, interest
        self.transfer_income = 0.0  # Government transfers
        self.total_income = 0.0

        # Consumption behavior (heterogeneous)
        # Key heterodox insight: MPC varies by wealth class
        wealth_class = np.random.random()

        if wealth_class < 0.6:  # Working class: high MPC
            self.mpc = np.random.uniform(0.85, 0.98)
            self.wealth_class = "worker"
        elif wealth_class < 0.9:  # Middle class: moderate MPC
            self.mpc = np.random.uniform(0.70, 0.85)
            self.wealth_class = "middle"
        else:  # Wealthy: low MPC
            self.mpc = np.random.uniform(0.40, 0.65)
            self.wealth_class = "wealthy"

        self.consumption = 0.0
        self.autonomous_consumption = np.random.uniform(5.0, 15.0)

        # Labor market status
        self.employed = False
        self.employer_id: Optional[int] = None
        self.wage = 0.0
        self.reservation_wage = np.random.uniform(0.8, 1.2)  # Minimum acceptable wage
        self.labor_supply = 1.0  # Hours willing to work (normalized)

        # Job search
        self.unemployed_duration = 0
        self.job_applications = 0
        self.wage_aspiration = 1.0  # Desired wage (adjusts with unemployment)

        # Portfolio (simplified)
        self.deposits = initial_wealth  # Bank deposits
        self.target_savings_rate = np.random.uniform(0.02, 0.15)

        # Behavioral parameters
        self.consumption_habit = 0.5  # Habits/inertia in consumption (Duesenberry)
        self.precautionary_motive = np.random.uniform(0.1, 0.3)  # Save more when uncertain

        # Expectations
        self.expected_income = 0.0
        self.income_uncertainty = 0.0

    def step(self, t: int, economy_state: Dict[str, Any]):
        """
        Household decision sequence:

        1. Update income expectations
        2. Search for job if unemployed
        3. Receive income (wages, capital income, transfers)
        4. Decide consumption
        5. Update wealth
        """
        self.reset_flows()

        # Update expectations
        if t > 0:
            self.update_income_expectations(economy_state)

        # Labor market behavior
        if not self.employed:
            self.search_for_job(economy_state)
            self.unemployed_duration += 1
        else:
            self.unemployed_duration = 0

        # Will receive income during market clearing
        # (wages paid by firms, transfers by government)

    def update_income_expectations(self, economy_state: Dict[str, Any]):
        """
        Form expectations about future income.

        Adaptive + influenced by macro conditions (unemployment rate).
        """
        past_income = self.get_history('total_income', length=4)

        if len(past_income) > 0:
            # Adaptive expectations
            self.expected_income = 0.6 * np.mean(past_income) + 0.4 * self.expected_income

            # Uncertainty increases with income volatility
            if len(past_income) > 1:
                self.income_uncertainty = np.std(past_income) / (np.mean(past_income) + 1.0)
        else:
            self.expected_income = self.total_income

        # Macro conditions affect expectations
        unemployment_rate = economy_state.get('unemployment_rate', 0.05)
        if unemployment_rate > 0.08:  # High unemployment
            self.income_uncertainty += 0.05  # Precautionary motive increases

    def search_for_job(self, economy_state: Dict[str, Any]):
        """
        Job search with realistic frictions.

        - Wage aspiration declines with unemployment duration (search theory)
        - Limited number of applications (search costs)
        - Random matching (not perfect information)
        """
        self.job_applications = np.random.randint(1, 5)  # Limited search

        # Wage aspiration declines with unemployment duration
        # Represents discouragement and necessity
        if self.unemployed_duration > 4:
            self.wage_aspiration *= 0.98  # Gradually lower expectations

        self.wage_aspiration = max(self.reservation_wage, self.wage_aspiration)

    def accept_job(self, employer_id: int, wage: float) -> bool:
        """
        Decide whether to accept job offer.

        Accept if wage >= reservation wage (adjusted for unemployment duration).
        """
        effective_reservation = self.reservation_wage * (0.95 ** min(self.unemployed_duration, 10))

        if wage >= effective_reservation:
            self.employed = True
            self.employer_id = employer_id
            self.wage = wage
            return True
        else:
            return False

    def lose_job(self):
        """
        Become unemployed.

        Triggers precautionary saving, cuts consumption.
        """
        self.employed = False
        self.employer_id = None
        self.wage = 0.0
        self.wage_aspiration = self.wage * 0.95 if self.wage > 0 else 1.0

    def receive_labor_income(self, wage: float):
        """Receive wage income from employer"""
        self.labor_income = wage
        self.total_income += wage
        self.state.income += wage

    def receive_transfer(self, amount: float):
        """Receive transfer from government (unemployment benefits, etc.)"""
        self.transfer_income = amount
        self.total_income += amount
        self.state.income += amount

    def receive_capital_income(self, amount: float):
        """Receive capital income (interest, dividends)"""
        self.capital_income = amount
        self.total_income += amount
        self.state.income += amount

    def decide_consumption(self, economy_state: Dict[str, Any]):
        """
        Consumption decision (Keynesian/Kaleckian).

        C = C_auto + MPC * Y + habit * C_-1 - precautionary * uncertainty * W

        Key features:
        - Current income matters (not just lifetime wealth)
        - MPC heterogeneity (inequality affects aggregate demand)
        - Habits/inertia (consumption smoothing without optimization)
        - Precautionary motive (uncertainty reduces consumption)
        """
        # Base consumption from income
        income_driven_consumption = self.mpc * self.total_income

        # Autonomous consumption (subsistence)
        base_consumption = self.autonomous_consumption

        # Habit persistence (last period's consumption matters)
        past_consumption = self.get_history('consumption', length=1)
        if len(past_consumption) > 0:
            habit_consumption = self.consumption_habit * past_consumption[0]
        else:
            habit_consumption = 0.0

        # Precautionary reduction (save when uncertain)
        precautionary_reduction = self.precautionary_motive * self.income_uncertainty * self.state.wealth

        # Total desired consumption
        desired_consumption = (
            base_consumption +
            0.5 * income_driven_consumption +
            0.5 * habit_consumption -
            precautionary_reduction
        )

        # Cannot consume more than income + wealth (liquidity constraint)
        max_consumption = self.total_income + self.state.wealth

        self.consumption = max(0.0, min(desired_consumption, max_consumption))

        # Ensure minimum consumption (subsistence)
        self.consumption = max(self.consumption, self.autonomous_consumption * 0.5)

        return self.consumption

    def execute_consumption(self, goods_purchased: float, price_paid: float):
        """
        Execute consumption spending.

        Updates wealth, expenditure.
        """
        actual_spending = goods_purchased * price_paid
        self.state.expenditure += actual_spending

        # Update wealth (stock-flow consistency)
        self.state.wealth += self.total_income - actual_spending

        # Deposits adjust
        self.deposits = self.state.wealth

    def update_wealth(self):
        """
        Update wealth from income and consumption.

        Stock-flow consistent: ΔW = Y - C
        """
        # Already handled in execute_consumption
        pass

    def get_consumption_by_class(self) -> str:
        """Return wealth class for distributional analysis"""
        return self.wealth_class

    def get_state_dict(self) -> Dict[str, Any]:
        """Return current state for logging"""
        return {
            'id': self.agent_id,
            'wealth': self.state.wealth,
            'income': self.total_income,
            'labor_income': self.labor_income,
            'consumption': self.consumption,
            'employed': self.employed,
            'wage': self.wage,
            'wealth_class': self.wealth_class,
            'mpc': self.mpc,
            'unemployed_duration': self.unemployed_duration
        }


class LaborMarket:
    """
    Labor market with search and matching.

    NOT a Walrasian market - features:
    - Search frictions
    - Random matching
    - Quantity rationing (unemployment)
    - Wage rigidity
    """

    def __init__(self):
        self.vacancies: List[Dict] = []
        self.job_seekers: List[Household] = []
        self.matches: List[tuple] = []
        self.unemployment_rate = 0.0
        self.average_wage = 1.0

    def post_vacancy(self, firm_id: int, wage: float, positions: int = 1):
        """Firm posts job vacancy"""
        for _ in range(int(positions)):
            self.vacancies.append({
                'firm_id': firm_id,
                'wage': wage
            })

    def add_job_seeker(self, household: Household):
        """Household searches for job"""
        self.job_seekers.append(household)

    def match(self) -> List[Dict]:
        """
        Random matching between vacancies and job seekers.

        Implements search frictions - not all unemployed find jobs.
        """
        if len(self.vacancies) == 0 or len(self.job_seekers) == 0:
            return []

        # Random matching (each seeker applies to random subset of vacancies)
        matches = []
        remaining_vacancies = list(self.vacancies)
        np.random.shuffle(remaining_vacancies)

        for seeker in self.job_seekers:
            if len(remaining_vacancies) == 0:
                break

            # Seeker considers random subset of vacancies
            n_applications = min(seeker.job_applications, len(remaining_vacancies))
            considered_vacancies = np.random.choice(
                len(remaining_vacancies),
                size=n_applications,
                replace=False
            )

            # Apply to best wage among considered
            best_idx = None
            best_wage = 0.0

            for idx in considered_vacancies:
                vacancy = remaining_vacancies[idx]
                if vacancy['wage'] > best_wage:
                    best_wage = vacancy['wage']
                    best_idx = idx

            # Accept if wage acceptable
            if best_idx is not None:
                vacancy = remaining_vacancies[best_idx]
                if seeker.accept_job(vacancy['firm_id'], vacancy['wage']):
                    matches.append({
                        'household_id': seeker.agent_id,
                        'firm_id': vacancy['firm_id'],
                        'wage': vacancy['wage']
                    })
                    remaining_vacancies.pop(best_idx)

        self.matches = matches
        return matches

    def clear(self, firms: List, households: List[Household]) -> Dict[str, Any]:
        """
        Clear labor market.

        Returns employment statistics.
        """
        # Collect vacancies
        self.vacancies = []
        for firm in firms:
            if firm.active and firm.labor_demand > 0:
                # Firms post vacancies at their offered wage
                labor_needed = firm.labor_demand - firm.labor
                if labor_needed > 0:
                    self.post_vacancy(firm.agent_id, firm.wage_offered, int(labor_needed) + 1)

        # Collect job seekers
        self.job_seekers = [h for h in households if not h.employed]

        # Match
        matches = self.match()

        # Update employment relationships
        firm_dict = {f.agent_id: f for f in firms}

        for match in matches:
            household = next(h for h in households if h.agent_id == match['household_id'])
            firm = firm_dict[match['firm_id']]

            # Firm hires household
            firm.workers.append(household.agent_id)
            firm.labor += 1.0  # Simplified: 1 worker = 1 unit labor

        # Calculate statistics
        employed = sum(1 for h in households if h.employed)
        labor_force = len(households)
        self.unemployment_rate = (labor_force - employed) / labor_force if labor_force > 0 else 0.0

        wages = [h.wage for h in households if h.employed and h.wage > 0]
        self.average_wage = np.mean(wages) if len(wages) > 0 else 1.0

        return {
            'employment': employed,
            'unemployment': labor_force - employed,
            'unemployment_rate': self.unemployment_rate,
            'average_wage': self.average_wage,
            'vacancies': len(self.vacancies),
            'matches': len(matches)
        }
