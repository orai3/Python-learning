"""
Government and Central Bank for ABM Macroeconomic Model

Implements policy institutions:
- Government: fiscal policy, transfers, public spending
- Central Bank: monetary policy, interest rates, reserves

References:
- Godley & Lavoie (2007): Stock-flow consistent government sector
- Kalecki (1971): Government spending and aggregate demand
- Minsky (1986): Government as stabilizer
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base import Agent, AgentType


class Government:
    """
    Government sector with fiscal policy.

    Key features:
    - Counter-cyclical fiscal policy (automatic stabilizers + discretionary)
    - Unemployment benefits
    - Progressive taxation
    - Public debt dynamics
    """

    def __init__(self):
        self.agent_type = AgentType.GOVERNMENT

        # Fiscal variables
        self.spending = 0.0
        self.tax_revenue = 0.0
        self.transfers = 0.0
        self.budget_deficit = 0.0
        self.public_debt = 0.0

        # Policy parameters
        self.government_spending_target = 100.0  # Baseline spending
        self.unemployment_benefit_rate = 0.5  # 50% of average wage
        self.tax_rate_labor = 0.20  # Labor income tax
        self.tax_rate_capital = 0.25  # Capital income tax
        self.tax_rate_corporate = 0.21  # Corporate profits tax

        # Counter-cyclical policy
        self.fiscal_response_coefficient = 0.5  # How much to increase spending in recession
        self.automatic_stabilizer_strength = 1.0  # Strength of automatic stabilizers

    def collect_taxes(self, households: List, firms: List) -> float:
        """
        Collect taxes from households and firms.

        Progressive taxation (higher rates on capital income).
        """
        from .households import Household
        from .firms import Firm

        total_tax = 0.0

        # Household taxes
        for h in households:
            # Labor income tax
            labor_tax = h.labor_income * self.tax_rate_labor

            # Capital income tax (higher rate - redistribution)
            capital_tax = h.capital_income * self.tax_rate_capital

            household_tax = labor_tax + capital_tax
            total_tax += household_tax

            # Deduct from household income
            h.total_income -= household_tax

        # Corporate taxes
        for f in firms:
            if f.active and f.profits > 0:
                corporate_tax = f.profits * self.tax_rate_corporate
                total_tax += corporate_tax

                # Reduce retained earnings
                f.retained_earnings -= corporate_tax

        self.tax_revenue = total_tax
        return total_tax

    def pay_transfers(self, households: List, economy_state: Dict[str, Any]) -> float:
        """
        Pay unemployment benefits and other transfers.

        Automatic stabilizers: transfers increase in recessions.
        """
        from .households import Household

        total_transfers = 0.0
        average_wage = economy_state.get('average_wage', 1.0)

        for h in households:
            if not h.employed:
                # Unemployment benefit
                benefit = self.unemployment_benefit_rate * average_wage
                h.receive_transfer(benefit)
                total_transfers += benefit

        self.transfers = total_transfers
        return total_transfers

    def decide_spending(self, economy_state: Dict[str, Any]) -> float:
        """
        Decide government spending.

        Counter-cyclical: increase spending in recessions (Keynesian).
        """
        # Baseline spending
        spending = self.government_spending_target

        # Counter-cyclical response
        unemployment_rate = economy_state.get('unemployment_rate', 0.05)
        gdp_growth = economy_state.get('gdp_growth', 0.0)

        # Increase spending if unemployment high or negative growth
        if unemployment_rate > 0.08:
            spending += self.fiscal_response_coefficient * 50.0 * (unemployment_rate - 0.08)

        if gdp_growth < 0:
            spending += self.fiscal_response_coefficient * 30.0 * abs(gdp_growth)

        self.spending = spending
        return spending

    def update_budget(self):
        """
        Update budget deficit and public debt.

        Stock-flow consistent: ΔDebt = Deficit
        """
        self.budget_deficit = self.spending + self.transfers - self.tax_revenue
        self.public_debt += self.budget_deficit

    def get_fiscal_stance(self) -> str:
        """Return fiscal stance: expansionary, neutral, or contractionary"""
        if self.budget_deficit > 0.03 * self.spending:
            return "expansionary"
        elif self.budget_deficit < -0.03 * self.spending:
            return "contractionary"
        else:
            return "neutral"

    def get_state_dict(self) -> Dict[str, Any]:
        """Return current state for logging"""
        return {
            'spending': self.spending,
            'tax_revenue': self.tax_revenue,
            'transfers': self.transfers,
            'budget_deficit': self.budget_deficit,
            'public_debt': self.public_debt,
            'fiscal_stance': self.get_fiscal_stance()
        }


class CentralBank:
    """
    Central bank with monetary policy.

    Key features:
    - Interest rate policy (Taylor rule or discretionary)
    - Reserve provision to banks
    - Lender of last resort (crisis response)
    """

    def __init__(self):
        self.agent_type = AgentType.CENTRAL_BANK

        # Monetary policy
        self.interest_rate = 0.02  # Policy rate (2%)
        self.target_inflation = 0.02  # Inflation target
        self.target_unemployment = 0.05  # Unemployment target (dual mandate)

        # Taylor rule parameters
        self.use_taylor_rule = True
        self.taylor_inflation_coef = 1.5  # Response to inflation
        self.taylor_output_coef = 0.5  # Response to output gap

        # Crisis response
        self.crisis_mode = False
        self.lender_of_last_resort_active = False

        # Reserves provided to banking system
        self.total_reserves = 0.0

    def set_interest_rate(self, economy_state: Dict[str, Any]):
        """
        Set policy interest rate.

        Can use Taylor rule or discretionary policy.
        """
        if self.use_taylor_rule:
            self.taylor_rule(economy_state)
        else:
            # Discretionary: can be set by policy experiments
            pass

        # Zero lower bound
        self.interest_rate = max(0.0, self.interest_rate)

        # Upper bound (prevent instability)
        self.interest_rate = min(0.15, self.interest_rate)

    def taylor_rule(self, economy_state: Dict[str, Any]):
        """
        Taylor rule for interest rate.

        i = r* + π* + α(π - π*) + β(y - y*)

        Where:
        - r* = neutral real rate
        - π* = inflation target
        - π = actual inflation
        - y - y* = output gap (proxied by unemployment gap)
        """
        neutral_rate = 0.02
        inflation = economy_state.get('inflation', 0.02)
        unemployment_rate = economy_state.get('unemployment_rate', 0.05)

        # Inflation gap
        inflation_gap = inflation - self.target_inflation

        # Output gap (proxied by unemployment gap, inversed)
        unemployment_gap = self.target_unemployment - unemployment_rate

        # Taylor rule
        self.interest_rate = (
            neutral_rate +
            self.target_inflation +
            self.taylor_inflation_coef * inflation_gap +
            self.taylor_output_coef * unemployment_gap
        )

    def detect_crisis(self, economy_state: Dict[str, Any], banks: List) -> bool:
        """
        Detect financial crisis.

        Criteria:
        - Rapid increase in bankruptcies
        - Bank failures
        - Credit crunch
        - Sharp GDP decline
        """
        from .banks import Bank

        # Check for bank distress
        banks_in_distress = sum(1 for b in banks if b.capital_adequacy_ratio < 0.06)
        distress_rate = banks_in_distress / len(banks) if len(banks) > 0 else 0.0

        # Credit crunch
        avg_lending_standards = np.mean([b.lending_standards for b in banks]) if len(banks) > 0 else 0.5

        # GDP collapse
        gdp_growth = economy_state.get('gdp_growth', 0.0)

        # Crisis conditions
        if (distress_rate > 0.2 or
            avg_lending_standards > 0.8 or
            gdp_growth < -0.05):
            self.crisis_mode = True
            return True
        else:
            self.crisis_mode = False
            return False

    def crisis_response(self, banks: List):
        """
        Crisis response: lender of last resort.

        - Cut interest rates
        - Provide liquidity to banks
        - Relax capital requirements (forbearance)
        """
        from .banks import Bank

        if self.crisis_mode:
            # Emergency rate cut
            self.interest_rate = max(0.0, self.interest_rate - 0.02)

            # Provide reserves to distressed banks
            for bank in banks:
                if bank.capital_adequacy_ratio < 0.08:
                    # Recapitalize (simplified)
                    capital_injection = bank.capital * 0.1
                    bank.capital += capital_injection
                    self.lender_of_last_resort_active = True

    def get_monetary_stance(self) -> str:
        """Return monetary stance"""
        if self.interest_rate < 0.01:
            return "very_loose"
        elif self.interest_rate < 0.025:
            return "loose"
        elif self.interest_rate < 0.04:
            return "neutral"
        elif self.interest_rate < 0.06:
            return "tight"
        else:
            return "very_tight"

    def get_state_dict(self) -> Dict[str, Any]:
        """Return current state for logging"""
        return {
            'interest_rate': self.interest_rate,
            'target_inflation': self.target_inflation,
            'monetary_stance': self.get_monetary_stance(),
            'crisis_mode': self.crisis_mode,
            'lolr_active': self.lender_of_last_resort_active
        }


class PolicyExperiment:
    """
    Framework for policy experiments.

    Allows testing different policy scenarios:
    - Fiscal austerity vs expansion
    - Different monetary rules
    - Regulatory changes
    """

    def __init__(self, name: str):
        self.name = name
        self.interventions: List[Dict[str, Any]] = []

    def add_intervention(self, time: int, agent_type: str,
                        parameter: str, value: Any):
        """
        Add policy intervention at specific time.

        Example: At t=50, set government.spending = 150
        """
        self.interventions.append({
            'time': time,
            'agent_type': agent_type,
            'parameter': parameter,
            'value': value
        })

    def apply_interventions(self, t: int, government: Government,
                           central_bank: CentralBank) -> List[str]:
        """
        Apply interventions scheduled for time t.

        Returns list of interventions applied.
        """
        applied = []

        for intervention in self.interventions:
            if intervention['time'] == t:
                agent_type = intervention['agent_type']
                parameter = intervention['parameter']
                value = intervention['value']

                if agent_type == 'government':
                    setattr(government, parameter, value)
                    applied.append(f"Government: {parameter} = {value}")
                elif agent_type == 'central_bank':
                    setattr(central_bank, parameter, value)
                    applied.append(f"Central Bank: {parameter} = {value}")

        return applied

    @staticmethod
    def create_austerity_experiment() -> 'PolicyExperiment':
        """
        Create fiscal austerity experiment.

        Test impact of cutting spending in recession.
        """
        exp = PolicyExperiment("Fiscal Austerity")

        # Cut spending at t=50
        exp.add_intervention(50, 'government', 'government_spending_target', 70.0)

        # Reduce transfers
        exp.add_intervention(50, 'government', 'unemployment_benefit_rate', 0.3)

        return exp

    @staticmethod
    def create_qe_experiment() -> 'PolicyExperiment':
        """
        Create quantitative easing experiment.

        Test impact of ultra-low rates + bank support.
        """
        exp = PolicyExperiment("Quantitative Easing")

        # Cut rates to zero
        exp.add_intervention(50, 'central_bank', 'interest_rate', 0.0)
        exp.add_intervention(50, 'central_bank', 'use_taylor_rule', False)

        return exp

    @staticmethod
    def create_green_new_deal_experiment() -> 'PolicyExperiment':
        """
        Create Green New Deal / large fiscal expansion experiment.
        """
        exp = PolicyExperiment("Green New Deal")

        # Large increase in government spending
        exp.add_intervention(50, 'government', 'government_spending_target', 200.0)

        # Increase transfers
        exp.add_intervention(50, 'government', 'unemployment_benefit_rate', 0.7)

        # Accommodative monetary policy
        exp.add_intervention(50, 'central_bank', 'interest_rate', 0.01)

        return exp
