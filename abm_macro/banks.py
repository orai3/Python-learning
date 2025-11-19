"""
Banking Sector for ABM Macroeconomic Model

Implements endogenous money creation and credit rationing:
- Banks CREATE credit (not intermediation of loanable funds!)
- Credit rationing based on borrower risk (Stiglitz-Weiss)
- Regulatory constraints (capital requirements)
- Default risk and bank fragility
- Pro-cyclical credit dynamics

References:
- Minsky (1986): Financial Instability Hypothesis
- Stiglitz & Weiss (1981): Credit rationing
- Werner (2014): Credit creation theory of banking
- Godley & Lavoie (2007): Stock-flow consistent banking
"""

import numpy as np
from typing import Dict, List, Optional, Any
from .base import Agent, AgentType


class Bank(Agent):
    """
    Commercial bank that creates credit endogenously.

    Key heterodox features:
    - Credit creation NOT intermediation (banks create deposits by lending)
    - Credit rationing is the norm (not market-clearing interest rates)
    - Pro-cyclical lending (reinforces booms and busts)
    - Regulatory constraints matter
    """

    def __init__(self, bank_id: int, initial_capital: float = 100.0):
        super().__init__(bank_id, AgentType.BANK)

        # Balance sheet
        self.capital = initial_capital  # Equity
        self.reserves = initial_capital * 0.1  # Reserves at central bank
        self.loans: Dict[int, float] = {}  # Loans to firms (firm_id: amount)
        self.deposits: Dict[int, float] = {}  # Household deposits (household_id: amount)

        # Credit policy parameters
        self.capital_adequacy_ratio = 0.08  # Basel-style requirement
        self.target_car = 0.12  # Target (buffer above requirement)
        self.reserve_ratio = 0.03  # Reserve requirement

        # Risk assessment
        self.risk_aversion = np.random.uniform(0.3, 0.7)
        self.max_leverage = 12.0  # Maximum loans/capital ratio

        # Credit rationing criteria
        self.max_debt_service_ratio = 0.4  # Max debt service / revenue
        self.min_profit_rate = -0.1  # Tolerance for losses

        # Performance
        self.interest_income = 0.0
        self.interest_expense = 0.0
        self.loan_losses = 0.0
        self.profits = 0.0

        # Credit cycle dynamics
        self.credit_growth_rate = 0.0
        self.nonperforming_loan_ratio = 0.0
        self.lending_standards: float = 0.5  # 0=loose, 1=tight

    def step(self, t: int, economy_state: Dict[str, Any]):
        """
        Bank's decision sequence:

        1. Assess capital position
        2. Adjust lending standards based on performance and macro conditions
        3. Process loan applications
        4. Collect interest, handle defaults
        5. Pay interest on deposits
        """
        self.reset_flows()

        # Assess financial position
        self.assess_capital_adequacy()

        # Adjust lending standards (pro-cyclical!)
        self.adjust_lending_standards(economy_state)

    def assess_capital_adequacy(self):
        """
        Check capital adequacy ratio (CAR).

        CAR = Capital / Risk-Weighted Assets
        If CAR too low → tighten lending
        """
        total_loans = sum(self.loans.values())

        if total_loans > 0:
            self.capital_adequacy_ratio = self.capital / total_loans
        else:
            self.capital_adequacy_ratio = 1.0

        # Update leverage constraint
        if self.capital_adequacy_ratio < self.target_car:
            # Capital constrained → tighten lending
            self.lending_standards = min(1.0, self.lending_standards + 0.05)
        else:
            # Well-capitalized → can ease standards
            self.lending_standards = max(0.0, self.lending_standards - 0.02)

    def adjust_lending_standards(self, economy_state: Dict[str, Any]):
        """
        Adjust lending standards based on macro conditions.

        PRO-CYCLICAL: Banks ease standards in booms, tighten in busts.
        This amplifies business cycles (Minskyan instability).
        """
        # Recent default experience
        if self.nonperforming_loan_ratio > 0.1:
            # High defaults → tighten
            self.lending_standards = min(1.0, self.lending_standards + 0.1)
        elif self.nonperforming_loan_ratio < 0.03:
            # Low defaults → ease (disaster myopia)
            self.lending_standards = max(0.0, self.lending_standards - 0.05)

        # Macro conditions (pro-cyclical)
        gdp_growth = economy_state.get('gdp_growth', 0.0)
        if gdp_growth > 0.03:
            # Boom → ease standards (overconfidence)
            self.lending_standards = max(0.0, self.lending_standards - 0.03)
        elif gdp_growth < -0.02:
            # Recession → tighten (flight to quality)
            self.lending_standards = min(1.0, self.lending_standards + 0.05)

    def evaluate_loan_application(self, firm_id: int, amount: float,
                                  firm_state: Dict[str, Any]) -> float:
        """
        Evaluate loan application and determine amount to approve.

        Credit rationing based on:
        - Firm's financial position (Minsky: hedge/speculative/Ponzi)
        - Debt service capacity
        - Profitability
        - Bank's capital constraints
        - Current lending standards

        Returns: approved amount (may be less than requested, or zero)
        """
        # Check if we have lending capacity
        current_leverage = sum(self.loans.values()) / self.capital if self.capital > 0 else 0
        if current_leverage >= self.max_leverage:
            return 0.0  # Cannot lend - capital constrained

        # Assess borrower risk
        revenue = firm_state.get('revenue', 1.0)
        existing_debt = firm_state.get('debt', 0.0)
        profits = firm_state.get('profits', 0.0)
        capital = firm_state.get('capital', 1.0)
        financial_position = firm_state.get('financial_position', 'hedge')

        # Calculate risk metrics
        if revenue > 0:
            # Debt service on new loan
            new_debt_service = (existing_debt + amount) * 0.05  # Assume 5% rate
            debt_service_ratio = new_debt_service / revenue
        else:
            debt_service_ratio = 1.0  # Very risky

        if capital > 0:
            profit_rate = profits / capital
        else:
            profit_rate = -0.5

        # Credit scoring (simplified)
        risk_score = 0.0

        # 1. Debt service capacity
        if debt_service_ratio > self.max_debt_service_ratio:
            risk_score += 0.4
        else:
            risk_score += debt_service_ratio / self.max_debt_service_ratio * 0.3

        # 2. Profitability
        if profit_rate < self.min_profit_rate:
            risk_score += 0.3
        else:
            risk_score += max(0, -profit_rate * 0.2)

        # 3. Financial position (Minsky)
        if financial_position == 'ponzi':
            risk_score += 0.4
        elif financial_position == 'speculative':
            risk_score += 0.2
        else:  # hedge
            risk_score += 0.0

        # Adjust for lending standards (pro-cyclical)
        risk_threshold = 0.3 + 0.4 * self.lending_standards

        # Approve decision
        if risk_score <= risk_threshold:
            # Approve full amount
            approved = amount
        elif risk_score <= risk_threshold + 0.2:
            # Partial approval (rationing)
            approved = amount * (1 - (risk_score - risk_threshold) / 0.2)
        else:
            # Reject
            approved = 0.0

        # Check lending capacity constraint
        available_capacity = (self.max_leverage * self.capital - sum(self.loans.values()))
        approved = min(approved, available_capacity)

        return max(0.0, approved)

    def make_loan(self, firm_id: int, amount: float) -> bool:
        """
        Make loan to firm.

        CREATES deposits (endogenous money creation).
        This is NOT intermediation of existing savings!
        """
        if amount <= 0:
            return False

        # Create loan (asset)
        if firm_id in self.loans:
            self.loans[firm_id] += amount
        else:
            self.loans[firm_id] = amount

        # Create deposit (liability) - this is money creation!
        # In full model, deposit goes to firm's account
        # Simplified here: just track total deposits

        return True

    def collect_interest(self, interest_rate: float) -> float:
        """
        Collect interest payments on loans.

        Some firms may default → loan losses.
        """
        total_interest = 0.0

        for firm_id, loan_amount in list(self.loans.items()):
            interest = loan_amount * interest_rate
            total_interest += interest

        self.interest_income = total_interest
        self.state.income += total_interest

        return total_interest

    def handle_default(self, firm_id: int, recovery_rate: float = 0.3):
        """
        Handle firm default.

        Loan losses reduce bank capital → credit crunch (contagion).
        """
        if firm_id in self.loans:
            loan_amount = self.loans[firm_id]
            loss = loan_amount * (1 - recovery_rate)

            self.loan_losses += loss
            self.capital -= loss  # Reduces bank equity

            # Write off loan
            self.loans[firm_id] = loan_amount * recovery_rate

            # Update NPL ratio
            total_loans = sum(self.loans.values())
            if total_loans > 0:
                self.nonperforming_loan_ratio = sum(
                    l for l in self.loans.values() if l > 0
                ) / total_loans

            # If bank capital depleted → bank failure (systemic crisis)
            if self.capital < 0.1 * sum(self.loans.values()):
                self.bankruptcy_flag = True

    def pay_deposit_interest(self, interest_rate: float) -> float:
        """Pay interest on household deposits"""
        deposit_rate = interest_rate * 0.5  # Deposits pay less than loan rate
        total_deposits = sum(self.deposits.values())

        interest_paid = total_deposits * deposit_rate
        self.interest_expense = interest_paid
        self.state.expenditure += interest_paid

        return interest_paid

    def calculate_profits(self) -> float:
        """Calculate bank profits"""
        self.profits = self.interest_income - self.interest_expense - self.loan_losses
        self.capital += self.profits * 0.9  # Retain 90%

        return self.profits

    def get_total_credit(self) -> float:
        """Total credit outstanding"""
        return sum(self.loans.values())

    def get_state_dict(self) -> Dict[str, Any]:
        """Return current state for logging"""
        return {
            'id': self.agent_id,
            'capital': self.capital,
            'total_loans': sum(self.loans.values()),
            'total_deposits': sum(self.deposits.values()),
            'capital_adequacy_ratio': self.capital_adequacy_ratio,
            'lending_standards': self.lending_standards,
            'npl_ratio': self.nonperforming_loan_ratio,
            'profits': self.profits,
            'interest_income': self.interest_income,
            'loan_losses': self.loan_losses
        }


class CreditMarket:
    """
    Credit market with rationing.

    NOT a market-clearing mechanism - rationing is fundamental.
    """

    def __init__(self, banks: List[Bank]):
        self.banks = banks
        self.total_credit_requested = 0.0
        self.total_credit_approved = 0.0
        self.rationing_rate = 0.0

    def process_loan_applications(self, firms: List, interest_rate: float) -> Dict[str, Any]:
        """
        Process loan applications from firms.

        Returns statistics on credit rationing.
        """
        from .firms import Firm  # Import here to avoid circular dependency

        total_requested = 0.0
        total_approved = 0.0
        n_rationed = 0

        for firm in firms:
            if not firm.active or firm.loan_request <= 0:
                continue

            total_requested += firm.loan_request

            # Firm applies to random bank (simplified - could use relationships)
            bank = np.random.choice(self.banks)

            # Bank evaluates application
            approved_amount = bank.evaluate_loan_application(
                firm.agent_id,
                firm.loan_request,
                firm.get_state_dict()
            )

            total_approved += approved_amount

            # Make loan if approved
            if approved_amount > 0:
                bank.make_loan(firm.agent_id, approved_amount)
                firm.execute_investment(approved_amount)
            else:
                firm.execute_investment(0.0)

            # Track rationing
            if approved_amount < firm.loan_request * 0.95:
                n_rationed += 1

        self.total_credit_requested = total_requested
        self.total_credit_approved = total_approved

        if total_requested > 0:
            self.rationing_rate = 1.0 - (total_approved / total_requested)
        else:
            self.rationing_rate = 0.0

        return {
            'credit_requested': total_requested,
            'credit_approved': total_approved,
            'rationing_rate': self.rationing_rate,
            'n_firms_rationed': n_rationed
        }

    def handle_defaults(self, firms: List):
        """
        Process firm bankruptcies and bank loan losses.

        Creates feedback loop: defaults → loan losses → tighter credit → more defaults
        """
        from .firms import Firm

        for firm in firms:
            if firm.bankruptcy_flag:
                # Find banks with loans to this firm
                for bank in self.banks:
                    if firm.agent_id in bank.loans:
                        # Bank suffers loss
                        recovery_rate = min(firm.capital / firm.state.debt, 0.5) if firm.state.debt > 0 else 0.0
                        bank.handle_default(firm.agent_id, recovery_rate)

        # Check for bank failures
        failed_banks = [b for b in self.banks if b.bankruptcy_flag]
        if len(failed_banks) > 0:
            print(f"WARNING: {len(failed_banks)} banks failed - systemic crisis!")

    def get_aggregate_credit(self) -> float:
        """Total credit in economy"""
        return sum(bank.get_total_credit() for bank in self.banks)

    def get_average_lending_standards(self) -> float:
        """Average lending standards across banks"""
        if len(self.banks) == 0:
            return 0.5
        return np.mean([bank.lending_standards for bank in self.banks])
