"""
Transfer Pricing and Profit Shifting Analysis

Transfer pricing manipulation allows multinational corporations to shift
profits from high-tax peripheral countries to low-tax havens, representing
a major mechanism of value extraction.

Key Concepts:
1. Transfer pricing: Prices charged for intra-firm transactions
2. Arm's length principle: Prices should match market rates
3. Profit shifting: Strategic mis-pricing to relocate profits
4. Tax havens: Low/no tax jurisdictions for booking profits
5. Illicit financial flows: Broader category including trade misinvoicing

Estimation Methods:
1. Price filter methods: Compare declared vs market prices
2. Regression-based approaches: Expected vs actual trade patterns
3. Country-by-country reporting: Profit/revenue mismatches
4. Mirror trade statistics: Discrepancies in bilateral trade

References:
- Cobham & Janský (2018): Global distribution of revenue loss from tax avoidance
- Crivelli et al. (2016): Base Erosion, Profit Shifting and Developing Countries
- Kar & Spanjers (2015): Illicit Financial Flows from Developing Countries
- Zucman (2014): Taxing across Borders
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class TradeTransaction:
    """Represents a trade transaction"""
    product: str
    exporter: str
    importer: str
    quantity: float
    declared_price: float
    year: int


class TransferPricingAnalyzer:
    """
    Analyzes transfer pricing manipulation and profit shifting.
    """

    def __init__(self):
        """Initialize transfer pricing analyzer"""
        self.transactions: List[TradeTransaction] = []
        self.market_prices: Dict[str, float] = {}  # Product -> market price
        self.tax_rates: Dict[str, float] = {}  # Country -> corporate tax rate

    def add_transaction(self, transaction: TradeTransaction):
        """
        Add trade transaction.

        Args:
            transaction: Trade transaction data
        """
        self.transactions.append(transaction)

    def set_market_price(self, product: str, price: float):
        """
        Set market/reference price for product.

        Args:
            product: Product identifier
            price: Market price
        """
        self.market_prices[product] = price

    def set_tax_rate(self, country: str, rate: float):
        """
        Set corporate tax rate for country.

        Args:
            country: Country code
            rate: Tax rate (0-1)
        """
        self.tax_rates[country] = rate

    def calculate_price_gaps(self) -> pd.DataFrame:
        """
        Calculate gaps between declared and market prices.

        Large gaps suggest transfer pricing manipulation.

        Returns:
            DataFrame with price gaps
        """
        results = []

        for trans in self.transactions:
            if trans.product in self.market_prices:
                market_price = self.market_prices[trans.product]
                price_gap = trans.declared_price - market_price
                price_gap_pct = (price_gap / market_price * 100) if market_price > 0 else 0

                # Determine direction of mispricing
                if price_gap_pct > 20:  # Over-invoicing
                    mispricing_type = "over-invoicing"
                elif price_gap_pct < -20:  # Under-invoicing
                    mispricing_type = "under-invoicing"
                else:
                    mispricing_type = "normal"

                value_gap = price_gap * trans.quantity

                results.append({
                    'product': trans.product,
                    'exporter': trans.exporter,
                    'importer': trans.importer,
                    'year': trans.year,
                    'quantity': trans.quantity,
                    'declared_price': trans.declared_price,
                    'market_price': market_price,
                    'price_gap': price_gap,
                    'price_gap_pct': price_gap_pct,
                    'value_gap': value_gap,
                    'mispricing_type': mispricing_type
                })

        return pd.DataFrame(results)

    def estimate_profit_shifting(self, company_data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate profit shifting using country-by-country data.

        Method: Compare actual profit margins with expected based on activity.
        High profits in low-tax havens + low profits in high-tax operating
        countries = likely profit shifting.

        Args:
            company_data: DataFrame with columns: country, revenue, profit, employees, assets

        Returns:
            Profit shifting estimates
        """
        # Add tax rates
        company_data['tax_rate'] = company_data['country'].map(self.tax_rates)

        # Calculate profit margins
        company_data['profit_margin'] = (company_data['profit'] / company_data['revenue'] * 100) \
                                       if 'revenue' in company_data.columns else 0

        # Expected profit based on activity (employees + assets)
        total_employees = company_data['employees'].sum()
        total_profit = company_data['profit'].sum()

        company_data['expected_profit'] = (
            company_data['employees'] / total_employees * total_profit
        )

        company_data['excess_profit'] = company_data['profit'] - company_data['expected_profit']

        # Identify tax havens (tax rate < 15%)
        company_data['tax_haven'] = company_data['tax_rate'] < 0.15

        # Profit in tax havens vs operating countries
        haven_profit = company_data[company_data['tax_haven']]['profit'].sum()
        haven_expected = company_data[company_data['tax_haven']]['expected_profit'].sum()
        haven_excess = haven_profit - haven_expected

        operating_profit = company_data[~company_data['tax_haven']]['profit'].sum()
        operating_expected = company_data[~company_data['tax_haven']]['expected_profit'].sum()
        operating_deficit = operating_expected - operating_profit

        return {
            'total_profit': total_profit,
            'tax_haven_profit': haven_profit,
            'tax_haven_expected_profit': haven_expected,
            'tax_haven_excess_profit': haven_excess,
            'operating_country_profit': operating_profit,
            'operating_country_expected_profit': operating_expected,
            'operating_country_deficit': operating_deficit,
            'estimated_profit_shifted': haven_excess,
            'profit_shifting_pct': (haven_excess / total_profit * 100) if total_profit > 0 else 0
        }

    def calculate_tax_loss(self, country: str, profit_shifted: float) -> Dict[str, float]:
        """
        Calculate tax revenue loss from profit shifting.

        Args:
            country: Country losing tax revenue
            profit_shifted: Amount of profit shifted out

        Returns:
            Tax loss estimates
        """
        if country not in self.tax_rates:
            return {}

        tax_rate = self.tax_rates[country]
        tax_loss = profit_shifted * tax_rate

        return {
            'profit_shifted_out': profit_shifted,
            'corporate_tax_rate': tax_rate * 100,
            'tax_revenue_loss': tax_loss,
            'tax_loss_as_pct_shifted': tax_rate * 100
        }

    def analyze_trade_misinvoicing(self, bilateral_trade: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze trade misinvoicing using mirror statistics.

        Method: Compare country A's reported exports to B with B's reported
        imports from A. Large discrepancies suggest misinvoicing.

        Args:
            bilateral_trade: DataFrame with columns: exporter, importer,
                           export_value, import_value, year

        Returns:
            Misinvoicing estimates
        """
        results = []

        for _, row in bilateral_trade.iterrows():
            exporter = row['exporter']
            importer = row['importer']
            export_value = row['export_value']
            import_value = row['import_value']

            # Account for CIF/FOB difference (~10%)
            expected_import = export_value * 1.10

            # Gap
            gap = import_value - expected_import
            gap_pct = (gap / expected_import * 100) if expected_import > 0 else 0

            # Classify
            if gap_pct > 20:
                classification = "import_over-invoicing"
            elif gap_pct < -20:
                classification = "export_over-invoicing"
            else:
                classification = "normal"

            results.append({
                'exporter': exporter,
                'importer': importer,
                'year': row['year'],
                'export_value': export_value,
                'import_value': import_value,
                'expected_import_value': expected_import,
                'gap': gap,
                'gap_pct': gap_pct,
                'classification': classification,
                'potential_illicit_flow': abs(gap) if abs(gap_pct) > 20 else 0
            })

        return pd.DataFrame(results)

    def estimate_illicit_financial_flows(self, country: str,
                                        misinvoicing_data: pd.DataFrame) -> Dict[str, float]:
        """
        Estimate illicit financial flows (IFFs) for a country.

        IFFs include:
        1. Trade misinvoicing
        2. Profit shifting
        3. Abusive transfer pricing
        4. Other capital flight

        Args:
            country: Country to analyze
            misinvoicing_data: Results from analyze_trade_misinvoicing

        Returns:
            IFF estimates
        """
        # Inflows (capital flight from country)
        export_misinvoicing = misinvoicing_data[
            (misinvoicing_data['exporter'] == country) &
            (misinvoicing_data['classification'] == 'export_over-invoicing')
        ]['potential_illicit_flow'].sum()

        import_misinvoicing = misinvoicing_data[
            (misinvoicing_data['importer'] == country) &
            (misinvoicing_data['classification'] == 'import_over-invoicing')
        ]['potential_illicit_flow'].sum()

        # Total IFF outflow
        total_iff = export_misinvoicing + import_misinvoicing

        return {
            'export_misinvoicing': export_misinvoicing,
            'import_misinvoicing': import_misinvoicing,
            'total_trade_misinvoicing': total_iff,
            'country': country
        }

    def calculate_base_erosion_index(self, country: str,
                                     corporate_tax_base: float,
                                     profit_shifted: float) -> Dict[str, float]:
        """
        Calculate base erosion and profit shifting (BEPS) index.

        Args:
            country: Country to analyze
            corporate_tax_base: Domestic corporate tax base
            profit_shifted: Profit shifted out

        Returns:
            BEPS metrics
        """
        if corporate_tax_base == 0:
            return {}

        base_erosion_rate = (profit_shifted / corporate_tax_base * 100)

        return {
            'corporate_tax_base': corporate_tax_base,
            'profit_shifted_out': profit_shifted,
            'base_erosion_rate_pct': base_erosion_rate,
            'remaining_tax_base': corporate_tax_base - profit_shifted,
            'erosion_severity': 'severe' if base_erosion_rate > 30 else
                              'moderate' if base_erosion_rate > 15 else
                              'low'
        }


class IPRentAnalyzer:
    """
    Analyzes intellectual property rent flows.

    IP rents include:
    1. Patent royalties
    2. Trademark/brand licensing
    3. Copyright fees
    4. Technology transfer fees
    5. Franchise fees
    """

    def __init__(self):
        """Initialize IP rent analyzer"""
        self.ip_flows: List[Dict] = []

    def add_ip_flow(self, payer_country: str, recipient_country: str,
                    amount: float, ip_type: str, year: int):
        """
        Add IP payment flow.

        Args:
            payer_country: Country making payment
            recipient_country: Country receiving payment
            amount: Payment amount
            ip_type: Type of IP (patent, trademark, etc.)
            year: Year
        """
        self.ip_flows.append({
            'payer': payer_country,
            'recipient': recipient_country,
            'amount': amount,
            'type': ip_type,
            'year': year
        })

    def calculate_net_flows(self) -> pd.DataFrame:
        """
        Calculate net IP rent flows by country.

        Returns:
            DataFrame with net flows
        """
        df = pd.DataFrame(self.ip_flows)

        results = []

        countries = set(df['payer'].unique()) | set(df['recipient'].unique())

        for country in countries:
            payments = df[df['payer'] == country]['amount'].sum()
            receipts = df[df['recipient'] == country]['amount'].sum()
            net_flow = receipts - payments

            results.append({
                'country': country,
                'ip_payments_out': payments,
                'ip_receipts_in': receipts,
                'net_ip_flow': net_flow,
                'is_net_recipient': net_flow > 0
            })

        return pd.DataFrame(results)

    def analyze_monopoly_rents(self, market_price: float,
                              production_cost: float,
                              ip_royalty_rate: float) -> Dict[str, float]:
        """
        Analyze monopoly rents from IP.

        IP creates monopoly power, allowing prices above competitive levels.

        Args:
            market_price: Price charged (with IP markup)
            production_cost: Actual cost of production
            ip_royalty_rate: Royalty rate charged

        Returns:
            Monopoly rent analysis
        """
        # Competitive price (cost + normal profit)
        competitive_price = production_cost * 1.15  # 15% normal profit

        # Monopoly rent
        monopoly_rent_per_unit = market_price - competitive_price

        # IP royalty
        ip_royalty_per_unit = market_price * ip_royalty_rate

        return {
            'market_price': market_price,
            'production_cost': production_cost,
            'competitive_price': competitive_price,
            'monopoly_rent_per_unit': monopoly_rent_per_unit,
            'ip_royalty_per_unit': ip_royalty_per_unit,
            'monopoly_markup_pct': ((market_price - competitive_price) / competitive_price * 100)
                                  if competitive_price > 0 else 0,
            'ip_royalty_rate_pct': ip_royalty_rate * 100
        }

    def estimate_pharmaceutical_rents(self, drug_price: float,
                                     generic_price: float,
                                     quantity: float) -> Dict[str, float]:
        """
        Estimate monopoly rents in pharmaceutical sector.

        Pharmaceuticals show extreme IP rent extraction:
        - Patented drugs: $1000s
        - Generic equivalents: $10s

        Args:
            drug_price: Patented drug price
            generic_price: Generic equivalent price
            quantity: Units sold

        Returns:
            Pharmaceutical rent analysis
        """
        rent_per_unit = drug_price - generic_price
        total_rent = rent_per_unit * quantity

        return {
            'patented_price': drug_price,
            'generic_price': generic_price,
            'monopoly_rent_per_unit': rent_per_unit,
            'quantity': quantity,
            'total_monopoly_rent': total_rent,
            'markup_over_generic_pct': ((drug_price - generic_price) / generic_price * 100)
                                      if generic_price > 0 else 0
        }

    def calculate_north_south_ip_flows(self, flows_df: pd.DataFrame,
                                       core_countries: List[str],
                                       periphery_countries: List[str]) -> Dict[str, float]:
        """
        Calculate IP flows from periphery to core.

        Args:
            flows_df: DataFrame with IP flows
            core_countries: List of core countries
            periphery_countries: List of periphery countries

        Returns:
            North-South IP flow analysis
        """
        # Flows from periphery to core
        south_to_north = flows_df[
            flows_df['payer'].isin(periphery_countries) &
            flows_df['recipient'].isin(core_countries)
        ]['amount'].sum()

        # Flows from core to periphery (minimal)
        north_to_south = flows_df[
            flows_df['payer'].isin(core_countries) &
            flows_df['recipient'].isin(periphery_countries)
        ]['amount'].sum()

        net_south_to_north = south_to_north - north_to_south

        return {
            'periphery_to_core_payments': south_to_north,
            'core_to_periphery_payments': north_to_south,
            'net_periphery_to_core': net_south_to_north,
            'asymmetry_ratio': (south_to_north / north_to_south) if north_to_south > 0 else float('inf')
        }
