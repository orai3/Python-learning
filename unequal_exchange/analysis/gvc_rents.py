"""
Global Value Chain Rent Extraction Analysis

Analyzes how rents are extracted in global value chains through:
1. Monopoly/oligopoly power at different chain segments
2. Intellectual property control
3. Brand value capture
4. Lead firm governance
5. Barriers to entry (upgrading constraints)

Based on:
- Kaplinsky & Morris (2001): A Handbook for Value Chain Research
- Gereffi (1994): The Organization of Buyer-Driven Commodity Chains
- Milberg & Winkler (2013): Outsourcing Economics
- Starrs (2014): The Chimera of Global Convergence
- Durand & Milberg (2020): Intellectual Monopoly in Global Value Chains

Value Chain Governance Types:
1. Market: Simple transactions, low coordination
2. Modular: Complex transactions, codified interfaces
3. Relational: Complex transactions, tacit knowledge
4. Captive: Small suppliers dependent on large buyers
5. Hierarchy: Vertical integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class GovernanceType(Enum):
    """GVC governance types (Gereffi et al. 2005)"""
    MARKET = "market"
    MODULAR = "modular"
    RELATIONAL = "relational"
    CAPTIVE = "captive"
    HIERARCHY = "hierarchy"


@dataclass
class ValueChainSegment:
    """Represents a segment in the value chain"""
    name: str
    country: str
    value_added: float
    labor_cost: float
    capital_intensity: float
    market_power: float  # 0-1 scale
    barriers_to_entry: float  # 0-1 scale
    ip_intensity: float  # 0-1 scale


class GVCRentExtractor:
    """
    Analyzes rent extraction patterns in global value chains.

    Distinguishes between:
    - Competitive returns (normal profits)
    - Technological rents (innovation-based)
    - Monopoly rents (market power)
    - Organizational rents (chain governance)
    - IP rents (patents, trademarks, copyrights)
    """

    def __init__(self):
        """Initialize GVC rent extractor"""
        self.value_chains: Dict[str, List[ValueChainSegment]] = {}
        self.competitive_profit_rate = 0.10  # Benchmark competitive rate

    def add_value_chain(self, chain_name: str, segments: List[ValueChainSegment]):
        """
        Add a value chain for analysis.

        Args:
            chain_name: Name of value chain (e.g., "iPhone", "Coffee", "Apparel")
            segments: List of chain segments
        """
        self.value_chains[chain_name] = segments

    def calculate_value_distribution(self, chain_name: str) -> pd.DataFrame:
        """
        Calculate distribution of value across chain segments.

        Args:
            chain_name: Value chain to analyze

        Returns:
            DataFrame with value distribution
        """
        if chain_name not in self.value_chains:
            raise ValueError(f"Chain {chain_name} not found")

        segments = self.value_chains[chain_name]

        results = []
        total_value = sum(s.value_added for s in segments)

        for segment in segments:
            value_share = (segment.value_added / total_value * 100) if total_value > 0 else 0

            # Estimate rents
            rents = self._estimate_rents(segment)

            results.append({
                'segment': segment.name,
                'country': segment.country,
                'value_added': segment.value_added,
                'value_share_pct': value_share,
                'labor_cost': segment.labor_cost,
                'profit': segment.value_added - segment.labor_cost,
                'profit_rate': ((segment.value_added - segment.labor_cost) / segment.labor_cost * 100)
                              if segment.labor_cost > 0 else 0,
                'monopoly_rent': rents['monopoly'],
                'ip_rent': rents['ip'],
                'technological_rent': rents['technological'],
                'total_rent': rents['total'],
                'rent_share_of_value': (rents['total'] / segment.value_added * 100)
                                      if segment.value_added > 0 else 0
            })

        return pd.DataFrame(results)

    def _estimate_rents(self, segment: ValueChainSegment) -> Dict[str, float]:
        """
        Estimate different types of rents in a segment.

        Args:
            segment: Value chain segment

        Returns:
            Dictionary with rent estimates
        """
        profit = segment.value_added - segment.labor_cost

        # Competitive profit (benchmark)
        competitive_profit = segment.labor_cost * self.competitive_profit_rate

        # Excess profit (rent)
        excess_profit = max(0, profit - competitive_profit)

        # Decompose rents
        monopoly_rent = excess_profit * segment.market_power
        ip_rent = excess_profit * segment.ip_intensity
        technological_rent = excess_profit * (1 - segment.market_power - segment.ip_intensity) * 0.5

        # Ensure no double counting
        total_rent = min(excess_profit, monopoly_rent + ip_rent + technological_rent)

        return {
            'monopoly': monopoly_rent,
            'ip': ip_rent,
            'technological': technological_rent,
            'total': total_rent
        }

    def analyze_smile_curve(self, chain_name: str) -> Dict[str, any]:
        """
        Analyze the "smile curve" - high value at ends, low in middle.

        Typical in electronics: High value in R&D/design and marketing/brand,
        low value in manufacturing.

        Args:
            chain_name: Value chain to analyze

        Returns:
            Smile curve analysis
        """
        dist = self.calculate_value_distribution(chain_name)

        # Identify upstream, midstream, downstream
        n = len(dist)
        if n < 3:
            return {}

        upstream_share = dist.iloc[:n//3]['value_share_pct'].mean()
        midstream_share = dist.iloc[n//3:2*n//3]['value_share_pct'].mean()
        downstream_share = dist.iloc[2*n//3:]['value_share_pct'].mean()

        # Smile intensity
        smile_intensity = (upstream_share + downstream_share) / (2 * midstream_share) if midstream_share > 0 else 0

        return {
            'upstream_value_share': upstream_share,
            'midstream_value_share': midstream_share,
            'downstream_value_share': downstream_share,
            'smile_intensity': smile_intensity,
            'has_smile_curve': smile_intensity > 1.0,
            'interpretation': 'Strong smile curve' if smile_intensity > 1.5 else
                            'Moderate smile curve' if smile_intensity > 1.0 else
                            'No smile curve (relatively even distribution)'
        }

    def calculate_lead_firm_extraction(self, chain_name: str,
                                      lead_firm_segments: List[str]) -> Dict[str, float]:
        """
        Calculate how much value lead firms extract vs suppliers.

        Args:
            chain_name: Value chain name
            lead_firm_segments: Segments controlled by lead firm

        Returns:
            Lead firm extraction metrics
        """
        dist = self.calculate_value_distribution(chain_name)

        # Lead firm value
        lead_firm_value = dist[dist['segment'].isin(lead_firm_segments)]['value_added'].sum()
        lead_firm_labor = dist[dist['segment'].isin(lead_firm_segments)]['labor_cost'].sum()
        lead_firm_rents = dist[dist['segment'].isin(lead_firm_segments)]['total_rent'].sum()

        # Supplier value
        supplier_value = dist[~dist['segment'].isin(lead_firm_segments)]['value_added'].sum()
        supplier_labor = dist[~dist['segment'].isin(lead_firm_segments)]['labor_cost'].sum()
        supplier_rents = dist[~dist['segment'].isin(lead_firm_segments)]['total_rent'].sum()

        total_value = lead_firm_value + supplier_value

        return {
            'lead_firm_value_share': (lead_firm_value / total_value * 100) if total_value > 0 else 0,
            'supplier_value_share': (supplier_value / total_value * 100) if total_value > 0 else 0,
            'lead_firm_rent_share': (lead_firm_rents / (lead_firm_rents + supplier_rents) * 100)
                                   if (lead_firm_rents + supplier_rents) > 0 else 0,
            'lead_firm_profit_rate': ((lead_firm_value - lead_firm_labor) / lead_firm_labor * 100)
                                    if lead_firm_labor > 0 else 0,
            'supplier_profit_rate': ((supplier_value - supplier_labor) / supplier_labor * 100)
                                   if supplier_labor > 0 else 0,
            'extraction_ratio': ((lead_firm_value / lead_firm_labor) / (supplier_value / supplier_labor))
                               if supplier_value > 0 and supplier_labor > 0 and lead_firm_labor > 0 else 0
        }

    def analyze_upgrading_barriers(self, chain_name: str) -> pd.DataFrame:
        """
        Analyze barriers to upgrading for peripheral countries.

        Upgrading types (Humphrey & Schmitz 2002):
        1. Process upgrading: More efficient production
        2. Product upgrading: Higher value products
        3. Functional upgrading: New functions in chain
        4. Chain upgrading: Move to new chains

        Args:
            chain_name: Value chain name

        Returns:
            DataFrame with upgrading analysis
        """
        segments = self.value_chains[chain_name]

        results = []

        for i, segment in enumerate(segments):
            # Barriers to moving to next segment
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                capital_barrier = abs(next_segment.capital_intensity - segment.capital_intensity)
                ip_barrier = next_segment.ip_intensity - segment.ip_intensity
                market_power_barrier = next_segment.market_power - segment.market_power

                upgrading_feasibility = 1.0 - min(1.0,
                    (capital_barrier + max(0, ip_barrier) + max(0, market_power_barrier)) / 3
                )
            else:
                upgrading_feasibility = 0.0

            results.append({
                'segment': segment.name,
                'country': segment.country,
                'capital_intensity': segment.capital_intensity,
                'ip_intensity': segment.ip_intensity,
                'market_power': segment.market_power,
                'barriers_to_entry': segment.barriers_to_entry,
                'upgrading_feasibility': upgrading_feasibility,
                'locked_in': upgrading_feasibility < 0.3
            })

        return pd.DataFrame(results)

    def calculate_buyer_driven_squeeze(self, chain_name: str,
                                       buyer_segment: str,
                                       supplier_segments: List[str]) -> Dict[str, float]:
        """
        Analyze buyer-driven commodity chains and supplier squeeze.

        In buyer-driven chains (apparel, electronics), large retailers/brands
        squeeze suppliers by using market power to drive down prices.

        Args:
            chain_name: Value chain name
            buyer_segment: Segment representing buyer (e.g., "Retail/Brand")
            supplier_segments: Manufacturing supplier segments

        Returns:
            Squeeze analysis
        """
        dist = self.calculate_value_distribution(chain_name)

        buyer_data = dist[dist['segment'] == buyer_segment].iloc[0]
        supplier_data = dist[dist['segment'].isin(supplier_segments)]

        avg_supplier_profit_rate = supplier_data['profit_rate'].mean()
        buyer_profit_rate = buyer_data['profit_rate']

        # Squeeze index
        squeeze_index = (buyer_profit_rate - avg_supplier_profit_rate) / avg_supplier_profit_rate \
                       if avg_supplier_profit_rate > 0 else 0

        return {
            'buyer_profit_rate': buyer_profit_rate,
            'avg_supplier_profit_rate': avg_supplier_profit_rate,
            'profit_rate_gap': buyer_profit_rate - avg_supplier_profit_rate,
            'squeeze_index': squeeze_index,
            'buyer_value_share': buyer_data['value_share_pct'],
            'supplier_value_share': supplier_data['value_share_pct'].sum(),
            'buyer_rent_extraction': buyer_data['total_rent'],
            'supplier_total_rent': supplier_data['total_rent'].sum()
        }

    def estimate_global_rent_flows(self) -> pd.DataFrame:
        """
        Estimate rent flows across countries in all value chains.

        Returns:
            DataFrame with rent flows by country
        """
        results = []

        for chain_name, segments in self.value_chains.items():
            dist = self.calculate_value_distribution(chain_name)

            for _, row in dist.iterrows():
                results.append({
                    'value_chain': chain_name,
                    'segment': row['segment'],
                    'country': row['country'],
                    'value_added': row['value_added'],
                    'monopoly_rent': row['monopoly_rent'],
                    'ip_rent': row['ip_rent'],
                    'technological_rent': row['technological_rent'],
                    'total_rent': row['total_rent']
                })

        df = pd.DataFrame(results)

        # Aggregate by country
        country_rents = df.groupby('country').agg({
            'value_added': 'sum',
            'monopoly_rent': 'sum',
            'ip_rent': 'sum',
            'technological_rent': 'sum',
            'total_rent': 'sum'
        }).reset_index()

        country_rents['rent_share_of_va'] = (
            country_rents['total_rent'] / country_rents['value_added'] * 100
        )

        return country_rents.sort_values('total_rent', ascending=False)

    def simulate_platform_capitalism_rents(self, platform_name: str,
                                          platform_value_share: float = 0.30,
                                          total_transaction_value: float = 1000) -> Dict[str, float]:
        """
        Simulate rent extraction in platform capitalism.

        Platform companies (Amazon, Alibaba, Uber) extract rents through:
        1. Network effects (monopoly power)
        2. Data extraction and monetization
        3. Algorithmic control
        4. Infrastructure control

        Args:
            platform_name: Platform name
            platform_value_share: Share of transaction value captured by platform
            total_transaction_value: Total GMV

        Returns:
            Platform rent analysis
        """
        platform_capture = total_transaction_value * platform_value_share
        producer_capture = total_transaction_value * (1 - platform_value_share)

        # Estimate platform costs (relatively low for digital platforms)
        platform_costs = platform_capture * 0.30  # 30% of revenue
        platform_rents = platform_capture - platform_costs

        return {
            'total_transaction_value': total_transaction_value,
            'platform_value_capture': platform_capture,
            'producer_value_capture': producer_capture,
            'platform_costs': platform_costs,
            'platform_rents': platform_rents,
            'platform_rent_rate': (platform_rents / platform_costs * 100) if platform_costs > 0 else 0,
            'rent_extraction_rate': (platform_rents / total_transaction_value * 100)
        }
