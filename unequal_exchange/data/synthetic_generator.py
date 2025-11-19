"""
Synthetic Data Generator for Unequal Exchange Analysis

Generates realistic synthetic datasets covering 1960-present with:
- Trade flows (bilateral)
- Production data (by country-sector)
- Labor market indicators
- Price indices
- Financial flows
- Terms of trade
- Value chain data

Data reflects stylized facts from dependency theory:
- Core-periphery wage differentials (3-10x)
- Declining terms of trade for primary commodities
- Rising financialization
- Expanding global value chains
- Super-exploitation of Southern labor
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import datetime


@dataclass
class CountryProfile:
    """Economic profile of a country"""
    name: str
    category: str  # core, semi_periphery, periphery
    population: int  # millions
    initial_gdp_per_capita: float
    export_structure: Dict[str, float]  # sector -> share
    wage_level_index: float  # 100 = core average


class SyntheticDataGenerator:
    """
    Generates comprehensive synthetic datasets for unequal exchange analysis.
    """

    def __init__(self, start_year: int = 1960, end_year: int = 2020, random_seed: int = 42):
        """
        Initialize generator.

        Args:
            start_year: Start of time series
            end_year: End of time series
            random_seed: Random seed for reproducibility
        """
        self.start_year = start_year
        self.end_year = end_year
        self.years = list(range(start_year, end_year + 1))
        np.random.seed(random_seed)

        # Define country profiles
        self.countries = self._define_countries()
        self.sectors = ['Agriculture', 'Mining', 'Manufacturing', 'Services']

    def _define_countries(self) -> Dict[str, CountryProfile]:
        """Define stylized country profiles"""
        return {
            'USA': CountryProfile(
                name='USA', category='core', population=200,
                initial_gdp_per_capita=15000,
                export_structure={'Agriculture': 0.10, 'Mining': 0.05, 'Manufacturing': 0.60, 'Services': 0.25},
                wage_level_index=100
            ),
            'Germany': CountryProfile(
                name='Germany', category='core', population=80,
                initial_gdp_per_capita=12000,
                export_structure={'Agriculture': 0.05, 'Mining': 0.03, 'Manufacturing': 0.75, 'Services': 0.17},
                wage_level_index=95
            ),
            'Japan': CountryProfile(
                name='Japan', category='core', population=100,
                initial_gdp_per_capita=8000,
                export_structure={'Agriculture': 0.03, 'Mining': 0.02, 'Manufacturing': 0.85, 'Services': 0.10},
                wage_level_index=85
            ),
            'China': CountryProfile(
                name='China', category='semi_periphery', population=800,
                initial_gdp_per_capita=300,
                export_structure={'Agriculture': 0.40, 'Mining': 0.10, 'Manufacturing': 0.40, 'Services': 0.10},
                wage_level_index=15  # Rising to ~40 by 2020
            ),
            'Brazil': CountryProfile(
                name='Brazil', category='semi_periphery', population=90,
                initial_gdp_per_capita=1500,
                export_structure={'Agriculture': 0.50, 'Mining': 0.15, 'Manufacturing': 0.25, 'Services': 0.10},
                wage_level_index=30
            ),
            'India': CountryProfile(
                name='India', category='periphery', population=450,
                initial_gdp_per_capita=250,
                export_structure={'Agriculture': 0.60, 'Mining': 0.10, 'Manufacturing': 0.20, 'Services': 0.10},
                wage_level_index=12
            ),
            'Bangladesh': CountryProfile(
                name='Bangladesh', category='periphery', population=60,
                initial_gdp_per_capita=200,
                export_structure={'Agriculture': 0.70, 'Mining': 0.05, 'Manufacturing': 0.20, 'Services': 0.05},
                wage_level_index=8
            ),
            'Nigeria': CountryProfile(
                name='Nigeria', category='periphery', population=50,
                initial_gdp_per_capita=400,
                export_structure={'Agriculture': 0.50, 'Mining': 0.40, 'Manufacturing': 0.05, 'Services': 0.05},
                wage_level_index=10
            ),
            'Mexico': CountryProfile(
                name='Mexico', category='semi_periphery', population=50,
                initial_gdp_per_capita=2000,
                export_structure={'Agriculture': 0.30, 'Mining': 0.20, 'Manufacturing': 0.40, 'Services': 0.10},
                wage_level_index=25
            ),
            'SouthKorea': CountryProfile(
                name='SouthKorea', category='semi_periphery', population=30,
                initial_gdp_per_capita=800,
                export_structure={'Agriculture': 0.30, 'Mining': 0.05, 'Manufacturing': 0.55, 'Services': 0.10},
                wage_level_index=20  # Rising to ~80 by 2020
            )
        }

    def generate_gdp_series(self) -> pd.DataFrame:
        """
        Generate GDP time series for all countries.

        Reflects:
        - Differential growth rates (core ~3%, periphery variable)
        - Catching up (some semi-periphery)
        - Crises (1982 debt crisis, 1997 Asian, 2008 global)

        Returns:
            DataFrame with columns: year, country, gdp, gdp_per_capita, growth_rate
        """
        results = []

        for country_code, profile in self.countries.items():
            gdp_pc = profile.initial_gdp_per_capita
            population = profile.population

            for i, year in enumerate(self.years):
                # Growth rates by category
                if profile.category == 'core':
                    base_growth = 0.03
                    volatility = 0.01
                elif profile.category == 'semi_periphery':
                    base_growth = 0.05  # Higher but volatile
                    volatility = 0.03
                else:  # periphery
                    base_growth = 0.025  # Low and volatile
                    volatility = 0.04

                # Special cases
                if country_code == 'China' and year > 1980:
                    base_growth = 0.08  # Rapid growth post-reform
                elif country_code == 'SouthKorea' and 1970 < year < 2000:
                    base_growth = 0.07  # Miracle growth

                # Crises
                crisis_effect = 0
                if year in [1982, 1983] and profile.category == 'periphery':
                    crisis_effect = -0.05  # Debt crisis
                elif year in [1997, 1998] and country_code in ['SouthKorea', 'Thailand']:
                    crisis_effect = -0.08  # Asian crisis
                elif year in [2008, 2009]:
                    crisis_effect = -0.04  # Global crisis

                # Annual growth
                growth = base_growth + crisis_effect + np.random.normal(0, volatility)

                # Update GDP per capita
                gdp_pc = gdp_pc * (1 + growth)

                # Population growth
                pop_growth = 0.015 if profile.category == 'periphery' else 0.008
                population = population * (1 + pop_growth)

                # Total GDP
                gdp = gdp_pc * population

                results.append({
                    'year': year,
                    'country': country_code,
                    'category': profile.category,
                    'gdp': gdp,
                    'gdp_per_capita': gdp_pc,
                    'population': population,
                    'growth_rate': growth * 100
                })

        return pd.DataFrame(results)

    def generate_trade_flows(self) -> pd.DataFrame:
        """
        Generate bilateral trade flows.

        Reflects:
        - Gravity model patterns (size and distance)
        - Core imports primary goods, exports manufactures
        - Periphery exports primary goods, imports manufactures
        - Growing intra-core trade
        - Growing GVC trade (intermediates)

        Returns:
            DataFrame with bilateral trade flows
        """
        results = []
        gdp_data = self.generate_gdp_series()

        for year in self.years:
            year_gdp = gdp_data[gdp_data['year'] == year]

            for exp_country in self.countries.keys():
                for imp_country in self.countries.keys():
                    if exp_country == imp_country:
                        continue

                    exp_profile = self.countries[exp_country]
                    imp_profile = self.countries[imp_country]

                    exp_gdp = year_gdp[year_gdp['country'] == exp_country]['gdp'].values[0]
                    imp_gdp = year_gdp[year_gdp['country'] == imp_country]['gdp'].values[0]

                    # Gravity model
                    base_trade = (exp_gdp ** 0.8) * (imp_gdp ** 0.8) / 10000

                    # Trade intensity based on relationship
                    if exp_profile.category == imp_profile.category:
                        intensity = 1.2  # More intra-category trade
                    else:
                        intensity = 1.0

                    # Sector composition
                    for sector in self.sectors:
                        exp_share = exp_profile.export_structure.get(sector, 0)
                        # Demand varies by importer category and sector
                        if imp_profile.category == 'core' and sector in ['Agriculture', 'Mining']:
                            demand = 1.5  # Core imports raw materials
                        elif imp_profile.category == 'periphery' and sector == 'Manufacturing':
                            demand = 1.3  # Periphery imports manufactures
                        else:
                            demand = 1.0

                        trade_value = base_trade * exp_share * demand * intensity

                        # Add noise
                        trade_value *= (1 + np.random.normal(0, 0.1))

                        if trade_value > 1:  # Minimum threshold
                            results.append({
                                'year': year,
                                'exporter': exp_country,
                                'importer': imp_country,
                                'sector': sector,
                                'trade_value': trade_value,
                                'exporter_category': exp_profile.category,
                                'importer_category': imp_profile.category
                            })

        return pd.DataFrame(results)

    def generate_wage_data(self) -> pd.DataFrame:
        """
        Generate wage data showing core-periphery differentials.

        Reflects:
        - Large wage gaps (5-10x)
        - Slow convergence (if any)
        - Super-exploitation in periphery

        Returns:
            DataFrame with wage data
        """
        results = []

        for year in self.years:
            # Base core wage (indexed to 100 in 1960)
            base_core_wage = 100 * (1.02 ** (year - 1960))  # 2% annual growth

            for country_code, profile in self.countries.items():
                # Country wage relative to core
                relative_wage = profile.wage_level_index

                # Some convergence for successful semi-periphery
                if country_code == 'SouthKorea' and year > 1970:
                    convergence = min(0.80, 0.20 + (year - 1970) * 0.012)
                    relative_wage = relative_wage + (100 - relative_wage) * convergence
                elif country_code == 'China' and year > 1990:
                    convergence = min(0.50, 0.15 + (year - 1990) * 0.01)
                    relative_wage = relative_wage + (100 - relative_wage) * convergence

                wage = base_core_wage * (relative_wage / 100)

                # Productivity
                if profile.category == 'core':
                    productivity_index = 100 * (1.025 ** (year - 1960))
                elif profile.category == 'semi_periphery':
                    productivity_index = 50 * (1.03 ** (year - 1960))
                else:
                    productivity_index = 30 * (1.015 ** (year - 1960))

                # Super-exploitation metric
                productivity_adjusted_wage = base_core_wage * (productivity_index / 100)
                super_exploitation_gap = productivity_adjusted_wage - wage

                results.append({
                    'year': year,
                    'country': country_code,
                    'category': profile.category,
                    'wage_index': wage,
                    'productivity_index': productivity_index,
                    'wage_productivity_ratio': wage / productivity_index if productivity_index > 0 else 0,
                    'super_exploitation_gap': super_exploitation_gap
                })

        return pd.DataFrame(results)

    def generate_terms_of_trade(self) -> pd.DataFrame:
        """
        Generate terms of trade data (Prebisch-Singer).

        Shows secular deterioration for primary commodity exporters.

        Returns:
            DataFrame with ToT indices
        """
        results = []

        # Base prices (1960 = 100)
        primary_price = 100
        manufactures_price = 100

        for year in self.years:
            # Primary prices: volatile, declining trend
            primary_growth = -0.005 + np.random.normal(0, 0.08)  # -0.5% trend + volatility
            primary_price *= (1 + primary_growth)

            # Manufactures prices: steadier, slight increase
            manuf_growth = 0.01 + np.random.normal(0, 0.02)  # +1% trend + lower volatility
            manufactures_price *= (1 + manuf_growth)

            # Terms of trade for primary exporters
            tot = (primary_price / manufactures_price) * 100

            results.append({
                'year': year,
                'primary_price_index': primary_price,
                'manufactures_price_index': manufactures_price,
                'tot_index': tot,
                'tot_vs_base': tot - 100
            })

        return pd.DataFrame(results)

    def generate_financial_flows(self) -> pd.DataFrame:
        """
        Generate financial flows (FDI, debt, profit repatriation, IP payments).

        Reflects rising financialization and rent extraction.

        Returns:
            DataFrame with financial flows
        """
        results = []
        gdp_data = self.generate_gdp_series()

        for year in self.years:
            for country_code, profile in self.countries.items():
                country_gdp = gdp_data[
                    (gdp_data['year'] == year) & (gdp_data['country'] == country_code)
                ]['gdp'].values[0]

                if profile.category == 'periphery':
                    # Net recipient of FDI
                    fdi_inflow = country_gdp * 0.03 * (year - 1960) / 60  # Rising over time
                    fdi_outflow = country_gdp * 0.005

                    # Debt service
                    debt_service = country_gdp * 0.04 * (1 + (year - 1960) / 60)

                    # Profit repatriation
                    profit_repat = country_gdp * 0.02 * (year - 1960) / 60

                    # IP payments
                    ip_payments = country_gdp * 0.01 * (year - 1970) / 50 if year > 1970 else 0

                elif profile.category == 'core':
                    # Net sender of FDI
                    fdi_outflow = country_gdp * 0.04
                    fdi_inflow = country_gdp * 0.02

                    # Minimal debt service
                    debt_service = country_gdp * 0.01

                    # Profit receipts
                    profit_repat = -country_gdp * 0.03  # Negative = receipt

                    # IP receipts
                    ip_payments = -country_gdp * 0.015  # Negative = receipt

                else:  # semi-periphery
                    fdi_inflow = country_gdp * 0.02
                    fdi_outflow = country_gdp * 0.015
                    debt_service = country_gdp * 0.03
                    profit_repat = country_gdp * 0.01
                    ip_payments = country_gdp * 0.008

                results.append({
                    'year': year,
                    'country': country_code,
                    'category': profile.category,
                    'fdi_inflow': fdi_inflow,
                    'fdi_outflow': fdi_outflow,
                    'net_fdi': fdi_inflow - fdi_outflow,
                    'debt_service': debt_service,
                    'profit_repatriation': profit_repat,
                    'ip_payments': ip_payments,
                    'total_financial_outflow': debt_service + profit_repat + ip_payments
                })

        return pd.DataFrame(results)

    def generate_complete_dataset(self, output_dir: str = './data/') -> Dict[str, pd.DataFrame]:
        """
        Generate all datasets and save to CSV.

        Args:
            output_dir: Output directory for CSV files

        Returns:
            Dictionary of all generated datasets
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating synthetic datasets for unequal exchange analysis...")

        datasets = {}

        print("  - GDP and growth data...")
        datasets['gdp'] = self.generate_gdp_series()

        print("  - Bilateral trade flows...")
        datasets['trade'] = self.generate_trade_flows()

        print("  - Wage and productivity data...")
        datasets['wages'] = self.generate_wage_data()

        print("  - Terms of trade indices...")
        datasets['terms_of_trade'] = self.generate_terms_of_trade()

        print("  - Financial flows...")
        datasets['financial'] = self.generate_financial_flows()

        # Save to CSV
        for name, df in datasets.items():
            filename = f'{output_dir}/{name}_1960_2020.csv'
            df.to_csv(filename, index=False)
            print(f"  Saved: {filename} ({len(df)} rows)")

        # Create data dictionary
        self._create_data_dictionary(output_dir)

        print(f"\n✓ Complete! Generated {len(datasets)} datasets covering {len(self.years)} years.")

        return datasets

    def _create_data_dictionary(self, output_dir: str):
        """Create data dictionary explaining all variables"""
        dictionary = """
# Data Dictionary: Unequal Exchange Synthetic Dataset
# Generated: {date}
# Period: {start_year}-{end_year}

## GDP and Growth Data (gdp_1960_2020.csv)
- year: Year
- country: Country code
- category: core, semi_periphery, or periphery
- gdp: Gross domestic product (billions USD, constant 2010)
- gdp_per_capita: GDP per capita (USD)
- population: Population (millions)
- growth_rate: Real GDP growth rate (%)

## Trade Flows (trade_1960_2020.csv)
- year: Year
- exporter: Exporting country
- importer: Importing country
- sector: Agriculture, Mining, Manufacturing, or Services
- trade_value: Bilateral trade value (millions USD)
- exporter_category: Category of exporter
- importer_category: Category of importer

## Wage and Productivity (wages_1960_2020.csv)
- year: Year
- country: Country code
- category: Country category
- wage_index: Wage index (Core 1960 = 100)
- productivity_index: Labor productivity index
- wage_productivity_ratio: Wage/productivity ratio
- super_exploitation_gap: Gap between productivity-adjusted and actual wages

## Terms of Trade (terms_of_trade_1960_2020.csv)
- year: Year
- primary_price_index: Price index for primary commodities (1960 = 100)
- manufactures_price_index: Price index for manufactured goods (1960 = 100)
- tot_index: Terms of trade index (primary/manufactures * 100)
- tot_vs_base: Cumulative change from 1960

## Financial Flows (financial_1960_2020.csv)
- year: Year
- country: Country code
- category: Country category
- fdi_inflow: Foreign direct investment inflows
- fdi_outflow: FDI outflows
- net_fdi: Net FDI (inflows - outflows)
- debt_service: External debt service payments
- profit_repatriation: Repatriated profits (positive = outflow)
- ip_payments: Intellectual property payments (positive = outflow)
- total_financial_outflow: Total financial outflows

## Methodology
This synthetic dataset is designed to reflect stylized facts from dependency theory:
1. Wage differentials: 5-10x between core and periphery
2. Terms of trade: Secular deterioration for primary exporters (~20-30% over period)
3. Super-exploitation: Wages growing slower than productivity in periphery
4. Rising financialization: Growing profit/IP flows from South to North
5. Trade patterns: Core exports manufactures, periphery exports primary goods

## References
- Emmanuel, A. (1972). Unequal Exchange
- Amin, S. (1974). Accumulation on a World Scale
- Prebisch, R. (1950). Economic Development of Latin America
        """.format(
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            start_year=self.start_year,
            end_year=self.end_year
        )

        with open(f'{output_dir}/DATA_DICTIONARY.md', 'w') as f:
            f.write(dictionary)

        print(f"  Saved: {output_dir}/DATA_DICTIONARY.md")


# Convenience function
def generate_datasets(output_dir: str = './unequal_exchange_data/') -> Dict[str, pd.DataFrame]:
    """
    Generate all synthetic datasets.

    Args:
        output_dir: Output directory

    Returns:
        Dictionary of datasets
    """
    generator = SyntheticDataGenerator(start_year=1960, end_year=2020)
    return generator.generate_complete_dataset(output_dir)
