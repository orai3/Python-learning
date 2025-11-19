"""
Multi-Country Input-Output Framework

Implements inter-country input-output analysis for studying:
- Global value chains
- Embodied labor in trade
- Value added decomposition
- Vertical specialization
- Production fragmentation

Based on:
- Leontief (1936): Input-Output Economics
- Miller & Blair (2009): Input-Output Analysis: Foundations and Extensions
- Timmer et al. (2015): World Input-Output Database
- Los, Timmer & de Vries (2016): Tracing Value-Added and Double Counting in GVCs

Mathematical Framework:
For n countries with m sectors each:
- x = (I - A)^(-1) * f  [Leontief model]
- A = multi-country technical coefficients matrix
- x = gross output vector
- f = final demand vector

Value added decomposition:
- VA = v̂ * (I - A)^(-1) * f
where v̂ = diagonal matrix of value-added coefficients
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class IOTableMetadata:
    """Metadata for input-output table"""
    countries: List[str]
    sectors: List[str]
    year: int
    currency: str = "USD"
    units: str = "millions"


class MultiCountryIOTable:
    """
    Multi-country input-output table for global value chain analysis.

    Handles construction, manipulation, and analysis of inter-country
    input-output tables.
    """

    def __init__(self, metadata: IOTableMetadata):
        """
        Initialize multi-country IO table.

        Args:
            metadata: Table metadata (countries, sectors, year, etc.)
        """
        self.metadata = metadata
        self.n_countries = len(metadata.countries)
        self.n_sectors = len(metadata.sectors)
        self.n_total = self.n_countries * self.n_sectors

        # Create index labels
        self.row_labels = pd.MultiIndex.from_product(
            [metadata.countries, metadata.sectors],
            names=['country', 'sector']
        )
        self.col_labels = self.row_labels.copy()

        # Initialize matrices
        self.Z = None  # Intermediate use matrix
        self.F = None  # Final demand matrix
        self.x = None  # Gross output vector
        self.VA = None  # Value added vector
        self.E = None  # Exports matrix
        self.M = None  # Imports matrix
        self.L = None  # Labor input vector

        # Computed matrices
        self.A = None  # Technical coefficients
        self.B = None  # Leontief inverse
        self.v = None  # Value added coefficients

    def set_intermediate_use(self, Z: Union[np.ndarray, pd.DataFrame]):
        """
        Set intermediate use matrix Z.

        Z[i,j] = intermediate use of product i in production of j

        Args:
            Z: Intermediate use matrix (n_total x n_total)
        """
        if isinstance(Z, pd.DataFrame):
            self.Z = Z
        else:
            self.Z = pd.DataFrame(
                Z,
                index=self.row_labels,
                columns=self.col_labels
            )

    def set_final_demand(self, F: Union[np.ndarray, pd.DataFrame]):
        """
        Set final demand matrix F.

        F[i,c] = final demand in country c for product i

        Args:
            F: Final demand matrix (n_total x n_countries)
        """
        if isinstance(F, pd.DataFrame):
            self.F = F
        else:
            col_labels = self.metadata.countries
            self.F = pd.DataFrame(
                F,
                index=self.row_labels,
                columns=col_labels
            )

    def set_value_added(self, VA: Union[np.ndarray, pd.Series]):
        """
        Set value added vector.

        VA[i] = value added in production of sector i

        Args:
            VA: Value added vector (n_total,)
        """
        if isinstance(VA, pd.Series):
            self.VA = VA
        else:
            self.VA = pd.Series(VA, index=self.row_labels)

    def set_labor_input(self, L: Union[np.ndarray, pd.Series]):
        """
        Set labor input vector (in hours or workers).

        L[i] = labor employed in sector i

        Args:
            L: Labor input vector (n_total,)
        """
        if isinstance(L, pd.Series):
            self.L = L
        else:
            self.L = pd.Series(L, index=self.row_labels)

    def calculate_gross_output(self):
        """
        Calculate gross output as sum of intermediate use and final demand.

        x = Z * 1 + F * 1
        where 1 is a vector of ones
        """
        if self.Z is None or self.F is None:
            raise ValueError("Must set Z and F before calculating gross output")

        intermediate_total = self.Z.sum(axis=1)
        final_total = self.F.sum(axis=1)
        self.x = intermediate_total + final_total

        return self.x

    def calculate_technical_coefficients(self):
        """
        Calculate technical coefficients matrix A.

        A[i,j] = Z[i,j] / x[j]
        = input of i per unit output of j
        """
        if self.Z is None or self.x is None:
            raise ValueError("Must set Z and calculate x first")

        # Avoid division by zero
        x_safe = self.x.replace(0, np.nan)

        self.A = self.Z.div(x_safe, axis=1)
        self.A = self.A.fillna(0)

        return self.A

    def calculate_leontief_inverse(self) -> pd.DataFrame:
        """
        Calculate Leontief inverse B = (I - A)^(-1).

        B[i,j] = total output of i needed per unit final demand of j
        (direct + indirect requirements)

        Returns:
            Leontief inverse matrix
        """
        if self.A is None:
            self.calculate_technical_coefficients()

        I = np.eye(self.n_total)
        A_array = self.A.values

        try:
            B_array = np.linalg.inv(I - A_array)
            self.B = pd.DataFrame(
                B_array,
                index=self.row_labels,
                columns=self.col_labels
            )
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in Leontief inverse. Check data consistency.")
            self.B = None

        return self.B

    def calculate_value_added_coefficients(self):
        """
        Calculate value added coefficients v[i] = VA[i] / x[i].

        v[i] = value added per unit gross output in sector i
        """
        if self.VA is None or self.x is None:
            raise ValueError("Must set VA and calculate x first")

        x_safe = self.x.replace(0, np.nan)
        self.v = self.VA / x_safe
        self.v = self.v.fillna(0)

        return self.v

    def calculate_labor_coefficients(self) -> pd.Series:
        """
        Calculate labor coefficients l[i] = L[i] / x[i].

        l[i] = labor hours per unit gross output in sector i

        Returns:
            Labor coefficients
        """
        if self.L is None or self.x is None:
            raise ValueError("Must set L and calculate x first")

        x_safe = self.x.replace(0, np.nan)
        l = self.L / x_safe
        l = l.fillna(0)

        return l

    def decompose_value_added(self) -> pd.DataFrame:
        """
        Decompose value added embodied in final demand.

        VA_embedded = v̂ * B * f

        Returns:
            DataFrame showing value added by source country/sector
            embodied in each country's final demand
        """
        if self.B is None:
            self.calculate_leontief_inverse()

        if self.v is None:
            self.calculate_value_added_coefficients()

        # Create diagonal matrix of v
        v_diag = np.diag(self.v.values)

        # Calculate v̂ * B
        vB = pd.DataFrame(
            v_diag @ self.B.values,
            index=self.row_labels,
            columns=self.col_labels
        )

        # Multiply by final demand for each country
        results = {}
        for country in self.metadata.countries:
            f_country = self.F[country].values
            va_embedded = vB.values @ f_country
            results[country] = va_embedded

        va_decomp = pd.DataFrame(
            results,
            index=self.row_labels
        )

        return va_decomp

    def calculate_embodied_labor(self) -> pd.DataFrame:
        """
        Calculate labor embodied in each country's final demand.

        L_embedded = l̂ * B * f

        Returns:
            DataFrame showing labor hours by source country/sector
            embodied in each country's final demand
        """
        if self.B is None:
            self.calculate_leontief_inverse()

        l = self.calculate_labor_coefficients()

        # Create diagonal matrix of l
        l_diag = np.diag(l.values)

        # Calculate l̂ * B
        lB = pd.DataFrame(
            l_diag @ self.B.values,
            index=self.row_labels,
            columns=self.col_labels
        )

        # Multiply by final demand for each country
        results = {}
        for country in self.metadata.countries:
            f_country = self.F[country].values
            labor_embedded = lB.values @ f_country
            results[country] = labor_embedded

        labor_decomp = pd.DataFrame(
            results,
            index=self.row_labels
        )

        return labor_decomp

    def calculate_vertical_specialization(self) -> pd.DataFrame:
        """
        Calculate vertical specialization (imported inputs in exports).

        VS = (imported intermediate inputs) / (gross exports)

        High VS indicates high participation in global value chains.

        Returns:
            DataFrame with VS metrics by country
        """
        results = []

        for country in self.metadata.countries:
            # Get this country's sectors
            country_mask = self.row_labels.get_level_values('country') == country

            # Calculate exports (sales to other countries)
            exports = 0
            for other_country in self.metadata.countries:
                if other_country != country:
                    # Intermediate exports
                    intermediate_exports = self.Z.loc[country_mask, :].filter(
                        like=other_country, axis=1
                    ).sum().sum()

                    # Final exports
                    if other_country in self.F.columns:
                        final_exports = self.F.loc[country_mask, other_country].sum()
                    else:
                        final_exports = 0

                    exports += intermediate_exports + final_exports

            # Calculate imported intermediates used by this country
            imported_intermediates = 0
            for other_country in self.metadata.countries:
                if other_country != country:
                    # Intermediates from other_country used in country's production
                    imported_int = self.Z.loc[
                        self.row_labels.get_level_values('country') == other_country,
                        country_mask
                    ].sum().sum()

                    imported_intermediates += imported_int

            # Vertical specialization ratio
            vs_ratio = (imported_intermediates / exports) if exports > 0 else 0

            results.append({
                'country': country,
                'exports': exports,
                'imported_intermediates': imported_intermediates,
                'vertical_specialization': vs_ratio
            })

        return pd.DataFrame(results)

    def calculate_gvc_participation(self) -> pd.DataFrame:
        """
        Calculate GVC participation indices (forward and backward linkages).

        Backward participation: foreign VA in exports
        Forward participation: domestic VA in other countries' exports

        Returns:
            DataFrame with participation metrics
        """
        # Decompose value added
        va_decomp = self.decompose_value_added()

        results = []

        for country in self.metadata.countries:
            country_mask = self.row_labels.get_level_values('country') == country

            # Calculate this country's exports
            exports = 0
            for other_country in self.metadata.countries:
                if other_country != country:
                    if other_country in self.F.columns:
                        exports += self.F.loc[country_mask, other_country].sum()

            # Backward participation: foreign VA in this country's exports
            # (VA from other countries embodied in this country's exports to third countries)
            foreign_va_in_exports = 0
            for other_country in self.metadata.countries:
                if other_country != country:
                    other_mask = self.row_labels.get_level_values('country') == other_country
                    # Foreign VA in this country's final demand
                    foreign_va = va_decomp.loc[other_mask, country].sum()
                    foreign_va_in_exports += foreign_va

            # Forward participation: domestic VA in other countries' exports
            domestic_va_in_foreign_exports = 0
            for other_country in self.metadata.countries:
                if other_country != country:
                    # This country's VA in other country's final demand
                    domestic_va = va_decomp.loc[country_mask, other_country].sum()
                    domestic_va_in_foreign_exports += domestic_va

            results.append({
                'country': country,
                'exports': exports,
                'backward_participation': foreign_va_in_exports / exports if exports > 0 else 0,
                'forward_participation': domestic_va_in_foreign_exports / exports if exports > 0 else 0,
                'total_gvc_participation': (foreign_va_in_exports + domestic_va_in_foreign_exports) / exports if exports > 0 else 0
            })

        return pd.DataFrame(results)

    def extract_bilateral_flows(self, country_a: str, country_b: str) -> Dict[str, pd.DataFrame]:
        """
        Extract bilateral flows between two countries.

        Args:
            country_a: First country
            country_b: Second country

        Returns:
            Dictionary with:
            - intermediates_a_to_b: Intermediate flows from A to B
            - intermediates_b_to_a: Intermediate flows from B to A
            - final_a_to_b: Final demand in B for A's products
            - final_b_to_a: Final demand in A for B's products
        """
        mask_a = self.row_labels.get_level_values('country') == country_a
        mask_b = self.row_labels.get_level_values('country') == country_b

        # Intermediate flows
        intermediates_a_to_b = self.Z.loc[mask_a, mask_b]
        intermediates_b_to_a = self.Z.loc[mask_b, mask_a]

        # Final demand flows
        final_a_to_b = self.F.loc[mask_a, country_b] if country_b in self.F.columns else None
        final_b_to_a = self.F.loc[mask_b, country_a] if country_a in self.F.columns else None

        return {
            'intermediates_a_to_b': intermediates_a_to_b,
            'intermediates_b_to_a': intermediates_b_to_a,
            'final_a_to_b': final_a_to_b,
            'final_b_to_a': final_b_to_a,
            'total_a_to_b': intermediates_a_to_b.sum().sum() + (final_a_to_b.sum() if final_a_to_b is not None else 0),
            'total_b_to_a': intermediates_b_to_a.sum().sum() + (final_b_to_a.sum() if final_b_to_a is not None else 0)
        }

    def calculate_value_appropriation(self) -> pd.DataFrame:
        """
        Calculate value appropriation patterns (who captures value in GVCs).

        Shows how much value created in country A is captured (as final demand)
        in country B.

        Returns:
            DataFrame showing value creation vs value capture
        """
        va_decomp = self.decompose_value_added()

        results = []

        for creating_country in self.metadata.countries:
            creating_mask = self.row_labels.get_level_values('country') == creating_country

            # Value created in this country
            value_created = self.VA[creating_mask].sum()

            # Value captured in each country's final demand
            for capturing_country in self.metadata.countries:
                value_captured = va_decomp.loc[creating_mask, capturing_country].sum()

                results.append({
                    'creating_country': creating_country,
                    'capturing_country': capturing_country,
                    'value_created': value_created,
                    'value_captured': value_captured,
                    'capture_share': value_captured / value_created if value_created > 0 else 0
                })

        df = pd.DataFrame(results)

        # Pivot for easier reading
        pivot = df.pivot(
            index='creating_country',
            columns='capturing_country',
            values='capture_share'
        )

        return pivot

    def analyze_dependency_structure(self) -> pd.DataFrame:
        """
        Analyze dependency relationships in production network.

        High dependence = country relies heavily on inputs from specific partners.

        Returns:
            DataFrame with dependency metrics
        """
        results = []

        for country in self.metadata.countries:
            country_mask = self.row_labels.get_level_values('country') == country

            # Total intermediates used by this country
            total_intermediates = self.Z.loc[:, country_mask].sum().sum()

            # Break down by source country
            for source in self.metadata.countries:
                source_mask = self.row_labels.get_level_values('country') == source

                # Intermediates from source to country
                intermediates_from_source = self.Z.loc[source_mask, country_mask].sum().sum()

                # Dependency ratio
                dependency = intermediates_from_source / total_intermediates if total_intermediates > 0 else 0

                results.append({
                    'dependent_country': country,
                    'source_country': source,
                    'intermediate_imports': intermediates_from_source,
                    'total_intermediates': total_intermediates,
                    'dependency_ratio': dependency
                })

        return pd.DataFrame(results)

    def get_summary_statistics(self) -> Dict[str, any]:
        """
        Generate summary statistics for the IO table.

        Returns:
            Dictionary with key statistics
        """
        if self.x is None:
            self.calculate_gross_output()

        stats = {
            'total_gross_output': self.x.sum(),
            'total_value_added': self.VA.sum() if self.VA is not None else None,
            'total_intermediate_use': self.Z.sum().sum() if self.Z is not None else None,
            'total_final_demand': self.F.sum().sum() if self.F is not None else None,
            'total_labor_input': self.L.sum() if self.L is not None else None,
            'n_countries': self.n_countries,
            'n_sectors': self.n_sectors,
            'year': self.metadata.year
        }

        # Country-level statistics
        country_stats = {}
        for country in self.metadata.countries:
            mask = self.row_labels.get_level_values('country') == country
            country_stats[country] = {
                'gross_output': self.x[mask].sum(),
                'value_added': self.VA[mask].sum() if self.VA is not None else None,
                'labor_input': self.L[mask].sum() if self.L is not None else None
            }

        stats['by_country'] = country_stats

        return stats
