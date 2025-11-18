"""
Sectoral Analysis Exercises: Shift-Share & Structural Decomposition
Heterodox Economics Focus

Exercises cover sectoral dynamics with emphasis on structural change,
deindustrialization, and comparative political economy perspectives.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXERCISE 1: Shift-Share Analysis - Decomposing Employment Change
# ============================================================================
# THEORY: Shift-share decomposes regional/sectoral change into national growth,
# industrial mix, and competitive effects. Useful for understanding
# deindustrialization and structural transformation.
# Reference: Dunn (1960), Esteban (2000)
# ============================================================================

def exercise_1_shift_share_analysis():
    """
    Problem: Analyze deindustrialization using shift-share decomposition.
    Compare manufacturing region to national trends.

    Reference: Structuralist development economics, Cambridge growth theory
    """
    print("=" * 80)
    print("EXERCISE 1: Shift-Share Analysis - Deindustrialization")
    print("=" * 80)

    # Generate synthetic employment data
    # National data (entire economy)
    sectors = ['Manufacturing', 'Services', 'Construction', 'Agriculture', 'Mining']

    # Time period: 1980 to 2020
    national_1980 = np.array([5000, 8000, 1500, 2000, 1000])  # Employment in thousands
    national_2020 = np.array([4500, 18000, 2200, 1200, 800])  # Shift to services

    # Regional data (e.g., "Rust Belt" manufacturing region)
    regional_1980 = np.array([1500, 2000, 400, 300, 200])  # Heavy manufacturing
    regional_2020 = np.array([600, 4500, 500, 200, 100])   # Severe deindustrialization

    # Create DataFrames
    df_national = pd.DataFrame({
        'Sector': sectors,
        '1980': national_1980,
        '2020': national_2020
    })

    df_regional = pd.DataFrame({
        'Sector': sectors,
        '1980': regional_1980,
        '2020': regional_2020
    })

    # SOLUTION: Implement Shift-Share Decomposition
    def shift_share_decomposition(regional_base, regional_end, national_base, national_end):
        """
        Shift-share decomposition of employment change

        Components:
        1. National growth effect: What if region grew at national rate?
        2. Industry mix effect: Is region specialized in growing/declining sectors?
        3. Competitive effect: Is region gaining/losing share within sectors?

        Args:
            regional_base: Regional employment by sector (base year)
            regional_end: Regional employment by sector (end year)
            national_base: National employment by sector (base year)
            national_end: National employment by sector (end year)

        Returns:
            DataFrame with decomposition results
        """
        # Total growth rates
        national_total_growth = (np.sum(national_end) - np.sum(national_base)) / np.sum(national_base)
        regional_total_growth = (np.sum(regional_end) - np.sum(regional_base)) / np.sum(regional_base)

        # Sectoral growth rates (national)
        sectoral_growth_rates = (national_end - national_base) / national_base

        # Regional share of each sector (base year)
        regional_shares_base = regional_base / national_base

        results = []

        for i in range(len(regional_base)):
            # Actual change
            actual_change = regional_end[i] - regional_base[i]

            # 1. National growth effect
            # What if regional sector grew at overall national rate?
            national_effect = regional_base[i] * national_total_growth

            # 2. Industry mix effect
            # Difference between sector-specific and overall national growth
            mix_effect = regional_base[i] * (sectoral_growth_rates[i] - national_total_growth)

            # 3. Competitive effect (regional shift)
            # Change in regional share of national sector employment
            expected_regional = regional_shares_base[i] * national_end[i]
            competitive_effect = regional_end[i] - expected_regional

            results.append({
                'national_effect': national_effect,
                'mix_effect': mix_effect,
                'competitive_effect': competitive_effect,
                'total_effect': actual_change,
                'check_sum': national_effect + mix_effect + competitive_effect
            })

        return pd.DataFrame(results)

    # Perform decomposition
    decomp = shift_share_decomposition(
        regional_1980, regional_2020,
        national_1980, national_2020
    )
    decomp['Sector'] = sectors

    # Add actual changes for reference
    decomp['actual_change'] = regional_2020 - regional_1980
    decomp['pct_change'] = ((regional_2020 - regional_1980) / regional_1980) * 100

    print("\nSHIFT-SHARE DECOMPOSITION RESULTS (Employment change in thousands):")
    print("=" * 80)
    print(decomp[['Sector', 'actual_change', 'national_effect', 'mix_effect',
                   'competitive_effect', 'pct_change']].round(1))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Shift-Share Analysis: Regional Deindustrialization (1980-2020)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Employment levels comparison
    x = np.arange(len(sectors))
    width = 0.35

    axes[0, 0].bar(x - width/2, regional_1980, width, label='1980', alpha=0.8, color='#1976D2')
    axes[0, 0].bar(x + width/2, regional_2020, width, label='2020', alpha=0.8, color='#D32F2F')
    axes[0, 0].set_xlabel('Sector')
    axes[0, 0].set_ylabel('Employment (thousands)')
    axes[0, 0].set_title('Regional Employment by Sector')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(sectors, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot 2: National comparison
    axes[0, 1].bar(x - width/2, national_1980/1000, width, label='1980', alpha=0.8, color='#1976D2')
    axes[0, 1].bar(x + width/2, national_2020/1000, width, label='2020', alpha=0.8, color='#D32F2F')
    axes[0, 1].set_xlabel('Sector')
    axes[0, 1].set_ylabel('Employment (millions)')
    axes[0, 1].set_title('National Employment by Sector')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(sectors, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Decomposition components (stacked bar)
    components = ['national_effect', 'mix_effect', 'competitive_effect']
    colors_comp = ['#4CAF50', '#FF9800', '#9C27B0']

    # Prepare data for stacking (separate positive and negative)
    bottom_pos = np.zeros(len(sectors))
    bottom_neg = np.zeros(len(sectors))

    for comp, color in zip(components, colors_comp):
        values = decomp[comp].values
        pos_values = np.maximum(values, 0)
        neg_values = np.minimum(values, 0)

        axes[1, 0].bar(x, pos_values, bottom=bottom_pos, label=comp.replace('_', ' ').title(),
                      alpha=0.8, color=color)
        axes[1, 0].bar(x, neg_values, bottom=bottom_neg,
                      alpha=0.8, color=color)

        bottom_pos += pos_values
        bottom_neg += neg_values

    # Add actual change points
    axes[1, 0].scatter(x, decomp['actual_change'], color='red', s=100, zorder=5,
                      marker='D', label='Actual Change', edgecolors='black', linewidths=2)

    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Sector')
    axes[1, 0].set_ylabel('Employment Change (thousands)')
    axes[1, 0].set_title('Shift-Share Decomposition by Component')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sectors, rotation=45, ha='right')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Component contributions (heatmap-style)
    comp_matrix = decomp[components].values.T
    im = axes[1, 1].imshow(comp_matrix, cmap='RdYlGn', aspect='auto',
                           vmin=-1000, vmax=1000)
    axes[1, 1].set_xticks(np.arange(len(sectors)))
    axes[1, 1].set_yticks(np.arange(len(components)))
    axes[1, 1].set_xticklabels(sectors, rotation=45, ha='right')
    axes[1, 1].set_yticklabels([c.replace('_', ' ').title() for c in components])
    axes[1, 1].set_title('Component Contributions (thousands)')

    # Add values
    for i in range(len(components)):
        for j in range(len(sectors)):
            text = axes[1, 1].text(j, i, f'{comp_matrix[i, j]:.0f}',
                                  ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=axes[1, 1], label='Employment Change')

    plt.tight_layout()
    plt.savefig('phase3-exercises/sectoral/ex1_shift_share.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: phase3-exercises/sectoral/ex1_shift_share.png")

    # ECONOMIC INTERPRETATION
    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION:")
    print("=" * 80)
    print("\n1. Shift-Share Components Explained:")
    print("   a) NATIONAL EFFECT: Regional employment change if region grew at national rate")
    print("      - Captures overall economic growth/decline")
    print("      - Usually positive in growing economies")
    print("   b) INDUSTRY MIX EFFECT: Impact of regional sectoral specialization")
    print("      - Positive if region specialized in fast-growing sectors")
    print("      - Negative if specialized in declining sectors")
    print("   c) COMPETITIVE EFFECT: Regional competitiveness within each sector")
    print("      - Gaining market share → positive")
    print("      - Losing market share → negative")

    print("\n2. Manufacturing Sector Analysis:")
    mfg_idx = 0  # Manufacturing is first sector
    print(f"   Total manufacturing job loss: {decomp['actual_change'].iloc[mfg_idx]:.0f}k "
          f"({decomp['pct_change'].iloc[mfg_idx]:.1f}%)")
    print(f"   - National effect: {decomp['national_effect'].iloc[mfg_idx]:.0f}k")
    print(f"   - Industry mix effect: {decomp['mix_effect'].iloc[mfg_idx]:.0f}k")
    print(f"   - Competitive effect: {decomp['competitive_effect'].iloc[mfg_idx]:.0f}k")
    print("\n   Interpretation:")
    if decomp['mix_effect'].iloc[mfg_idx] < 0:
        print("   ✓ Industry mix NEGATIVE: Manufacturing declining nationally")
    if decomp['competitive_effect'].iloc[mfg_idx] < 0:
        print("   ✓ Competitive effect NEGATIVE: Region losing mfg jobs faster than national avg")
        print("     → Double whammy: Declining sector + regional uncompetitiveness")

    print("\n3. Services Sector Analysis:")
    svc_idx = 1
    print(f"   Total services job gain: {decomp['actual_change'].iloc[svc_idx]:.0f}k "
          f"({decomp['pct_change'].iloc[svc_idx]:.1f}%)")
    print(f"   - National effect: {decomp['national_effect'].iloc[svc_idx]:.0f}k")
    print(f"   - Industry mix effect: {decomp['mix_effect'].iloc[svc_idx]:.0f}k")
    print(f"   - Competitive effect: {decomp['competitive_effect'].iloc[svc_idx]:.0f}k")
    if decomp['competitive_effect'].iloc[svc_idx] > 0:
        print("   ✓ Region successfully transitioning to services")
    else:
        print("   ✗ Service growth below national average")

    print("\n4. Structural Transformation Patterns:")
    print("   This mirrors actual US/UK deindustrialization:")
    print("   - 1980s: Manufacturing employment collapse")
    print("   - Service sector expansion (but often lower wages)")
    print("   - Regional disparities widen (Rust Belt vs Sun Belt)")
    print("   - Competitive losses suggest:")
    print("     * Capital flight to lower-wage regions/countries")
    print("     * Technological obsolescence")
    print("     * Inadequate regional industrial policy")

    print("\n5. Heterodox Perspective:")
    print("   Post-Keynesian/Structuralist insights:")
    print("   - Deindustrialization NOT inevitable 'market outcome'")
    print("   - Policy choices matter: Trade policy, industrial policy, labor protection")
    print("   - 'Comparative advantage' can be created, not just endowed")
    print("   - Service jobs often can't replace manufacturing wages/stability")
    print("   - Regional inequality reflects political economy, not just 'efficiency'")

    print("\n6. Policy Implications:")
    print("   Mix effect negative → Need national industrial policy")
    print("   Competitive effect negative → Need regional development strategy")
    print("   - Possible interventions:")
    print("     * Targeted R&D support for manufacturing")
    print("     * Managed trade to protect strategic sectors")
    print("     * Public investment in infrastructure")
    print("     * Training/education programs")
    print("     * Regional development banks/finance")

    return decomp, df_national, df_regional


# ============================================================================
# EXERCISE 2: Structural Decomposition Analysis (SDA)
# ============================================================================
# THEORY: SDA decomposes changes in economic aggregates into technical change,
# demand composition, and scale effects using input-output framework.
# Reference: Structural economics, ecological economics
# ============================================================================

def exercise_2_structural_decomposition():
    """
    Problem: Decompose changes in CO2 emissions into scale, composition,
    and intensity effects. Classic application of SDA.

    Reference: Ang & Zhang (2000), Grossman & Krueger (1995) EKC
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Structural Decomposition - CO2 Emissions")
    print("=" * 80)

    # Simplified economy with 3 sectors
    sectors = ['Manufacturing', 'Services', 'Energy']

    # Base year (1990)
    gdp_1990 = 1000  # Total GDP in billions
    sector_shares_1990 = np.array([0.35, 0.50, 0.15])  # Share of GDP
    emission_intensity_1990 = np.array([0.8, 0.2, 2.5])  # tons CO2 per $ million GDP

    # End year (2020)
    gdp_2020 = 2000  # GDP doubled
    sector_shares_2020 = np.array([0.25, 0.65, 0.10])  # Shift to services
    emission_intensity_2020 = np.array([0.5, 0.15, 1.8])  # Technological improvement

    # SOLUTION: Implement Index Decomposition Analysis (IDA)
    def structural_decomposition_emissions(gdp_base, shares_base, intensity_base,
                                          gdp_end, shares_end, intensity_end):
        """
        Decompose emission changes into:
        1. Scale effect: GDP growth (holding everything else constant)
        2. Composition effect: Sectoral restructuring
        3. Intensity effect: Technological change in emissions per unit output

        Using Logarithmic Mean Divisia Index (LMDI) method

        Args:
            gdp_base, gdp_end: Total GDP
            shares_base, shares_end: Sectoral shares of GDP
            intensity_base, intensity_end: Emission intensity by sector

        Returns:
            Decomposition results
        """
        # Calculate emissions
        emissions_base = gdp_base * np.sum(shares_base * intensity_base)
        emissions_end = gdp_end * np.sum(shares_end * intensity_end)
        total_change = emissions_end - emissions_base

        # Sectoral emissions
        sectoral_emissions_base = gdp_base * shares_base * intensity_base
        sectoral_emissions_end = gdp_end * shares_end * intensity_end

        # LMDI weights (logarithmic mean)
        def lmdi_weight(val_base, val_end):
            if val_base == val_end:
                return val_base
            return (val_end - val_base) / (np.log(val_end) - np.log(val_base))

        weights = np.array([lmdi_weight(sectoral_emissions_base[i], sectoral_emissions_end[i])
                           for i in range(len(sectors))])

        # Decomposition components
        # 1. Scale effect (GDP growth)
        scale_effect = np.sum(weights * np.log(gdp_end / gdp_base))

        # 2. Composition effect (structural change)
        composition_effect = np.sum(weights * np.log(shares_end / shares_base))

        # 3. Intensity effect (technological change)
        intensity_effect = np.sum(weights * np.log(intensity_end / intensity_base))

        # Sectoral contributions to composition effect
        comp_contributions = weights * np.log(shares_end / shares_base)

        # Sectoral contributions to intensity effect
        inten_contributions = weights * np.log(intensity_end / intensity_base)

        return {
            'emissions_base': emissions_base,
            'emissions_end': emissions_end,
            'total_change': total_change,
            'scale_effect': scale_effect,
            'composition_effect': composition_effect,
            'intensity_effect': intensity_effect,
            'comp_contributions': comp_contributions,
            'inten_contributions': inten_contributions,
            'check_sum': scale_effect + composition_effect + intensity_effect
        }

    # Perform decomposition
    results = structural_decomposition_emissions(
        gdp_1990, sector_shares_1990, emission_intensity_1990,
        gdp_2020, sector_shares_2020, emission_intensity_2020
    )

    # Create summary DataFrame
    decomp_summary = pd.DataFrame({
        'Component': ['Scale Effect', 'Composition Effect', 'Intensity Effect', 'Total Change'],
        'Value': [results['scale_effect'], results['composition_effect'],
                  results['intensity_effect'], results['total_change']],
        'Percentage': [
            results['scale_effect'] / results['total_change'] * 100,
            results['composition_effect'] / results['total_change'] * 100,
            results['intensity_effect'] / results['total_change'] * 100,
            100
        ]
    })

    print("\nSTRUCTURAL DECOMPOSITION RESULTS:")
    print("=" * 80)
    print(f"Base year (1990) emissions: {results['emissions_base']:.1f} million tons CO2")
    print(f"End year (2020) emissions: {results['emissions_end']:.1f} million tons CO2")
    print(f"Total change: {results['total_change']:.1f} million tons CO2 "
          f"({results['total_change']/results['emissions_base']*100:.1f}%)")
    print(f"\nDecomposition check (should equal total): {results['check_sum']:.1f}")
    print("\n" + str(decomp_summary.round(2)))

    # Sectoral details
    sectoral_df = pd.DataFrame({
        'Sector': sectors,
        'Share_1990': sector_shares_1990,
        'Share_2020': sector_shares_2020,
        'Intensity_1990': emission_intensity_1990,
        'Intensity_2020': emission_intensity_2020,
        'Composition_Contribution': results['comp_contributions'],
        'Intensity_Contribution': results['inten_contributions']
    })

    print("\nSECTORAL DETAILS:")
    print("=" * 80)
    print(sectoral_df.round(3))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Structural Decomposition Analysis: CO2 Emissions (1990-2020)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Decomposition waterfall chart
    categories = ['1990\nBaseline', 'Scale\nEffect', 'Composition\nEffect',
                  'Intensity\nEffect', '2020\nActual']
    values = [results['emissions_base'], results['scale_effect'],
              results['composition_effect'], results['intensity_effect'], 0]

    cumulative = [results['emissions_base']]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(results['emissions_end'])

    colors_waterfall = ['#1976D2', '#4CAF50', '#FF9800', '#D32F2F', '#1976D2']

    for i in range(len(categories)):
        if i == 0 or i == len(categories) - 1:
            # Start and end bars
            axes[0, 0].bar(i, cumulative[i], color=colors_waterfall[i], alpha=0.7, edgecolor='black')
        else:
            # Change bars
            if values[i] >= 0:
                axes[0, 0].bar(i, values[i], bottom=cumulative[i-1],
                              color=colors_waterfall[i], alpha=0.7, edgecolor='black')
            else:
                axes[0, 0].bar(i, abs(values[i]), bottom=cumulative[i],
                              color=colors_waterfall[i], alpha=0.7, edgecolor='black')

            # Connecting lines
            axes[0, 0].plot([i-0.4, i-0.4], [cumulative[i-1], cumulative[i]],
                           'k--', linewidth=1, alpha=0.5)

    axes[0, 0].set_xticks(range(len(categories)))
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].set_ylabel('CO2 Emissions (million tons)')
    axes[0, 0].set_title('Waterfall Decomposition')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (cat, val) in enumerate(zip(categories, cumulative)):
        axes[0, 0].text(i, val + 20, f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')

    # Plot 2: Effect contributions (pie chart)
    effect_values = [abs(results['scale_effect']), abs(results['composition_effect']),
                     abs(results['intensity_effect'])]
    effect_labels = ['Scale\n(GDP Growth)', 'Composition\n(Structural Change)',
                     'Intensity\n(Technology)']
    effect_colors = ['#4CAF50', '#FF9800', '#D32F2F']

    axes[0, 1].pie(effect_values, labels=effect_labels, autopct='%1.1f%%',
                   colors=effect_colors, startangle=90)
    axes[0, 1].set_title('Decomposition of Absolute Effects')

    # Plot 3: Sectoral structure change
    x = np.arange(len(sectors))
    width = 0.35

    axes[1, 0].bar(x - width/2, sector_shares_1990 * 100, width,
                   label='1990', alpha=0.8, color='#1976D2')
    axes[1, 0].bar(x + width/2, sector_shares_2020 * 100, width,
                   label='2020', alpha=0.8, color='#D32F2F')
    axes[1, 0].set_xlabel('Sector')
    axes[1, 0].set_ylabel('Share of GDP (%)')
    axes[1, 0].set_title('Structural Change: Sectoral Composition')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(sectors)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Emission intensity change
    axes[1, 1].bar(x - width/2, emission_intensity_1990, width,
                   label='1990', alpha=0.8, color='#1976D2')
    axes[1, 1].bar(x + width/2, emission_intensity_2020, width,
                   label='2020', alpha=0.8, color='#D32F2F')
    axes[1, 1].set_xlabel('Sector')
    axes[1, 1].set_ylabel('CO2 Intensity (tons per $M GDP)')
    axes[1, 1].set_title('Technological Change: Emission Intensity')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(sectors)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('phase3-exercises/sectoral/ex2_structural_decomposition.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: phase3-exercises/sectoral/ex2_structural_decomposition.png")

    # ECONOMIC INTERPRETATION
    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION:")
    print("=" * 80)
    print("\n1. Scale Effect (GDP Growth):")
    print(f"   Value: +{results['scale_effect']:.1f} million tons")
    print("   - Economic growth increases emissions (all else equal)")
    print("   - Doubling GDP → substantial emission pressure")
    print("   - This is the 'throughput' problem in ecological economics")

    print("\n2. Composition Effect (Structural Change):")
    print(f"   Value: {results['composition_effect']:.1f} million tons")
    if results['composition_effect'] < 0:
        print("   - NEGATIVE: Shift toward cleaner sectors reduces emissions")
        print("   - Manufacturing share fell (high emissions)")
        print("   - Services share rose (lower emissions)")
        print("   - This is the 'composition' part of Environmental Kuznets Curve")
    else:
        print("   - POSITIVE: Shift toward dirtier sectors")

    print("\n3. Intensity Effect (Technological Change):")
    print(f"   Value: {results['intensity_effect']:.1f} million tons")
    if results['intensity_effect'] < 0:
        print("   - NEGATIVE: Cleaner technology reduces emissions")
        print("   - All sectors improved emission efficiency")
        print("   - Reflects:")
        print("     * Energy efficiency improvements")
        print("     * Shift to cleaner energy sources")
        print("     * Environmental regulations")
        print("     * Technological innovation")
    else:
        print("   - POSITIVE: Technology worsening (rare)")

    print("\n4. Net Effect:")
    if results['total_change'] > 0:
        print(f"   Emissions INCREASED by {results['total_change']:.1f} million tons")
        print("   - Scale effect dominated composition + intensity effects")
        print("   - Economic growth outpaced efficiency gains")
        print("   - Absolute decoupling NOT achieved")
    else:
        print(f"   Emissions DECREASED by {abs(results['total_change']):.1f} million tons")
        print("   - Composition + intensity effects dominated scale")
        print("   - Absolute decoupling achieved!")

    print("\n5. Environmental Kuznets Curve (EKC) Context:")
    print("   EKC hypothesis: Emissions rise with income, then fall")
    print("   - Our example shows potential turning point dynamics")
    print("   - Scale effect: Why poor countries have rising emissions")
    print("   - Composition + intensity: Why rich countries can reduce emissions")
    print("   - BUT: Heterodox critique of EKC:")
    print("     * May reflect offshoring (consumption vs production emissions)")
    print("     * Not automatic - requires policy (regulations, carbon pricing)")
    print("     * Historical path dependence matters")

    print("\n6. Policy Implications:")
    print("   To reduce emissions, policymakers can target:")
    print("   a) Scale: Manage economic growth (degrowth debate)")
    print("      - Controversial: Growth needed for development")
    print("      - Alternative: Green growth, circular economy")
    print("   b) Composition: Industrial policy for structural change")
    print("      - Support low-emission sectors")
    print("      - Phase out high-emission industries")
    print("      - BUT: Deindustrialization can hurt employment")
    print("   c) Intensity: Technology policy")
    print("      - R&D subsidies for clean tech")
    print("      - Carbon pricing to incentivize efficiency")
    print("      - Regulations (emission standards)")
    print("\n   Heterodox perspective:")
    print("   - Market alone won't deliver (externalities, coordination failures)")
    print("   - Need active state role (green industrial policy)")
    print("   - Just transition crucial (protect workers in dirty industries)")

    return results, decomp_summary, sectoral_df


# ============================================================================
# EXTENSION CHALLENGES
# ============================================================================

def extension_challenges():
    """
    Advanced exercises for deeper exploration
    """
    print("\n" + "=" * 80)
    print("EXTENSION CHALLENGES")
    print("=" * 80)

    challenges = [
        {
            "title": "Challenge 1: Multi-Period Shift-Share",
            "description": "Extend shift-share to multiple time periods (e.g., by decade). "
                          "Analyze how sources of regional change evolved over time. "
                          "Did competitive effects strengthen or weaken?",
            "skills": "Panel analysis, temporal decomposition",
            "reference": "Barff & Knight (1988)"
        },
        {
            "title": "Challenge 2: Spatial Shift-Share",
            "description": "Implement spatial extension that includes neighbor effects. "
                          "Does regional performance depend on neighboring regions? "
                          "Apply to understanding regional clusters and spillovers.",
            "skills": "Spatial econometrics, network effects",
            "reference": "Nazara & Hewings (2004)"
        },
        {
            "title": "Challenge 3: Input-Output SDA",
            "description": "Implement full input-output SDA using technical coefficient matrices. "
                          "Decompose sectoral output changes into final demand, technology, "
                          "and Leontief inverse effects. More complex than simple IDA.",
            "skills": "Matrix algebra, I-O economics",
            "reference": "Miller & Blair (2009), Hoekstra & van den Bergh (2003)"
        },
        {
            "title": "Challenge 4: Inequality Decomposition by Sector",
            "description": "Combine distributional and sectoral analysis. Decompose income "
                          "inequality changes into: (1) within-sector inequality, "
                          "(2) between-sector inequality, (3) employment reallocation. "
                          "How much does deindustrialization explain rising inequality?",
            "skills": "Theil decomposition, variance analysis",
            "reference": "Mookherjee & Shorrocks (1982)"
        },
        {
            "title": "Challenge 5: Productivity Decomposition",
            "description": "Decompose aggregate productivity growth into: "
                          "(1) within-sector productivity growth, (2) reallocation effects "
                          "(labor moving from low to high productivity sectors). "
                          "Apply to understanding productivity slowdown.",
            "skills": "Growth accounting, decomposition methods",
            "reference": "Baily, Hulten & Campbell (1992)"
        },
        {
            "title": "Challenge 6: Consumption-Based Emissions",
            "description": "Extend SDA to consumption-based (vs production-based) emissions. "
                          "Use input-output to allocate embodied emissions in imports/exports. "
                          "How much have rich countries 'offshored' their emissions?",
            "skills": "Multi-region I-O, carbon footprinting",
            "reference": "Peters & Hertwich (2008), Davis & Caldeira (2010)"
        }
    ]

    for i, challenge in enumerate(challenges, 1):
        print(f"\n{challenge['title']}")
        print(f"  Description: {challenge['description']}")
        print(f"  Skills: {challenge['skills']}")
        print(f"  Reference: {challenge['reference']}")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("SECTORAL ANALYSIS EXERCISES - HETERODOX ECONOMICS")
    print("=" * 80)
    print("\nThis module contains exercises on sectoral dynamics:")
    print("1. Shift-share analysis (regional deindustrialization)")
    print("2. Structural decomposition (emissions, EKC)")
    print("\nEach exercise includes:")
    print("  ✓ Economic problem grounded in structural economics")
    print("  ✓ Complete Python implementation")
    print("  ✓ Detailed decomposition results")
    print("  ✓ Heterodox economic interpretation")
    print("  ✓ Policy implications")
    print("  ✓ Extension challenges")
    print("\n" + "=" * 80)

    # Run exercises
    shift_share_results, nat_data, reg_data = exercise_1_shift_share_analysis()
    sda_results, sda_summary, sda_sectoral = exercise_2_structural_decomposition()
    extension_challenges()

    print("\n" + "=" * 80)
    print("ALL EXERCISES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run this file: python 03_sectoral_analysis_exercises.py")
    print("2. Examine the decomposition visualizations")
    print("3. Apply to real data (BLS employment, EPA emissions)")
    print("4. Attempt the extension challenges")
    print("\nKey Takeaway:")
    print("Decomposition methods reveal the drivers of structural change.")
    print("Different components have different policy levers.")
    print("Understanding 'what changed' is first step to 'how to change it'.")
    print("=" * 80 + "\n")
