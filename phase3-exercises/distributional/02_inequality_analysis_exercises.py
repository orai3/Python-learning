"""
Distributional Analysis Exercises: Inequality Metrics & Lorenz Curves
Heterodox Economics Focus

Exercises cover inequality measurement with emphasis on Post-Keynesian,
Marxian, and institutionalist perspectives on distribution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXERCISE 1: Lorenz Curves and Gini Coefficient
# ============================================================================
# THEORY: Lorenz curve visualizes income/wealth distribution. Gini coefficient
# quantifies inequality. Essential for analyzing functional & personal distribution.
# Reference: Lorenz (1905), Gini (1912), Atkinson (1970)
# ============================================================================

def exercise_1_lorenz_and_gini():
    """
    Problem: Calculate and visualize Lorenz curves and Gini coefficients for
    different distributional scenarios (egalitarian, moderate, highly unequal).

    Reference: Atkinson (1970), Piketty (2014)
    """
    print("=" * 80)
    print("EXERCISE 1: Lorenz Curves and Gini Coefficient")
    print("=" * 80)

    # Generate synthetic income distributions
    np.random.seed(42)
    n = 10000

    # Scenario 1: Relatively egalitarian (Nordic model)
    income_nordic = np.random.lognormal(mean=10.5, sigma=0.5, size=n)

    # Scenario 2: Moderate inequality (Post-war US/Europe)
    income_moderate = np.random.lognormal(mean=10.5, sigma=0.8, size=n)

    # Scenario 3: High inequality (Contemporary US)
    income_high = np.random.lognormal(mean=10.5, sigma=1.2, size=n)

    # Scenario 4: Extreme inequality (Gilded Age)
    # Create with heavy right tail
    income_extreme = np.concatenate([
        np.random.lognormal(mean=9.5, sigma=0.6, size=int(0.9*n)),  # Bottom 90%
        np.random.lognormal(mean=12.5, sigma=0.8, size=int(0.1*n))  # Top 10%
    ])

    scenarios = {
        'Nordic Model': income_nordic,
        'Post-War': income_moderate,
        'Contemporary US': income_high,
        'Gilded Age': income_extreme
    }

    # SOLUTION: Implement Lorenz curve and Gini coefficient
    def lorenz_curve(incomes):
        """
        Calculate Lorenz curve coordinates

        Args:
            incomes: Array of income values

        Returns:
            cum_pop: Cumulative population share
            cum_income: Cumulative income share
        """
        # Sort incomes
        sorted_incomes = np.sort(incomes)

        # Calculate cumulative shares
        cum_income = np.cumsum(sorted_incomes)
        cum_income = cum_income / cum_income[-1]  # Normalize to [0, 1]

        # Population shares
        n = len(incomes)
        cum_pop = np.arange(1, n + 1) / n

        # Add origin point
        cum_pop = np.insert(cum_pop, 0, 0)
        cum_income = np.insert(cum_income, 0, 0)

        return cum_pop, cum_income

    def gini_coefficient(incomes):
        """
        Calculate Gini coefficient

        Args:
            incomes: Array of income values

        Returns:
            gini: Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        sorted_incomes = np.sort(incomes)
        n = len(incomes)
        cumsum = np.cumsum(sorted_incomes)

        # Gini formula: G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n + 1) / n
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_incomes)) / (n * np.sum(sorted_incomes)) - (n + 1) / n

        return gini

    def percentile_shares(incomes, percentiles=[10, 50, 90]):
        """
        Calculate income shares by percentile

        Args:
            incomes: Array of income values
            percentiles: List of percentile cutoffs

        Returns:
            Dictionary of shares
        """
        sorted_incomes = np.sort(incomes)
        total_income = np.sum(sorted_incomes)

        shares = {}
        prev_pct = 0
        for pct in percentiles:
            idx_start = int(len(sorted_incomes) * prev_pct / 100)
            idx_end = int(len(sorted_incomes) * pct / 100)
            share = np.sum(sorted_incomes[idx_start:idx_end]) / total_income * 100
            shares[f'P{prev_pct}-P{pct}'] = share
            prev_pct = pct

        # Top percentile
        idx = int(len(sorted_incomes) * percentiles[-1] / 100)
        shares[f'P{percentiles[-1]}-P100'] = np.sum(sorted_incomes[idx:]) / total_income * 100

        return shares

    # Calculate metrics for all scenarios
    results = {}
    for name, incomes in scenarios.items():
        cum_pop, cum_income = lorenz_curve(incomes)
        gini = gini_coefficient(incomes)
        shares = percentile_shares(incomes)

        results[name] = {
            'cum_pop': cum_pop,
            'cum_income': cum_income,
            'gini': gini,
            'shares': shares,
            'mean': np.mean(incomes),
            'median': np.median(incomes)
        }

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Income Distribution Analysis Across Historical Regimes',
                 fontsize=14, fontweight='bold')

    # Plot 1: Lorenz Curves
    colors = ['#2E7D32', '#1976D2', '#D32F2F', '#7B1FA2']
    for (name, data), color in zip(results.items(), colors):
        axes[0, 0].plot(data['cum_pop'], data['cum_income'],
                       label=f"{name} (Gini={data['gini']:.3f})",
                       linewidth=2.5, color=color)

    # Perfect equality line
    axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Equality', alpha=0.5)
    axes[0, 0].set_xlabel('Cumulative Population Share', fontsize=11)
    axes[0, 0].set_ylabel('Cumulative Income Share', fontsize=11)
    axes[0, 0].set_title('Lorenz Curves Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].legend(loc='upper left', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1])

    # Add shading for one example (Contemporary US)
    us_data = results['Contemporary US']
    axes[0, 0].fill_between(us_data['cum_pop'], us_data['cum_income'],
                            us_data['cum_pop'], alpha=0.2, color='#D32F2F')

    # Plot 2: Gini Coefficients Comparison
    gini_values = [data['gini'] for data in results.values()]
    bars = axes[0, 1].bar(range(len(scenarios)), gini_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(range(len(scenarios)))
    axes[0, 1].set_xticklabels(scenarios.keys(), rotation=15, ha='right')
    axes[0, 1].set_ylabel('Gini Coefficient', fontsize=11)
    axes[0, 1].set_title('Inequality Comparison (Gini Coefficient)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 0.8])
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, gini in zip(bars, gini_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{gini:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Income shares by percentile
    percentile_groups = ['P0-P10', 'P10-P50', 'P50-P90', 'P90-P100']
    x = np.arange(len(percentile_groups))
    width = 0.2

    for i, (name, data) in enumerate(results.items()):
        shares_values = [data['shares'][group] for group in percentile_groups]
        axes[1, 0].bar(x + i*width, shares_values, width, label=name,
                      color=colors[i], alpha=0.7, edgecolor='black')

    axes[1, 0].set_xlabel('Population Percentile', fontsize=11)
    axes[1, 0].set_ylabel('Income Share (%)', fontsize=11)
    axes[1, 0].set_title('Income Share by Percentile Group', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(x + width * 1.5)
    axes[1, 0].set_xticklabels(percentile_groups)
    axes[1, 0].legend(fontsize=8, loc='upper left')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Mean vs Median (inequality indicator)
    mean_median_ratios = [(data['mean'] / data['median']) for data in results.values()]
    bars = axes[1, 1].bar(range(len(scenarios)), mean_median_ratios,
                         color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(range(len(scenarios)))
    axes[1, 1].set_xticklabels(scenarios.keys(), rotation=15, ha='right')
    axes[1, 1].set_ylabel('Mean / Median Ratio', fontsize=11)
    axes[1, 1].set_title('Mean-Median Ratio (Skewness Indicator)', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=1, color='red', linestyle='--', linewidth=1.5, label='No skew (ratio=1)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    for bar, ratio in zip(bars, mean_median_ratios):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{ratio:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('phase3-exercises/distributional/ex1_lorenz_gini.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: phase3-exercises/distributional/ex1_lorenz_gini.png")

    # Print detailed results
    print("\nDETAILED INEQUALITY METRICS:")
    print("=" * 80)
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Gini Coefficient: {data['gini']:.4f}")
        print(f"  Mean Income: ${data['mean']:,.2f}")
        print(f"  Median Income: ${data['median']:,.2f}")
        print(f"  Mean/Median Ratio: {data['mean']/data['median']:.3f}")
        print(f"  Income Shares:")
        for group, share in data['shares'].items():
            print(f"    {group}: {share:.2f}%")

    # ECONOMIC INTERPRETATION
    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION:")
    print("=" * 80)
    print("1. Historical Context:")
    print("   - Nordic Model: Strong unions, welfare state, progressive taxation")
    print("   - Post-War: Embedded liberalism, capital-labor accord")
    print("   - Contemporary US: Neoliberalism, financialization, union decline")
    print("   - Gilded Age: Laissez-faire, weak labor, robber barons")
    print("\n2. Gini Coefficient Interpretation:")
    print("   - 0.0 = Perfect equality (everyone has same income)")
    print("   - 0.25-0.35 = Low inequality (Nordic countries)")
    print("   - 0.35-0.45 = Moderate inequality (Western Europe)")
    print("   - 0.45+ = High inequality (US, Latin America)")
    print("   - 0.6+ = Extreme inequality (historical or failed states)")
    print("\n3. Top 10% Share - Key Indicator:")
    print("   - Nordic: ~25-30% (moderate concentration)")
    print("   - Post-War: ~30-35% (manageable)")
    print("   - Contemporary: ~45-50% (high concentration)")
    print("   - Gilded Age: ~50%+ (plutocratic)")
    print("\n4. Heterodox Perspective:")
    print("   - Inequality is NOT a 'natural' market outcome")
    print("   - Reflects power relations, institutions, policy choices")
    print("   - Different regimes produce radically different distributions")
    print("   - High inequality can undermine demand (wage-led growth)")
    print("\n5. Policy Implications:")
    print("   - Progressive taxation can dramatically reduce inequality")
    print("   - Labor market institutions matter (unions, minimum wage)")
    print("   - Financial regulation affects top-end concentration")
    print("   - Social programs reduce bottom-end deprivation")

    return results


# ============================================================================
# EXERCISE 2: Palma Ratio and Alternative Inequality Measures
# ============================================================================
# THEORY: Palma ratio focuses on extremes (top 10% vs bottom 40%), arguing
# middle class share is relatively stable. Alternative to Gini for policy.
# Reference: Palma (2011), Cobham & Sumner (2013)
# ============================================================================

def exercise_2_palma_ratio():
    """
    Problem: Calculate multiple inequality measures and compare their
    sensitivity to different parts of the distribution.

    Reference: Palma (2011), Atkinson (1970)
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Palma Ratio and Alternative Inequality Measures")
    print("=" * 80)

    # Generate income distribution data
    np.random.seed(42)
    n = 10000

    # Base distribution
    base_income = np.random.lognormal(mean=10.5, sigma=0.9, size=n)

    # Create variations by manipulating different parts of distribution
    scenarios = {}

    # 1. Base case
    scenarios['Base'] = base_income.copy()

    # 2. Bottom squeeze: Reduce bottom 40% incomes by 20%
    bottom_squeeze = base_income.copy()
    bottom_40_idx = int(0.4 * n)
    sorted_idx = np.argsort(bottom_squeeze)
    bottom_squeeze[sorted_idx[:bottom_40_idx]] *= 0.8
    scenarios['Bottom Squeeze'] = bottom_squeeze

    # 3. Top surge: Increase top 10% incomes by 50%
    top_surge = base_income.copy()
    top_10_idx = int(0.9 * n)
    sorted_idx = np.argsort(top_surge)
    top_surge[sorted_idx[top_10_idx:]] *= 1.5
    scenarios['Top Surge'] = top_surge

    # 4. Middle erosion: Reduce middle 50% by 10%, redistribute to top
    middle_erosion = base_income.copy()
    sorted_idx = np.argsort(middle_erosion)
    middle_start = int(0.4 * n)
    middle_end = int(0.9 * n)
    middle_erosion[sorted_idx[middle_start:middle_end]] *= 0.9
    middle_erosion[sorted_idx[middle_end:]] *= 1.3
    scenarios['Middle Erosion'] = middle_erosion

    # SOLUTION: Implement multiple inequality measures
    def inequality_measures(incomes):
        """
        Calculate comprehensive set of inequality measures

        Args:
            incomes: Array of income values

        Returns:
            Dictionary of inequality metrics
        """
        sorted_incomes = np.sort(incomes)
        n = len(incomes)
        total_income = np.sum(incomes)

        # Percentile cutoffs
        p10 = int(0.1 * n)
        p40 = int(0.4 * n)
        p50 = int(0.5 * n)
        p60 = int(0.6 * n)
        p90 = int(0.9 * n)

        # Income shares
        bottom_10_share = np.sum(sorted_incomes[:p10]) / total_income
        bottom_40_share = np.sum(sorted_incomes[:p40]) / total_income
        middle_50_share = np.sum(sorted_incomes[p40:p90]) / total_income
        top_10_share = np.sum(sorted_incomes[p90:]) / total_income
        top_1_share = np.sum(sorted_incomes[int(0.99*n):]) / total_income

        # Palma Ratio: Top 10% / Bottom 40%
        palma = top_10_share / bottom_40_share

        # Gini coefficient (from previous exercise)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_incomes)) / (n * total_income) - (n + 1) / n

        # P90/P10 ratio (decile ratio)
        p90_p10_ratio = sorted_incomes[p90] / sorted_incomes[p10]

        # P90/P50 ratio (top inequality)
        p90_p50_ratio = sorted_incomes[p90] / sorted_incomes[p50]

        # P50/P10 ratio (bottom inequality)
        p50_p10_ratio = sorted_incomes[p50] / sorted_incomes[p10]

        # Atkinson index (ε = 0.5, moderate inequality aversion)
        epsilon = 0.5
        if epsilon == 1:
            atkinson = 1 - np.exp(np.mean(np.log(sorted_incomes))) / np.mean(sorted_incomes)
        else:
            mean_income = np.mean(sorted_incomes)
            atkinson = 1 - (np.mean(sorted_incomes ** (1 - epsilon)) ** (1 / (1 - epsilon))) / mean_income

        # Theil index
        mean_income = np.mean(sorted_incomes)
        theil = np.mean((sorted_incomes / mean_income) * np.log(sorted_incomes / mean_income))

        return {
            'gini': gini,
            'palma': palma,
            'p90_p10': p90_p10_ratio,
            'p90_p50': p90_p50_ratio,
            'p50_p10': p50_p10_ratio,
            'atkinson': atkinson,
            'theil': theil,
            'bottom_10_share': bottom_10_share * 100,
            'bottom_40_share': bottom_40_share * 100,
            'middle_50_share': middle_50_share * 100,
            'top_10_share': top_10_share * 100,
            'top_1_share': top_1_share * 100
        }

    # Calculate metrics for all scenarios
    results = {name: inequality_measures(inc) for name, inc in scenarios.items()}

    # Create comparison DataFrame
    metrics_df = pd.DataFrame(results).T
    print("\nINEQUALITY METRICS COMPARISON:")
    print("=" * 80)
    print(metrics_df.round(3))

    # Calculate percentage changes from base
    pct_changes = {}
    base_metrics = results['Base']
    for scenario in ['Bottom Squeeze', 'Top Surge', 'Middle Erosion']:
        changes = {}
        for metric in base_metrics.keys():
            base_val = base_metrics[metric]
            new_val = results[scenario][metric]
            pct_change = ((new_val - base_val) / base_val) * 100
            changes[metric] = pct_change
        pct_changes[scenario] = changes

    pct_changes_df = pd.DataFrame(pct_changes).T
    print("\nPERCENTAGE CHANGES FROM BASE (%):")
    print("=" * 80)
    print(pct_changes_df.round(2))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Alternative Inequality Measures: Sensitivity Analysis',
                 fontsize=14, fontweight='bold')

    # Plot 1: Key ratios comparison
    scenarios_list = list(scenarios.keys())
    x = np.arange(len(scenarios_list))
    width = 0.25

    palma_vals = [results[s]['palma'] for s in scenarios_list]
    gini_vals = [results[s]['gini'] * 5 for s in scenarios_list]  # Scale for visibility
    p90_p10_vals = [results[s]['p90_p10'] / 2 for s in scenarios_list]  # Scale

    axes[0, 0].bar(x - width, palma_vals, width, label='Palma Ratio', alpha=0.8)
    axes[0, 0].bar(x, gini_vals, width, label='Gini × 5', alpha=0.8)
    axes[0, 0].bar(x + width, p90_p10_vals, width, label='P90/P10 ÷ 2', alpha=0.8)
    axes[0, 0].set_xlabel('Scenario')
    axes[0, 0].set_ylabel('Index Value (scaled for comparison)')
    axes[0, 0].set_title('Key Inequality Indices Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(scenarios_list, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Income shares
    bottom_40 = [results[s]['bottom_40_share'] for s in scenarios_list]
    middle_50 = [results[s]['middle_50_share'] for s in scenarios_list]
    top_10 = [results[s]['top_10_share'] for s in scenarios_list]

    axes[0, 1].bar(x, bottom_40, width*2, label='Bottom 40%', alpha=0.8, color='#2196F3')
    axes[0, 1].bar(x, middle_50, width*2, bottom=bottom_40,
                   label='Middle 50%', alpha=0.8, color='#4CAF50')
    axes[0, 1].bar(x, top_10, width*2, bottom=np.array(bottom_40) + np.array(middle_50),
                   label='Top 10%', alpha=0.8, color='#F44336')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Income Share (%)')
    axes[0, 1].set_title('Income Distribution by Group')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(scenarios_list, rotation=15, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Sensitivity heatmap
    metrics_to_plot = ['gini', 'palma', 'p90_p10', 'atkinson', 'theil']
    sensitivity_data = pct_changes_df[metrics_to_plot].values

    im = axes[1, 0].imshow(sensitivity_data, cmap='RdYlGn_r', aspect='auto', vmin=-10, vmax=50)
    axes[1, 0].set_xticks(np.arange(len(metrics_to_plot)))
    axes[1, 0].set_yticks(np.arange(len(pct_changes)))
    axes[1, 0].set_xticklabels(metrics_to_plot)
    axes[1, 0].set_yticklabels(pct_changes.keys())
    axes[1, 0].set_title('Metric Sensitivity to Distributional Changes (%)')

    # Add values to heatmap
    for i in range(len(pct_changes)):
        for j in range(len(metrics_to_plot)):
            text = axes[1, 0].text(j, i, f'{sensitivity_data[i, j]:.1f}',
                                  ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=axes[1, 0], label='% Change from Base')

    # Plot 4: Palma components
    top_10_shares = [results[s]['top_10_share'] for s in scenarios_list]
    bottom_40_shares = [results[s]['bottom_40_share'] for s in scenarios_list]

    x_pos = np.arange(len(scenarios_list))
    axes[1, 1].plot(x_pos, top_10_shares, 'ro-', linewidth=2, markersize=10,
                   label='Top 10% Share', alpha=0.7)
    axes[1, 1].plot(x_pos, bottom_40_shares, 'bo-', linewidth=2, markersize=10,
                   label='Bottom 40% Share', alpha=0.7)
    axes[1, 1].set_xlabel('Scenario')
    axes[1, 1].set_ylabel('Income Share (%)')
    axes[1, 1].set_title('Palma Ratio Components')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(scenarios_list, rotation=15, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase3-exercises/distributional/ex2_palma_ratio.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: phase3-exercises/distributional/ex2_palma_ratio.png")

    # ECONOMIC INTERPRETATION
    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION:")
    print("=" * 80)
    print("1. Palma Ratio Advantages:")
    print("   - Focuses on distributional extremes (policy-relevant)")
    print("   - Middle 50% share relatively stable empirically (Palma's observation)")
    print("   - More intuitive than Gini: ratio of actual shares")
    print("   - Sensitive to both top and bottom changes")
    print("\n2. Metric Sensitivity Analysis:")
    print("   - Gini: Moderately sensitive to all distributional changes")
    print("   - Palma: Highly sensitive to top/bottom, insensitive to middle")
    print("   - P90/P10: Sensitive to extremes but ignores middle mass")
    print("   - Atkinson: Customizable inequality aversion (ε parameter)")
    print("\n3. Policy Implications by Scenario:")
    print("   a) Bottom Squeeze (e.g., benefit cuts, minimum wage erosion):")
    print("      - Palma rises sharply (bottom 40% share falls)")
    print("      - P50/P10 ratio increases (bottom falling behind median)")
    print("      - Policy: Strengthen social safety net, raise minimum wage")
    print("   b) Top Surge (e.g., financial deregulation, top tax cuts):")
    print("      - All metrics rise, but Palma especially responsive")
    print("      - Top 10% and especially top 1% shares balloon")
    print("      - Policy: Progressive taxation, financial regulation")
    print("   c) Middle Erosion (e.g., automation, offshoring):")
    print("      - 'Hollowing out' of middle class")
    print("      - Bifurcation into high-skill/low-skill jobs")
    print("      - Policy: Education, industrial policy, union support")
    print("\n4. Heterodox Perspective:")
    print("   - Choice of metric reflects implicit value judgments")
    print("   - Palma ratio emphasizes power relations (top vs bottom)")
    print("   - No single metric captures all aspects of inequality")
    print("   - Need to examine entire distribution, not just summary stats")
    print("\n5. Real-World Application:")
    print("   - Compare across countries: Nordic (Palma ~0.8) vs US (Palma ~1.7)")
    print("   - Track over time: US Palma rose from ~1.0 (1970s) to ~1.7 (2010s)")
    print("   - Palma < 1 → bottom 40% have more than top 10% (rare)")
    print("   - Palma > 2 → extreme inequality (many developing countries)")

    return results, scenarios


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
            "title": "Challenge 1: Functional vs Personal Distribution",
            "description": "Decompose personal income inequality into functional distribution "
                          "(labor vs capital income) and within-category inequality. How much of "
                          "total inequality is due to capital concentration vs wage dispersion? "
                          "Use Theil decomposition.",
            "skills": "Decomposition methods, variance analysis",
            "reference": "Atkinson (2009), Piketty & Saez (2003)"
        },
        {
            "title": "Challenge 2: Generalized Entropy Indices",
            "description": "Implement full family of generalized entropy measures (GE(α) for various α). "
                          "Analyze how sensitivity to different parts of distribution changes with α. "
                          "When would you use GE(0) (Theil-L) vs GE(1) (Theil-T) vs GE(2)?",
            "skills": "Entropy measures, parameter sensitivity",
            "reference": "Shorrocks (1980), Cowell (2011)"
        },
        {
            "title": "Challenge 3: Wealth Concentration Analysis",
            "description": "Extend analysis to wealth distribution (much more unequal than income). "
                          "Calculate wealth Gini, top 1% and 0.1% shares. Model Pareto distribution "
                          "for top tail. Compare US wealth concentration (Gini ~0.85) to income (~0.48).",
            "skills": "Pareto distribution, tail fitting, wealth data",
            "reference": "Saez & Zucman (2016), Piketty (2014)"
        },
        {
            "title": "Challenge 4: Distributional National Accounts",
            "description": "Implement DINA (Distributional National Accounts) methodology to ensure "
                          "micro inequality data matches macro aggregates. Allocate 100% of national "
                          "income to individuals, including imputed components (retained earnings, etc.).",
            "skills": "Data reconciliation, imputation, macro-micro linkage",
            "reference": "Piketty, Saez & Zucman (2018)"
        },
        {
            "title": "Challenge 5: Inequality Decomposition by Source",
            "description": "Decompose inequality by income source (wages, capital, transfers, etc.). "
                          "Calculate 'pseudo-Gini' for each source and source contributions. "
                          "Which sources are inequality-increasing vs equalizing?",
            "skills": "Gini decomposition, Shapley values",
            "reference": "Lerman & Yitzhaki (1985), Shorrocks (2013)"
        },
        {
            "title": "Challenge 6: Inequality and Growth Regimes",
            "description": "Build simple macro model linking inequality to growth via consumption/demand. "
                          "Simulate 'wage-led' vs 'profit-led' growth regimes. At what inequality level "
                          "does demand constraint bind?",
            "skills": "Macro modeling, Kaleckian/Post-Keynesian theory",
            "reference": "Bhaduri & Marglin (1990), Stockhammer (2017)"
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
    print("DISTRIBUTIONAL ANALYSIS EXERCISES - HETERODOX ECONOMICS")
    print("=" * 80)
    print("\nThis module contains exercises on inequality measurement:")
    print("1. Lorenz curves and Gini coefficient (basic)")
    print("2. Palma ratio and alternative measures (intermediate)")
    print("\nEach exercise includes:")
    print("  ✓ Economic problem grounded in distributional economics")
    print("  ✓ Complete Python implementation")
    print("  ✓ Comparative visualizations")
    print("  ✓ Heterodox economic interpretation")
    print("  ✓ Policy implications")
    print("  ✓ Extension challenges")
    print("\n" + "=" * 80)

    # Run exercises
    lorenz_results = exercise_1_lorenz_and_gini()
    palma_results, palma_scenarios = exercise_2_palma_ratio()
    extension_challenges()

    print("\n" + "=" * 80)
    print("ALL EXERCISES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run this file: python 02_inequality_analysis_exercises.py")
    print("2. Examine the generated visualizations")
    print("3. Try calculating these metrics with real data (OECD, World Bank)")
    print("4. Attempt the extension challenges")
    print("\nKey Takeaway:")
    print("Inequality is multidimensional - different metrics reveal different aspects.")
    print("Policy interventions affect different parts of distribution differently.")
    print("Understanding which metric to use requires understanding what you care about.")
    print("=" * 80 + "\n")
