"""
Comprehensive Example: Agent-Based Macroeconomic Model

This script demonstrates the full capabilities of the ABM:
1. Basic simulation
2. Policy experiments (austerity, QE, fiscal expansion)
3. Comparison with representative agent model
4. Visualization and analysis

For heterodox economics research and teaching.
"""

import numpy as np
import matplotlib.pyplot as plt
from abm_macro import (
    MacroeconomyABM,
    ABMVisualizer,
    PolicyExperiment,
    RepresentativeAgentModel,
    compare_abm_vs_representative,
    compare_policy_experiments
)


def example_1_basic_simulation():
    """
    Example 1: Basic ABM simulation showing emergent macro dynamics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC ABM SIMULATION")
    print("="*80)

    # Create economy
    economy = MacroeconomyABM(
        n_firms=1000,
        n_households=5000,
        n_banks=10,
        random_seed=42
    )

    # Run simulation
    results = economy.run(n_periods=300)

    # Visualize
    viz = ABMVisualizer(economy)

    print("\n📊 Creating visualizations...")

    # Macro dashboard
    viz.plot_macro_dashboard(save_path='output/abm_dashboard.png')

    # Distributional analysis
    viz.plot_distributional_analysis(save_path='output/abm_distribution.png')

    # Business cycle analysis
    viz.plot_business_cycle_analysis(save_path='output/abm_cycles.png')

    print("\n✓ Example 1 complete!")

    return economy, results


def example_2_policy_experiments():
    """
    Example 2: Compare different policy regimes.

    Tests:
    - Baseline (no intervention)
    - Fiscal austerity
    - Quantitative easing
    - Green New Deal (fiscal expansion)
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: POLICY EXPERIMENTS")
    print("="*80)

    results_dict = {}

    # 1. Baseline
    print("\n1️⃣ Running BASELINE scenario...")
    econ_baseline = MacroeconomyABM(n_firms=800, n_households=4000, n_banks=10, random_seed=42)
    results_dict['Baseline'] = econ_baseline.run(n_periods=200)

    # 2. Fiscal Austerity
    print("\n2️⃣ Running FISCAL AUSTERITY scenario...")
    econ_austerity = MacroeconomyABM(n_firms=800, n_households=4000, n_banks=10, random_seed=42)
    austerity_exp = PolicyExperiment.create_austerity_experiment()
    results_dict['Austerity'] = econ_austerity.run(n_periods=200, policy_experiment=austerity_exp)

    # 3. Quantitative Easing
    print("\n3️⃣ Running QUANTITATIVE EASING scenario...")
    econ_qe = MacroeconomyABM(n_firms=800, n_households=4000, n_banks=10, random_seed=42)
    qe_exp = PolicyExperiment.create_qe_experiment()
    results_dict['QE'] = econ_qe.run(n_periods=200, policy_experiment=qe_exp)

    # 4. Green New Deal
    print("\n4️⃣ Running GREEN NEW DEAL scenario...")
    econ_gnd = MacroeconomyABM(n_firms=800, n_households=4000, n_banks=10, random_seed=42)
    gnd_exp = PolicyExperiment.create_green_new_deal_experiment()
    results_dict['Green New Deal'] = econ_gnd.run(n_periods=200, policy_experiment=gnd_exp)

    # Compare
    print("\n📊 Creating policy comparison plots...")
    compare_policy_experiments(results_dict, save_path='output/policy_comparison.png')

    # Print summary statistics
    print("\n" + "="*80)
    print("POLICY EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    for name, results in results_dict.items():
        avg_gdp = np.mean(results.time_series['gdp'][-50:])
        avg_unemp = np.mean(results.time_series['unemployment_rate'][-50:])
        avg_gini = np.mean(results.time_series['gini_wealth'][-50:])
        final_debt = results.time_series['public_debt'][-1]

        print(f"\n{name}:")
        print(f"  Average GDP (last 50 periods): {avg_gdp:.1f}")
        print(f"  Average Unemployment: {avg_unemp:.1%}")
        print(f"  Average Wealth Gini: {avg_gini:.3f}")
        print(f"  Final Public Debt: {final_debt:.1f}")

    print("\n✓ Example 2 complete!")

    return results_dict


def example_3_abm_vs_representative_agent():
    """
    Example 3: Compare ABM with Representative Agent model.

    Shows what heterogeneity reveals that RA models miss.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: ABM vs REPRESENTATIVE AGENT MODEL")
    print("="*80)

    # Run ABM
    print("\n🔷 Running Agent-Based Model...")
    abm = MacroeconomyABM(n_firms=500, n_households=2000, n_banks=10, random_seed=42)
    abm_results = abm.run(n_periods=200)

    # Run Representative Agent
    print("\n🔶 Running Representative Agent Model...")
    ra_model = RepresentativeAgentModel()
    ra_results = ra_model.simulate(n_periods=200, shock_time=100, shock_size=-0.1)

    # Compare
    print("\n📊 Creating comparison plots...")
    compare_abm_vs_representative(
        abm_results.time_series,
        ra_results,
        save_path='output/abm_vs_ra.png'
    )

    # Analysis
    print("\n" + "="*80)
    print("KEY DIFFERENCES: ABM vs REPRESENTATIVE AGENT")
    print("="*80)

    print("\n1. UNEMPLOYMENT:")
    print(f"   ABM: Average {np.mean(abm_results.time_series['unemployment_rate']):.1%}")
    print(f"   RA:  0.0% (full employment assumed)")

    print("\n2. INEQUALITY:")
    print(f"   ABM: Wealth Gini = {np.mean(abm_results.time_series['gini_wealth']):.3f}")
    print(f"   RA:  N/A (single agent)")

    print("\n3. BUSINESS CYCLES:")
    gdp_volatility_abm = np.std(np.diff(abm_results.time_series['gdp']))
    gdp_volatility_ra = np.std(np.diff(ra_results['gdp']))
    print(f"   ABM: GDP volatility = {gdp_volatility_abm:.2f} (endogenous)")
    print(f"   RA:  GDP volatility = {gdp_volatility_ra:.2f} (only from shocks)")

    print("\n4. FINANCIAL SECTOR:")
    print(f"   ABM: Credit/GDP = {abm_results.time_series['total_credit'][-1] / abm_results.time_series['gdp'][-1]:.1%}")
    print(f"   RA:  Often no banking sector")

    print("\n5. FIRM DYNAMICS:")
    total_bankruptcies = sum(abm_results.time_series['bankruptcies'])
    print(f"   ABM: {total_bankruptcies} bankruptcies (creative destruction)")
    print(f"   RA:  0 (immortal representative firm)")

    print("\n✓ Example 3 complete!")

    return abm_results, ra_results


def example_4_custom_policy_experiment():
    """
    Example 4: Create and test a custom policy experiment.

    Scenario: Implementing a Job Guarantee program
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: CUSTOM POLICY - JOB GUARANTEE")
    print("="*80)

    # Create custom experiment
    job_guarantee = PolicyExperiment("Job Guarantee")

    # At t=50, implement policies:
    # - Increase government spending (hiring unemployed)
    job_guarantee.add_intervention(50, 'government', 'government_spending_target', 180.0)

    # - Higher unemployment benefits (fallback if can't all be hired)
    job_guarantee.add_intervention(50, 'government', 'unemployment_benefit_rate', 0.8)

    # - Accommodative monetary policy
    job_guarantee.add_intervention(50, 'central_bank', 'interest_rate', 0.01)
    job_guarantee.add_intervention(50, 'central_bank', 'use_taylor_rule', False)

    # Run
    print("\n🚀 Simulating Job Guarantee policy...")
    economy = MacroeconomyABM(n_firms=800, n_households=4000, n_banks=10, random_seed=42)
    results = economy.run(n_periods=200, policy_experiment=job_guarantee)

    # Analyze impact
    print("\n" + "="*80)
    print("JOB GUARANTEE RESULTS")
    print("="*80)

    # Before vs after
    unemp_before = np.mean(results.time_series['unemployment_rate'][30:50])
    unemp_after = np.mean(results.time_series['unemployment_rate'][70:90])

    gdp_before = np.mean(results.time_series['gdp'][30:50])
    gdp_after = np.mean(results.time_series['gdp'][70:90])

    gini_before = np.mean(results.time_series['gini_income'][30:50])
    gini_after = np.mean(results.time_series['gini_income'][70:90])

    print(f"\nBefore Job Guarantee (t=30-50):")
    print(f"  Unemployment: {unemp_before:.1%}")
    print(f"  GDP: {gdp_before:.1f}")
    print(f"  Income Gini: {gini_before:.3f}")

    print(f"\nAfter Job Guarantee (t=70-90):")
    print(f"  Unemployment: {unemp_after:.1%}")
    print(f"  GDP: {gdp_after:.1f}")
    print(f"  Income Gini: {gini_after:.3f}")

    print(f"\nChanges:")
    print(f"  Unemployment: {(unemp_after - unemp_before):.1%} ({(unemp_after - unemp_before)/unemp_before*100:.1f}%)")
    print(f"  GDP: {(gdp_after - gdp_before):.1f} ({(gdp_after - gdp_before)/gdp_before*100:.1f}%)")
    print(f"  Inequality: {(gini_after - gini_before):.3f} ({(gini_after - gini_before)/gini_before*100:.1f}%)")

    # Visualize
    viz = ABMVisualizer(economy)
    viz.plot_macro_dashboard(save_path='output/job_guarantee.png')

    print("\n✓ Example 4 complete!")

    return results


def example_5_distributional_dynamics():
    """
    Example 5: Deep dive into distributional dynamics.

    Shows how inequality affects macro outcomes (heterodox insight).
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: DISTRIBUTIONAL DYNAMICS")
    print("="*80)

    # Run simulation
    economy = MacroeconomyABM(n_firms=1000, n_households=5000, n_banks=10, random_seed=42)
    results = economy.run(n_periods=300)

    # Get distributional data
    dist_data = economy.get_distributional_data()

    # Analysis by wealth class
    print("\n" + "="*80)
    print("CONSUMPTION PATTERNS BY WEALTH CLASS")
    print("="*80)

    for wealth_class in ['worker', 'middle', 'wealthy']:
        class_households = [i for i, wc in enumerate(dist_data['wealth_class']) if wc == wealth_class]

        if len(class_households) == 0:
            continue

        avg_wealth = np.mean([dist_data['wealth'][i] for i in class_households])
        avg_income = np.mean([dist_data['income'][i] for i in class_households])
        avg_consumption = np.mean([dist_data['consumption'][i] for i in class_households])

        if avg_income > 0:
            mpc = avg_consumption / avg_income
        else:
            mpc = 0.0

        print(f"\n{wealth_class.upper()} Class:")
        print(f"  Population share: {len(class_households) / len(dist_data['wealth']):.1%}")
        print(f"  Average wealth: {avg_wealth:.2f}")
        print(f"  Average income: {avg_income:.2f}")
        print(f"  Average consumption: {avg_consumption:.2f}")
        print(f"  Implied MPC: {mpc:.2f}")

    # Wealth concentration
    print("\n" + "="*80)
    print("WEALTH CONCENTRATION")
    print("="*80)

    sorted_wealth = np.sort(dist_data['wealth'])
    total_wealth = np.sum(sorted_wealth)

    top_1pct_wealth = np.sum(sorted_wealth[-int(0.01 * len(sorted_wealth)):])
    top_10pct_wealth = np.sum(sorted_wealth[-int(0.10 * len(sorted_wealth)):])
    bottom_50pct_wealth = np.sum(sorted_wealth[:int(0.50 * len(sorted_wealth))])

    print(f"\nTop 1% wealth share: {top_1pct_wealth / total_wealth:.1%}")
    print(f"Top 10% wealth share: {top_10pct_wealth / total_wealth:.1%}")
    print(f"Bottom 50% wealth share: {bottom_50pct_wealth / total_wealth:.1%}")

    # Create detailed distributional plots
    viz = ABMVisualizer(economy)
    viz.plot_distributional_analysis(save_path='output/detailed_distribution.png')

    print("\n✓ Example 5 complete!")

    return dist_data


def run_all_examples():
    """Run all examples in sequence."""
    import os

    # Create output directory
    os.makedirs('output', exist_ok=True)

    print("\n" + "="*80)
    print("AGENT-BASED MACROECONOMIC MODEL")
    print("Comprehensive Examples for Heterodox Economics Research")
    print("="*80)

    # Run all examples
    print("\n🎯 Running all examples...")

    econ1, res1 = example_1_basic_simulation()
    res2 = example_2_policy_experiments()
    abm_res, ra_res = example_3_abm_vs_representative_agent()
    res4 = example_4_custom_policy_experiment()
    dist = example_5_distributional_dynamics()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE!")
    print("="*80)
    print("\n📁 Output files saved to 'output/' directory:")
    print("   - abm_dashboard.png")
    print("   - abm_distribution.png")
    print("   - abm_cycles.png")
    print("   - policy_comparison.png")
    print("   - abm_vs_ra.png")
    print("   - job_guarantee.png")
    print("   - detailed_distribution.png")

    print("\n✅ All visualizations generated!")
    print("\n💡 Next steps:")
    print("   1. Examine the output plots")
    print("   2. Modify parameters and re-run")
    print("   3. Create your own policy experiments")
    print("   4. Use for teaching or research")

    print("\n" + "="*80)


if __name__ == "__main__":
    # Run all examples
    run_all_examples()

    # Or run individual examples:
    # example_1_basic_simulation()
    # example_2_policy_experiments()
    # example_3_abm_vs_representative_agent()
    # example_4_custom_policy_experiment()
    # example_5_distributional_dynamics()
