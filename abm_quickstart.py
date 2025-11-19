"""
Quick Start: Agent-Based Macroeconomic Model

Minimal example to see the ABM in action.
For full examples, see abm_example_comprehensive.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from abm_macro import MacroeconomyABM, ABMVisualizer

# Create output directory
os.makedirs('output', exist_ok=True)

print("\n" + "="*70)
print("AGENT-BASED MACROECONOMIC MODEL - QUICK START")
print("="*70)

# Create economy with heterogeneous agents
print("\n🏗️  Creating economy with:")
print("   • 1000 heterogeneous firms")
print("   • 5000 heterogeneous households")
print("   • 10 banks")

economy = MacroeconomyABM(
    n_firms=1000,
    n_households=5000,
    n_banks=10,
    random_seed=42
)

# Run simulation
print("\n🚀 Running simulation for 300 periods...")
results = economy.run(n_periods=300)

# Create visualizations
print("\n📊 Creating visualizations...")

viz = ABMVisualizer(economy)

# Macro dashboard
print("   1. Macro dashboard...")
fig1 = viz.plot_macro_dashboard(save_path='output/quickstart_dashboard.png')

# Distributional analysis
print("   2. Distributional analysis...")
fig2 = viz.plot_distributional_analysis(save_path='output/quickstart_distribution.png')

# Business cycles
print("   3. Business cycle analysis...")
fig3 = viz.plot_business_cycle_analysis(save_path='output/quickstart_cycles.png')

# Print summary statistics
print("\n" + "="*70)
print("SIMULATION RESULTS SUMMARY")
print("="*70)

print(f"\nMacro Aggregates:")
print(f"  Final GDP: {results.final_state.gdp:.1f}")
print(f"  Final Consumption: {results.final_state.consumption:.1f}")
print(f"  Final Investment: {results.final_state.investment:.1f}")
print(f"  Average GDP growth: {np.mean(np.diff(results.time_series['gdp'])):.2f} per period")

print(f"\nLabor Market:")
print(f"  Final unemployment rate: {results.final_state.unemployment_rate:.1%}")
print(f"  Average wage: {results.final_state.average_wage:.2f}")
print(f"  Employment: {results.final_state.employment} / {results.final_state.labor_force}")

print(f"\nDistribution:")
print(f"  Wealth Gini: {results.final_state.gini_wealth:.3f}")
print(f"  Income Gini: {results.final_state.gini_income:.3f}")
print(f"  Wage share of GDP: {results.final_state.wage_share:.1%}")
print(f"  Profit share of GDP: {results.final_state.profit_share:.1%}")

print(f"\nFinancial Sector:")
print(f"  Total credit: {results.final_state.total_credit:.1f}")
print(f"  Debt-to-GDP ratio: {results.final_state.debt_to_gdp:.1%}")
print(f"  Total bankruptcies: {sum(results.time_series['bankruptcies'])}")

print(f"\nGovernment:")
print(f"  Public debt: {results.time_series['public_debt'][-1]:.1f}")
print(f"  Budget deficit: {results.time_series['government_deficit'][-1]:.1f}")

print(f"\nMonetary Policy:")
print(f"  Policy interest rate: {results.final_state.interest_rate:.2%}")
print(f"  Inflation rate: {results.final_state.inflation:.2%}")

# Show emergent dynamics
print("\n" + "="*70)
print("EMERGENT DYNAMICS (What Representative Agent Models Miss)")
print("="*70)

print(f"\n✓ Business Cycles:")
gdp_volatility = np.std(np.diff(results.time_series['gdp']))
print(f"  GDP volatility: {gdp_volatility:.2f}")
print(f"  → Endogenous cycles emerge from micro interactions!")

print(f"\n✓ Involuntary Unemployment:")
avg_unemployment = np.mean(results.time_series['unemployment_rate'])
print(f"  Average unemployment: {avg_unemployment:.1%}")
print(f"  → Not assumed away - emerges from rationing!")

print(f"\n✓ Wealth Inequality:")
dist_data = economy.get_distributional_data()
sorted_wealth = np.sort(dist_data['wealth'])
top_10pct = np.sum(sorted_wealth[-int(0.1*len(sorted_wealth)):]) / np.sum(sorted_wealth)
print(f"  Top 10% wealth share: {top_10pct:.1%}")
print(f"  → Inequality affects aggregate demand (different MPCs)!")

print(f"\n✓ Financial Fragility:")
max_bankruptcies = max(results.time_series['bankruptcies'])
print(f"  Peak bankruptcies in one period: {max_bankruptcies}")
print(f"  → Minskyan instability - stability breeds instability!")

print(f"\n✓ Credit Cycles:")
credit_volatility = np.std(np.diff(results.time_series['total_credit']))
avg_rationing = np.mean(results.time_series['credit_rationing_rate'])
print(f"  Average credit rationing: {avg_rationing:.1%}")
print(f"  → Pro-cyclical credit amplifies fluctuations!")

# Output file locations
print("\n" + "="*70)
print("OUTPUT FILES")
print("="*70)
print("\n📁 Visualizations saved to output/ directory:")
print("   • quickstart_dashboard.png - Full macro dashboard")
print("   • quickstart_distribution.png - Inequality & distribution")
print("   • quickstart_cycles.png - Business cycle dynamics")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
1. Open the PNG files to see the visualizations

2. Run full examples:
   python abm_example_comprehensive.py

3. Try policy experiments:
   from abm_macro import PolicyExperiment
   experiment = PolicyExperiment.create_austerity_experiment()
   results = economy.run(n_periods=300, policy_experiment=experiment)

4. Compare with Representative Agent model:
   from abm_macro import RepresentativeAgentModel
   ra_model = RepresentativeAgentModel()
   ra_results = ra_model.simulate(n_periods=300)

5. Modify parameters and explore!

Read ABM_README.md for comprehensive documentation.
""")

print("="*70)
print("✅ Quick start complete!")
print("="*70 + "\n")

# Display plots
plt.show()
