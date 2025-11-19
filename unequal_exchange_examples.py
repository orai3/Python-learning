"""
Example Scripts for Unequal Exchange Framework

Demonstrates key functionality with worked examples.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def example_1_basic_emmanuel_model():
    """Example 1: Basic Emmanuel unequal exchange calculation"""
    print("=" * 70)
    print("EXAMPLE 1: Emmanuel Unequal Exchange Model")
    print("=" * 70)

    from unequal_exchange.models.emmanuel import EmmanuelModel, EmmanuelParameters
    from unequal_exchange.core.theoretical_base import ProductionData, CountryCategory

    # Initialize model
    params = EmmanuelParameters(global_profit_rate=0.15)
    model = EmmanuelModel(parameters=params)

    # Add USA (core country)
    usa_data = ProductionData(
        gross_output=1000,
        labor_hours=100,
        wage_rate=50,  # $50/hour
        capital_stock=500,
        intermediate_inputs=400
    )
    model.add_country('USA', CountryCategory.CORE, usa_data)
    print(f"\n📍 USA (Core):")
    print(f"   Wage rate: ${usa_data.wage_rate}/hour")
    print(f"   Labor productivity: ${usa_data.labor_productivity:.2f}/hour")
    print(f"   Rate of exploitation: {usa_data.rate_of_exploitation*100:.1f}%")

    # Add Bangladesh (periphery country)
    bangladesh_data = ProductionData(
        gross_output=100,
        labor_hours=120,
        wage_rate=5,  # $5/hour - 10x difference!
        capital_stock=50,
        intermediate_inputs=40
    )
    model.add_country('Bangladesh', CountryCategory.PERIPHERY, bangladesh_data)
    print(f"\n📍 Bangladesh (Periphery):")
    print(f"   Wage rate: ${bangladesh_data.wage_rate}/hour")
    print(f"   Labor productivity: ${bangladesh_data.labor_productivity:.2f}/hour")
    print(f"   Rate of exploitation: {bangladesh_data.rate_of_exploitation*100:.1f}%")

    # Set trade flows
    trade_matrix = pd.DataFrame({
        'USA': [0, 50],
        'Bangladesh': [80, 0]
    }, index=['USA', 'Bangladesh'])
    model.set_trade_flows(trade_matrix)

    print(f"\n💱 Trade Flows:")
    print(f"   Bangladesh → USA: $50M")
    print(f"   USA → Bangladesh: $80M")

    # Calculate value transfers
    transfers = model.calculate_value_transfers()

    print(f"\n💸 Value Transfers (Emmanuel Mechanism):")
    for _, row in transfers.iterrows():
        print(f"   {row['exporter']} → {row['importer']}: ${row['value_transfer']:.2f}M")
        print(f"      (Transfer = {row['transfer_pct_of_trade']:.1f}% of trade value)")

    # Summary statistics
    stats = model.get_summary_statistics()
    print(f"\n📊 Summary:")
    print(f"   Core net gain: ${stats['core_net_gain']:.2f}M")
    print(f"   Periphery net loss: ${stats['periphery_net_loss']:.2f}M")
    print(f"   Average transfer as % of trade: {stats['avg_transfer_pct']:.1f}%")

    print("\n" + "="*70 + "\n")


def example_2_gvc_analysis():
    """Example 2: Global Value Chain Rent Extraction (iPhone Example)"""
    print("=" * 70)
    print("EXAMPLE 2: Global Value Chain Analysis - iPhone")
    print("=" * 70)

    from unequal_exchange.analysis.gvc_rents import (
        GVCRentExtractor, ValueChainSegment
    )

    # Create analyzer
    analyzer = GVCRentExtractor()

    # Define iPhone value chain (stylized data based on estimates)
    iphone_chain = [
        ValueChainSegment(
            name="R&D/Design",
            country="USA",
            value_added=300,  # $300 per unit
            labor_cost=50,
            capital_intensity=0.8,
            market_power=0.9,
            barriers_to_entry=0.9,
            ip_intensity=0.95
        ),
        ValueChainSegment(
            name="Component Manufacturing",
            country="SouthKorea",
            value_added=120,
            labor_cost=60,
            capital_intensity=0.7,
            market_power=0.5,
            barriers_to_entry=0.6,
            ip_intensity=0.3
        ),
        ValueChainSegment(
            name="Assembly",
            country="China",
            value_added=50,
            labor_cost=40,
            capital_intensity=0.6,
            market_power=0.2,
            barriers_to_entry=0.3,
            ip_intensity=0.1
        ),
        ValueChainSegment(
            name="Marketing/Brand",
            country="USA",
            value_added=200,
            labor_cost=30,
            capital_intensity=0.5,
            market_power=0.95,
            barriers_to_entry=0.9,
            ip_intensity=0.8
        ),
        ValueChainSegment(
            name="Retail",
            country="Global",
            value_added=130,
            labor_cost=40,
            capital_intensity=0.4,
            market_power=0.7,
            barriers_to_entry=0.5,
            ip_intensity=0.4
        )
    ]

    analyzer.add_value_chain("iPhone", iphone_chain)

    # Analyze value distribution
    print("\n💰 Value Distribution Across Chain:")
    distribution = analyzer.calculate_value_distribution("iPhone")
    for _, row in distribution.iterrows():
        print(f"\n   {row['segment']} ({row['country']}):")
        print(f"      Value added: ${row['value_added']:.0f} ({row['value_share_pct']:.1f}%)")
        print(f"      Profit rate: {row['profit_rate']:.1f}%")
        print(f"      Total rents: ${row['total_rent']:.2f}")
        print(f"      - Monopoly rent: ${row['monopoly_rent']:.2f}")
        print(f"      - IP rent: ${row['ip_rent']:.2f}")

    # Analyze smile curve
    print("\n😊 Smile Curve Analysis:")
    smile = analyzer.analyze_smile_curve("iPhone")
    print(f"   Upstream value share: {smile['upstream_value_share']:.1f}%")
    print(f"   Midstream value share: {smile['midstream_value_share']:.1f}%")
    print(f"   Downstream value share: {smile['downstream_value_share']:.1f}%")
    print(f"   Smile intensity: {smile['smile_intensity']:.2f}")
    print(f"   → {smile['interpretation']}")

    # Lead firm extraction
    print("\n🏢 Lead Firm (Apple) Extraction:")
    lead_firm = analyzer.calculate_lead_firm_extraction(
        "iPhone",
        ["R&D/Design", "Marketing/Brand"]
    )
    print(f"   Apple's value share: {lead_firm['lead_firm_value_share']:.1f}%")
    print(f"   Supplier value share: {lead_firm['supplier_value_share']:.1f}%")
    print(f"   Apple profit rate: {lead_firm['lead_firm_profit_rate']:.1f}%")
    print(f"   Supplier profit rate: {lead_firm['supplier_profit_rate']:.1f}%")
    print(f"   Extraction ratio: {lead_firm['extraction_ratio']:.2f}x")

    print("\n" + "="*70 + "\n")


def example_3_policy_simulation():
    """Example 3: Policy Simulation - South-South Cooperation"""
    print("=" * 70)
    print("EXAMPLE 3: Policy Simulation - South-South Cooperation")
    print("=" * 70)

    from unequal_exchange.policy.simulations import PolicySimulator

    # Initialize simulator
    simulator = PolicySimulator()

    # Simulate South-South cooperation
    south_countries = ['Brazil', 'India', 'Nigeria', 'Bangladesh']
    results = simulator.simulate_south_south_cooperation(
        south_countries=south_countries,
        cooperation_intensity=0.7,  # 70% cooperation
        years=20
    )

    print("\n🤝 South-South Cooperation Scenario")
    print(f"   Countries: {', '.join(south_countries)}")
    print(f"   Cooperation intensity: 70%")
    print(f"   Time horizon: 20 years")

    # Results at year 20
    final_year = results[results['scenario'] == 'south_south_cooperation'].iloc[-1]
    baseline = results[results['scenario'] == 'baseline'].iloc[0]

    print(f"\n📊 Results After 20 Years:")
    print(f"   Intra-South trade share: {final_year['intra_south_trade_share']*100:.1f}%")
    print(f"   Average productivity gain: {((final_year['avg_productivity']/baseline['avg_productivity'])-1)*100:.1f}%")
    print(f"   Value transfer reduction: ${final_year['transfer_reduction_from_baseline']:.2f}M")
    print(f"   Cumulative transfer savings: ${final_year['cumulative_transfer_reduction']:.2f}M")
    print(f"   Terms of trade improvement: +{final_year['tot_improvement']:.2f}%")

    print("\n💡 Interpretation:")
    print("   Through increased intra-South trade and technology cooperation,")
    print("   peripheral countries can significantly reduce value transfers")
    print("   to core countries while improving productivity and living standards.")

    print("\n" + "="*70 + "\n")


def example_4_data_generation_and_viz():
    """Example 4: Generate Synthetic Data and Visualize"""
    print("=" * 70)
    print("EXAMPLE 4: Synthetic Data Generation and Visualization")
    print("=" * 70)

    from unequal_exchange.data.synthetic_generator import SyntheticDataGenerator

    # Generate data
    print("\n📦 Generating synthetic datasets (1960-2020)...")
    generator = SyntheticDataGenerator(start_year=1960, end_year=2020)
    datasets = generator.generate_complete_dataset('./unequal_exchange_data/')

    # Show summary
    print(f"\n✓ Generated {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"   - {name}: {len(df)} rows, {len(df.columns)} columns")

    # Quick analysis: Terms of trade deterioration
    tot_data = datasets['terms_of_trade']
    print(f"\n📉 Prebisch-Singer Analysis:")
    print(f"   ToT in 1960: {tot_data.iloc[0]['tot_index']:.2f}")
    print(f"   ToT in 2020: {tot_data.iloc[-1]['tot_index']:.2f}")
    print(f"   Total deterioration: {tot_data.iloc[-1]['tot_vs_base']:.2f}%")

    # Wage data analysis
    wage_data = datasets['wages']
    wage_2020 = wage_data[wage_data['year'] == 2020]

    print(f"\n💵 Wage Differentials in 2020:")
    for _, row in wage_2020.iterrows():
        print(f"   {row['country']:15} ({row['category']:14}): Wage={row['wage_index']:6.1f}, Productivity={row['productivity_index']:6.1f}")

    # Create visualization
    print(f"\n📈 Creating visualizations...")
    from unequal_exchange.visualization.core_periphery_plots import CorePeripheryVisualizer

    viz = CorePeripheryVisualizer(style='academic')

    # Plot terms of trade
    fig = viz.plot_terms_of_trade(tot_data, figsize=(12, 6))
    plt.savefig('./unequal_exchange_data/terms_of_trade.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: terms_of_trade.png")

    print("\n" + "="*70 + "\n")


def example_5_complete_analysis():
    """Example 5: Complete Analysis Workflow"""
    print("=" * 70)
    print("EXAMPLE 5: Complete Analysis Workflow")
    print("=" * 70)
    print("\nThis example demonstrates a complete research workflow:")
    print("1. Generate data")
    print("2. Run unequal exchange models")
    print("3. Calculate transfer totals")
    print("4. Simulate policy alternatives")
    print("5. Visualize results")
    print("\n(See README.md for detailed implementation)")

    print("\n" + "="*70 + "\n")


def main():
    """Run all examples"""
    print("\n" + "🌍 UNEQUAL EXCHANGE FRAMEWORK - EXAMPLE SCRIPTS 🌍\n")

    # Run examples
    example_1_basic_emmanuel_model()
    example_2_gvc_analysis()
    example_3_policy_simulation()
    example_4_data_generation_and_viz()
    example_5_complete_analysis()

    print("\n✅ All examples completed successfully!")
    print("\nNext steps:")
    print("  - Explore the data/ directory for generated datasets")
    print("  - See README.md for comprehensive documentation")
    print("  - Run the PyQt6 GUI: python -m unequal_exchange.gui.main_application")
    print("  - Adapt examples for your own research questions\n")


if __name__ == '__main__':
    main()
