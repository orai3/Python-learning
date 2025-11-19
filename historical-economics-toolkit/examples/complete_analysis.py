"""
Complete Historical Economic Analysis Example
=============================================

Demonstrates full workflow using the Historical Economics Toolkit:

1. Generate synthetic historical data
2. Detect structural breaks and periodization
3. Analyze long waves (Kondratiev cycles)
4. Identify and analyze crises
5. Examine hegemonic transitions
6. Replicate major studies (Brenner, Arrighi, Duménil-Lévy)
7. Create comprehensive visualizations

This script produces publication-quality analysis of 150 years of capitalist development.
"""

import sys
sys.path.append('../modules')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import toolkit modules
from data_generator import HistoricalEconomicDataGenerator
from periodization import (StructuralBreakDetector, RegimeSwitchingModel,
                           RegulationSchoolPeriodization, detect_all_breaks)
from long_wave_analysis import LongWaveAnalyzer, SchumpeterianCycles, TechnologyRevolutions
from crisis_hegemony import CrisisAnalyzer, HegemonyAnalyzer, analyze_crisis_hegemony_relationship
from visualization import HistoricalPlotter, create_summary_dashboard

# Import replication studies
sys.path.append('../replications')
from major_studies import (BrennerAnalysis, ArrighiAnalysis, DumenilLevyAnalysis,
                           compare_all_frameworks)

import warnings
warnings.filterwarnings('ignore')


def main():
    """Run complete historical economic analysis."""

    print("=" * 70)
    print("HISTORICAL ECONOMIC DATA ANALYSIS TOOLKIT")
    print("Advanced Analysis of Long-Run Capitalist Development")
    print("=" * 70)
    print()

    # ========================================
    # STEP 1: GENERATE HISTORICAL DATA
    # ========================================
    print("STEP 1: Generating Historical Economic Data (1870-2020)")
    print("-" * 70)

    generator = HistoricalEconomicDataGenerator(
        start_year=1870,
        end_year=2020,
        frequency='A',
        seed=42
    )

    countries = ['USA', 'UK', 'Germany', 'France', 'Japan']
    print(f"Countries: {', '.join(countries)}")
    print(f"Time span: 1870-2020 (151 years)")
    print(f"Frequency: Annual")
    print()

    data = generator.generate_complete_dataset(countries=countries)

    print(f"Dataset shape: {data.shape}")
    print(f"Variables: {len(data.columns)}")
    print()

    # Save dataset
    data.to_csv('../data/historical_economic_data.csv', index=False)
    print("Data saved to: ../data/historical_economic_data.csv")
    print()

    # Focus on USA for detailed analysis
    usa_data = data[data['country'] == 'USA'].copy()

    # ========================================
    # STEP 2: STRUCTURAL BREAK DETECTION
    # ========================================
    print("STEP 2: Detecting Structural Breaks")
    print("-" * 70)

    detector = StructuralBreakDetector(usa_data)

    # Bai-Perron test on GDP growth
    print("Running Bai-Perron test on GDP growth...")
    bp_results = detector.bai_perron_test('gdp_growth', max_breaks=5)

    print(f"Detected breaks: {bp_results['break_years']}")
    print(f"Number of breaks: {bp_results['n_breaks']}")
    print()

    # CUSUM test
    print("Running CUSUM test on profit rate...")
    cusum_results = detector.cusum_test('profit_rate')
    print(f"CUSUM detected breaks: {cusum_results['break_years']}")
    print()

    # Test known historical breaks
    print("Testing known historical breaks (Chow tests):")
    historical_breaks = [1914, 1929, 1945, 1973, 2008]

    for year in historical_breaks:
        chow = detector.chow_test('gdp_growth', year)
        sig = "SIGNIFICANT" if chow.get('significant', False) else "Not significant"
        print(f"  {year}: {sig} (p-value: {chow.get('p_value', 'N/A')})")
    print()

    # ========================================
    # STEP 3: REGULATION SCHOOL PERIODIZATION
    # ========================================
    print("STEP 3: Regulation School Periodization")
    print("-" * 70)

    reg_school = RegulationSchoolPeriodization(usa_data)

    regime_results = reg_school.identify_regimes(
        variables=['wage_share', 'financialization', 'institutional_coordination', 'labor_militancy'],
        n_regimes=4
    )

    print("Identified regimes:")
    regime_chars = regime_results['regime_characteristics']
    regime_chars_labeled = reg_school.label_historical_regimes(regime_chars)

    for _, regime in regime_chars_labeled.iterrows():
        print(f"\n{regime['regime_label']}:")
        print(f"  Period: {regime['start_year']}-{regime['end_year']} ({regime['duration']} years)")
        print(f"  Wage share: {regime.get('wage_share_mean', 0):.3f}")
        print(f"  Financialization: {regime.get('financialization_mean', 0):.3f}")
    print()

    # ========================================
    # STEP 4: KONDRATIEV LONG WAVE ANALYSIS
    # ========================================
    print("STEP 4: Kondratiev Long Wave Analysis")
    print("-" * 70)

    wave_analyzer = LongWaveAnalyzer(usa_data)

    # Spectral analysis
    print("Performing spectral analysis on GDP...")
    spectral = wave_analyzer.spectral_analysis('gdp')

    if spectral['dominant_periods']:
        print("Dominant periods detected:")
        for i, period in enumerate(spectral['dominant_periods'][:3]):
            print(f"  {i+1}. {period['period']:.1f} years (power: {period['power']:.2e})")
    print()

    # Identify Kondratiev waves
    print("Identifying Kondratiev waves...")
    wave_results = wave_analyzer.identify_kondratiev_waves('gdp')

    print(f"Number of complete waves detected: {wave_results['n_waves']}")
    if wave_results['average_period']:
        print(f"Average period: {wave_results['average_period']:.1f} years")
        print(f"Average expansion: {wave_results['average_expansion_duration']:.1f} years")
        print(f"Average contraction: {wave_results['average_contraction_duration']:.1f} years")

    if wave_results['waves']:
        print("\nDetected waves:")
        for wave in wave_results['waves']:
            print(f"  Wave {wave['wave_id']}: {wave['trough1_year']}-{wave['peak_year']}-{wave['trough2_year']}")
    print()

    # Schumpeterian decomposition
    print("Schumpeterian multi-frequency cycle decomposition...")
    schump = SchumpeterianCycles(usa_data)
    cycles = schump.extract_all_cycles('gdp')
    print("  Decomposed into: Trend + Kitchin (3-4yr) + Juglar (7-11yr) + Kondratiev (45-65yr)")
    print()

    # ========================================
    # STEP 5: CRISIS ANALYSIS
    # ========================================
    print("STEP 5: Economic Crisis Analysis")
    print("-" * 70)

    crisis_analyzer = CrisisAnalyzer(usa_data)

    # Detect crises
    print("Detecting economic crises...")
    crises = crisis_analyzer.detect_crises(variable='gdp_growth', threshold=-0.02)

    print(f"Total crises detected: {len(crises)}")
    print("\nMajor crises:")
    for _, crisis in crises.iterrows():
        print(f"  {crisis['start_year']}-{crisis['end_year']}: "
              f"Duration={crisis['duration']}yr, Severity={crisis['severity']:.3f}")
    print()

    # Crisis clustering
    print("Analyzing crisis clustering...")
    clustering = crisis_analyzer.analyze_crisis_clustering(crises)
    print(f"Coefficient of variation: {clustering['coefficient_of_variation']:.3f}")
    print(f"Interpretation: {clustering['clustering_interpretation']}")
    print()

    # Crisis frequency by period
    print("Crisis frequency by historical period:")
    freq_by_period = crisis_analyzer.crisis_frequency_by_period(crises)

    for _, period in freq_by_period.iterrows():
        print(f"  {period['period']} ({period['start_year']}-{period['end_year']}): "
              f"{period['n_crises']} crises, frequency={period['frequency']:.3f}")
    print()

    # Systemic crises
    systemic = crisis_analyzer.identify_systemic_crises(crises)
    print(f"Systemic crises (severe/structural): {len(systemic)}")
    for sc in systemic:
        print(f"  {sc['start_year']}: {sc['systemic_reason']}")
    print()

    # ========================================
    # STEP 6: HEGEMONIC CYCLE ANALYSIS
    # ========================================
    print("STEP 6: Hegemonic Cycle Analysis (Arrighi Framework)")
    print("-" * 70)

    hegemony_analyzer = HegemonyAnalyzer(usa_data)

    # Detect hegemonic transitions
    print("Detecting hegemonic transitions...")
    transitions = hegemony_analyzer.detect_hegemonic_transitions()

    print(f"Hegemonic transitions detected: {len(transitions)}")
    for trans in transitions:
        print(f"  {trans['start_year']}-{trans['end_year']}: "
              f"Hegemony declined from {trans['hegemony_start']:.3f} to {trans['hegemony_end']:.3f}")
    print()

    # Accumulation phases
    print("Classifying material vs financial expansion phases...")
    phases = hegemony_analyzer.classify_accumulation_phase()

    phase_summary = phases.groupby('accumulation_phase').size()
    print(f"Material expansion years: {phase_summary.get('Material Expansion', 0)}")
    print(f"Financial expansion years: {phase_summary.get('Financial Expansion', 0)}")
    print()

    # Crisis-hegemony relationship
    print("Testing crisis-hegemony relationship...")
    crisis_heg = analyze_crisis_hegemony_relationship(usa_data)

    print(f"Crises in material expansion: {crisis_heg['crises_in_material_expansion']}")
    print(f"Crises in financial expansion: {crisis_heg['crises_in_financial_expansion']}")
    print(f"Crisis rate (material): {crisis_heg['crisis_rate_material']:.4f}")
    print(f"Crisis rate (financial): {crisis_heg['crisis_rate_financial']:.4f}")

    if crisis_heg['ratio_financial_to_material']:
        print(f"Ratio: {crisis_heg['ratio_financial_to_material']:.2f}x more crises in financial phase")
    print()

    # ========================================
    # STEP 7: REPLICATION STUDIES
    # ========================================
    print("STEP 7: Replicating Major Heterodox Studies")
    print("-" * 70)

    # BRENNER
    print("\n[A] ROBERT BRENNER - Long Downturn Thesis")
    print("-" * 50)

    brenner = BrennerAnalysis(usa_data)

    profit_trends = brenner.calculate_profit_rate_trend()
    print(profit_trends['interpretation'])
    print()

    # ARRIGHI
    print("\n[B] GIOVANNI ARRIGHI - Systemic Cycles of Accumulation")
    print("-" * 50)

    arrighi = ArrighiAnalysis(usa_data)

    us_cycle = arrighi.analyze_us_cycle()
    print(f"US Hegemonic Cycle:")
    print(f"  Material expansion: {us_cycle['material_expansion']}")
    print(f"  Financial expansion: {us_cycle['financial_expansion']}")
    print(f"  Financialization (material phase): {us_cycle['material_avg_financialization']:.3f}")
    print(f"  Financialization (financial phase): {us_cycle['financial_avg_financialization']:.3f}")
    print(f"\n{us_cycle['interpretation']}")
    print()

    # DUMÉNIL-LÉVY
    print("\n[C] DUMÉNIL & LÉVY - Class Power and Neoliberalism")
    print("-" * 50)

    dumenil_levy = DumenilLevyAnalysis(usa_data)

    neoliberal = dumenil_levy.analyze_neoliberal_restoration()
    print(neoliberal['interpretation'])
    print()

    # COMPARATIVE FRAMEWORK ANALYSIS
    print("\n[D] Comparative Framework Analysis")
    print("-" * 50)

    comparison = compare_all_frameworks(usa_data)
    print(comparison['synthesis'])
    print()

    # ========================================
    # STEP 8: VISUALIZATION
    # ========================================
    print("STEP 8: Creating Visualizations")
    print("-" * 70)

    plotter = HistoricalPlotter(usa_data)

    # Convert regime characteristics to list of dicts for plotting
    regime_periods = regime_chars_labeled.to_dict('records')

    print("Generating plots...")

    # 1. Long-run trends
    print("  1. Long-run trends with regime shading...")
    fig1 = plotter.plot_long_run_trends(
        variables=['gdp_growth', 'wage_share', 'profit_rate'],
        regime_periods=regime_periods,
        title='Long-Run Economic Trends: USA (1870-2020)'
    )
    fig1.savefig('../docs/long_run_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Crisis timeline
    print("  2. Crisis timeline...")
    fig2 = plotter.plot_crisis_timeline(crises, variable='gdp_growth')
    fig2.savefig('../docs/crisis_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Kondratiev decomposition
    print("  3. Kondratiev wave decomposition...")
    fig3 = plotter.plot_kondratiev_decomposition(
        'gdp',
        wave_results['long_wave_series'],
        wave_results['waves']
    )
    fig3.savefig('../docs/kondratiev_waves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Hegemonic cycle
    print("  4. Hegemonic cycle...")
    fig4 = plotter.plot_hegemonic_cycle(transitions=transitions)
    fig4.savefig('../docs/hegemonic_cycle.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Distribution dynamics
    print("  5. Distribution dynamics...")
    fig5 = plotter.plot_distribution_dynamics(
        wage_share=True,
        inequality=True,
        regimes=regime_periods
    )
    fig5.savefig('../docs/distribution_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Profit squeeze
    print("  6. Profit squeeze dynamics...")
    fig6 = plotter.plot_profit_squeeze()
    fig6.savefig('../docs/profit_squeeze.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Summary dashboard
    print("  7. Summary dashboard...")
    fig7 = create_summary_dashboard(usa_data, crises, regime_periods)
    fig7.savefig('../docs/summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Comparative countries
    print("  8. Comparative country analysis...")
    fig8 = plotter.plot_comparative_countries(
        'gdp',
        countries=countries,
        normalize=True
    )
    fig8.savefig('../docs/comparative_countries.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nAll visualizations saved to ../docs/")
    print()

    # ========================================
    # STEP 9: EXPORT RESULTS
    # ========================================
    print("STEP 9: Exporting Results")
    print("-" * 70)

    # Export key results to CSV
    regime_chars_labeled.to_csv('../data/regime_periods.csv', index=False)
    print("Regime periods saved to: ../data/regime_periods.csv")

    crises.to_csv('../data/detected_crises.csv', index=False)
    print("Crises saved to: ../data/detected_crises.csv")

    profit_trends['periods'].to_csv('../data/brenner_profit_trends.csv', index=False)
    print("Brenner profit trends saved to: ../data/brenner_profit_trends.csv")

    # Create summary report
    print("\nCreating summary report...")

    summary = {
        'analysis_period': '1870-2020',
        'n_years': len(usa_data),
        'n_regimes': len(regime_chars_labeled),
        'n_crises': len(crises),
        'n_systemic_crises': len(systemic),
        'n_kondratiev_waves': wave_results['n_waves'],
        'avg_wave_period': wave_results['average_period'],
        'n_hegemonic_transitions': len(transitions)
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv('../data/analysis_summary.csv', index=False)
    print("Summary saved to: ../data/analysis_summary.csv")
    print()

    # ========================================
    # CONCLUSION
    # ========================================
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print("Summary of findings:")
    print(f"  - Analyzed {len(usa_data)} years of economic history (1870-2020)")
    print(f"  - Identified {len(regime_chars_labeled)} distinct regimes of accumulation")
    print(f"  - Detected {len(crises)} economic crises ({len(systemic)} systemic)")
    print(f"  - Found {wave_results['n_waves']} complete Kondratiev waves")
    print(f"  - Identified {len(transitions)} hegemonic transition periods")
    print()
    print("All results, data, and visualizations saved to:")
    print("  - Data: ../data/")
    print("  - Visualizations: ../docs/")
    print()
    print("Next steps:")
    print("  1. Review visualizations in ../docs/")
    print("  2. Examine detailed results in ../data/")
    print("  3. Adapt analysis for your research questions")
    print("  4. Replace synthetic data with real historical data")
    print("  5. Extend toolkit with additional methods")
    print()
    print("For questions or contributions, see README.md")
    print("=" * 70)


if __name__ == '__main__':
    main()
