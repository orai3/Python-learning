"""
Quick test script to verify toolkit functionality
"""

import sys
sys.path.append('modules')

import pandas as pd
from data_generator import HistoricalEconomicDataGenerator
from periodization import StructuralBreakDetector
from long_wave_analysis import LongWaveAnalyzer
from crisis_hegemony import CrisisAnalyzer

print("Testing Historical Economics Toolkit")
print("=" * 60)

# 1. Generate data
print("\n1. Generating sample data (1870-2020, USA)...")
generator = HistoricalEconomicDataGenerator(
    start_year=1870,
    end_year=2020,
    frequency='A',
    seed=42
)

data = generator.generate_complete_dataset(countries=['USA'])
print(f"   Generated {len(data)} observations")
print(f"   Variables: {len(data.columns)}")

# 2. Detect breaks
print("\n2. Detecting structural breaks...")
detector = StructuralBreakDetector(data)
breaks = detector.bai_perron_test('gdp_growth', max_breaks=5)
print(f"   Detected {len(breaks['break_years'])} breaks: {breaks['break_years']}")

# 3. Analyze long waves
print("\n3. Analyzing Kondratiev waves...")
wave_analyzer = LongWaveAnalyzer(data)
waves = wave_analyzer.identify_kondratiev_waves('gdp')
print(f"   Found {waves['n_waves']} complete waves")
if waves['average_period']:
    print(f"   Average period: {waves['average_period']:.1f} years")

# 4. Detect crises
print("\n4. Detecting economic crises...")
crisis_analyzer = CrisisAnalyzer(data)
crises = crisis_analyzer.detect_crises(threshold=-0.02)
print(f"   Detected {len(crises)} crises")
if len(crises) > 0:
    print(f"   First crisis: {crises.iloc[0]['start_year']}")
    print(f"   Most severe: {crises['severity'].max():.3f}")

# 5. Save sample
print("\n5. Saving sample data...")
sample = data.head(20)
sample.to_csv('data/sample_data.csv', index=False)
print("   Saved to: data/sample_data.csv")

print("\n" + "=" * 60)
print("Toolkit test completed successfully!")
print("=" * 60)
