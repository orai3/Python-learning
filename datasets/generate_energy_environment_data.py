"""
Synthetic Energy & Environmental Economics Dataset
===================================================
Generates comprehensive data on energy systems and environmental economics.

Includes:
- Energy consumption by sector (industry, transport, residential, commercial)
- Energy sources (fossil fuels, renewables, nuclear)
- Emissions and environmental indicators
- Energy prices and costs
- Green transition dynamics
- Regional disparities

Annual data 1970-2024, with heterodox focus on:
- Energy-GDP relationship (decoupling)
- Just transition and distributional effects
- Political economy of energy systems
"""

import numpy as np
import pandas as pd

np.random.seed(48)

# Configuration
START_YEAR = 1970
END_YEAR = 2024
YEARS = END_YEAR - START_YEAR + 1

years = np.arange(START_YEAR, END_YEAR + 1)

# ============================================================================
# GDP (baseline for energy demand)
# ============================================================================
gdp_growth = 0.025 + np.random.normal(0, 0.01, YEARS)
gdp = np.zeros(YEARS)
gdp[0] = 100
for t in range(1, YEARS):
    gdp[t] = gdp[t-1] * (1 + gdp_growth[t])

# ============================================================================
# TOTAL ENERGY CONSUMPTION
# ============================================================================
# Energy intensity (energy/GDP) declining over time (efficiency improvements)
energy_intensity_trend = 1.0 - 0.4 * (1 - np.exp(-np.arange(YEARS) / 30))
energy_intensity = energy_intensity_trend + np.random.normal(0, 0.02, YEARS)

# Total energy consumption (quadrillion BTU or similar units)
total_energy = gdp * energy_intensity

# ============================================================================
# ENERGY BY SECTOR
# ============================================================================
# Shares evolving over time

# Industrial (declining share as economy de-industrializes)
industrial_share = 40 - 8 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 1, YEARS)
industrial_share = np.clip(industrial_share, 25, 42)

# Transport (stable to slightly rising)
transport_share = 28 + 2 * (1 - np.exp(-np.arange(YEARS) / 35)) + np.random.normal(0, 1, YEARS)
transport_share = np.clip(transport_share, 25, 35)

# Residential (declining share due to efficiency)
residential_share = 22 - 4 * (1 - np.exp(-np.arange(YEARS) / 30)) + np.random.normal(0, 1, YEARS)
residential_share = np.clip(residential_share, 15, 24)

# Commercial (rising with service economy)
commercial_share = 100 - industrial_share - transport_share - residential_share

# Energy consumption by sector
industrial_energy = total_energy * (industrial_share / 100)
transport_energy = total_energy * (transport_share / 100)
residential_energy = total_energy * (residential_share / 100)
commercial_energy = total_energy * (commercial_share / 100)

# ============================================================================
# ENERGY BY SOURCE
# ============================================================================

# Coal (declining from high base)
coal_share = 35 - 25 * (1 - np.exp(-np.arange(YEARS) / 20)) + np.random.normal(0, 1, YEARS)
coal_share = np.clip(coal_share, 8, 40)

# Oil (declining slowly)
oil_share = 40 - 10 * (1 - np.exp(-np.arange(YEARS) / 30)) + np.random.normal(0, 1, YEARS)
oil_share = np.clip(oil_share, 28, 45)

# Natural gas (rising then stabilizing)
gas_share = 15 + 10 * (1 - np.exp(-np.arange(YEARS) / 25)) + np.random.normal(0, 1, YEARS)
gas_share = np.clip(gas_share, 12, 30)

# Nuclear (rising then declining)
nuclear_peak = 30  # Peak around year 2000
nuclear_share = 5 + 10 * np.exp(-((np.arange(YEARS) - nuclear_peak) ** 2) / 300) + \
               np.random.normal(0, 0.5, YEARS)
nuclear_share = np.clip(nuclear_share, 3, 18)

# Renewables (low base, rapid recent growth)
# Accelerating after 2000
renewables_acceleration = np.maximum(0, np.arange(YEARS) - 30)  # After 2000
renewables_share = 5 + 15 * (1 - np.exp(-renewables_acceleration / 12)) + np.random.normal(0, 0.5, YEARS)
renewables_share = np.clip(renewables_share, 3, 35)

# Normalize shares to sum to 100
total_share = coal_share + oil_share + gas_share + nuclear_share + renewables_share
coal_share = coal_share / total_share * 100
oil_share = oil_share / total_share * 100
gas_share = gas_share / total_share * 100
nuclear_share = nuclear_share / total_share * 100
renewables_share = renewables_share / total_share * 100

# Energy consumption by source
coal_energy = total_energy * (coal_share / 100)
oil_energy = total_energy * (oil_share / 100)
gas_energy = total_energy * (gas_share / 100)
nuclear_energy = total_energy * (nuclear_share / 100)
renewables_energy = total_energy * (renewables_share / 100)

# Breakdown of renewables
hydro_share_of_renewables = 70 - 40 * (1 - np.exp(-renewables_acceleration / 10))
hydro_share_of_renewables = np.clip(hydro_share_of_renewables, 25, 80)

wind_share_of_renewables = 5 + 35 * (1 - np.exp(-renewables_acceleration / 8))
solar_share_of_renewables = 100 - hydro_share_of_renewables - wind_share_of_renewables
solar_share_of_renewables = np.clip(solar_share_of_renewables, 0, 50)

hydro_energy = renewables_energy * (hydro_share_of_renewables / 100)
wind_energy = renewables_energy * (wind_share_of_renewables / 100)
solar_energy = renewables_energy * (solar_share_of_renewables / 100)

# ============================================================================
# EMISSIONS
# ============================================================================

# CO2 emissions (million tonnes)
# Depends on fossil fuel use and carbon intensity

# Carbon intensity by fuel (kg CO2 per unit energy)
coal_intensity = 95
oil_intensity = 75
gas_intensity = 55

co2_emissions = (coal_energy * coal_intensity +
                oil_energy * oil_intensity +
                gas_energy * gas_intensity) / 1000  # Convert to appropriate units

# Total greenhouse gas emissions (CO2 equivalent, includes methane, N2O, etc.)
ghg_emissions = co2_emissions * 1.2

# Emissions per capita
population = gdp * 0.8 * (1.01 ** np.arange(YEARS))  # Growing population
emissions_per_capita = co2_emissions / population

# Emissions intensity (emissions per GDP)
emissions_intensity = co2_emissions / gdp

# ============================================================================
# ENERGY PRICES
# ============================================================================

# Oil price ($/barrel, volatile with geopolitical shocks)
oil_price_trend = 30 + 50 * (1 - np.exp(-np.arange(YEARS) / 25))

# Add major price shocks
oil_shocks = np.zeros(YEARS)
oil_shocks[3:7] = 40  # 1973 oil crisis
oil_shocks[9:12] = 30  # 1979 oil crisis
oil_shocks[38:42] = 60  # 2008 spike

oil_price = oil_price_trend + oil_shocks + np.random.normal(0, 5, YEARS)
oil_price = np.clip(oil_price, 15, 150)

# Natural gas price (correlated with oil but lower)
gas_price = oil_price * 0.4 + np.random.normal(0, 1, YEARS)
gas_price = np.clip(gas_price, 2, 30)

# Coal price (lower, more stable)
coal_price = 40 + 20 * (1 - np.exp(-np.arange(YEARS) / 30)) + np.random.normal(0, 2, YEARS)
coal_price = np.clip(coal_price, 30, 100)

# Electricity price (cents/kWh)
electricity_price = 8 + 4 * (1 - np.exp(-np.arange(YEARS) / 25)) + \
                   0.05 * oil_price + np.random.normal(0, 0.3, YEARS)
electricity_price = np.clip(electricity_price, 6, 18)

# Renewable energy costs (declining dramatically)
# Solar PV cost ($/Watt)
solar_cost = 50 * np.exp(-renewables_acceleration / 6) + 0.3
solar_cost = np.clip(solar_cost, 0.3, 60)

# Wind cost ($/Watt)
wind_cost = 8 * np.exp(-renewables_acceleration / 10) + 1.5
wind_cost = np.clip(wind_cost, 1.2, 10)

# ============================================================================
# ENVIRONMENTAL INDICATORS
# ============================================================================

# Air quality index (100 = 1970 baseline, lower is better)
air_quality = 100 - 30 * (1 - np.exp(-np.arange(YEARS) / 25)) + \
             0.2 * (coal_share + oil_share) + np.random.normal(0, 3, YEARS)
air_quality = np.clip(air_quality, 40, 110)

# Renewable energy investment (% of GDP)
renewables_investment = 0.1 + 2.5 * (renewables_share / 100) + np.random.normal(0, 0.1, YEARS)
renewables_investment = np.clip(renewables_investment, 0, 4)

# Energy R&D spending (% of GDP)
energy_rd = 0.05 + 0.15 * (1 - np.exp(-renewables_acceleration / 10)) + np.random.normal(0, 0.01, YEARS)
energy_rd = np.clip(energy_rd, 0.02, 0.3)

# Carbon price ($/tonne CO2, recent introduction)
carbon_price = np.zeros(YEARS)
carbon_price_start = 35  # Around 2005
if YEARS > carbon_price_start:
    carbon_price[carbon_price_start:] = 5 + 40 * (1 - np.exp(-(np.arange(YEARS - carbon_price_start) / 8))) + \
                                        np.random.normal(0, 3, YEARS - carbon_price_start)
carbon_price = np.clip(carbon_price, 0, 100)

# ============================================================================
# EMPLOYMENT IN ENERGY SECTORS
# ============================================================================

# Fossil fuel employment (thousands, declining)
fossil_employment = 800 - 400 * (1 - np.exp(-np.arange(YEARS) / 30)) + np.random.normal(0, 20, YEARS)
fossil_employment = np.clip(fossil_employment, 300, 900)

# Renewable energy employment (thousands, rising)
renewable_employment = 50 + 800 * (1 - np.exp(-renewables_acceleration / 10)) + np.random.normal(0, 30, YEARS)
renewable_employment = np.clip(renewable_employment, 40, 1200)

# ============================================================================
# ENERGY SECURITY
# ============================================================================

# Energy import dependency (%)
import_dependency = 25 + 15 * (1 - np.exp(-np.arange(YEARS) / 20)) - \
                   0.3 * renewables_share + np.random.normal(0, 2, YEARS)
import_dependency = np.clip(import_dependency, 10, 60)

# Energy supply diversity index (0-1, higher = more diverse)
# Increases as energy mix becomes more balanced
max_share = np.maximum.reduce([coal_share, oil_share, gas_share, nuclear_share, renewables_share])
diversity_index = 1 - (max_share - 20) / 80  # Stylized
diversity_index = np.clip(diversity_index, 0.3, 0.9)

# ============================================================================
# ENERGY POVERTY AND DISTRIBUTIONAL EFFECTS
# ============================================================================

# Energy burden (% of income spent on energy)
# Higher for low-income households
avg_energy_burden = 5 + 0.1 * electricity_price + np.random.normal(0, 0.2, YEARS)

# Low-income energy burden (2-3x average)
low_income_energy_burden = avg_energy_burden * 2.5

# Energy poverty rate (% unable to afford adequate energy)
energy_poverty_rate = 10 + 0.5 * electricity_price - 0.2 * gdp_growth * 100 + np.random.normal(0, 1, YEARS)
energy_poverty_rate = np.clip(energy_poverty_rate, 5, 20)

# ============================================================================
# REGIONAL DISPARITIES
# ============================================================================

# Renewable potential varies by region
# Share of renewable energy capacity in different regions

rural_renewables_share = renewables_share * 1.3  # Rural areas have more capacity
urban_renewables_share = renewables_share * 0.7  # Cities lag

# ============================================================================
# CONSTRUCT DATAFRAME
# ============================================================================

data = pd.DataFrame({
    'year': years,
    'gdp': gdp.round(2),
    'population': population.round(2),

    # Total energy
    'total_energy_consumption': total_energy.round(2),
    'energy_intensity': energy_intensity.round(3),

    # Energy by sector (% of total)
    'industrial_share': industrial_share.round(2),
    'transport_share': transport_share.round(2),
    'residential_share': residential_share.round(2),
    'commercial_share': commercial_share.round(2),

    # Energy by sector (absolute)
    'industrial_energy': industrial_energy.round(2),
    'transport_energy': transport_energy.round(2),
    'residential_energy': residential_energy.round(2),
    'commercial_energy': commercial_energy.round(2),

    # Energy by source (% of total)
    'coal_share': coal_share.round(2),
    'oil_share': oil_share.round(2),
    'gas_share': gas_share.round(2),
    'nuclear_share': nuclear_share.round(2),
    'renewables_share': renewables_share.round(2),

    # Energy by source (absolute)
    'coal_energy': coal_energy.round(2),
    'oil_energy': oil_energy.round(2),
    'gas_energy': gas_energy.round(2),
    'nuclear_energy': nuclear_energy.round(2),
    'renewables_energy': renewables_energy.round(2),

    # Renewable breakdown
    'hydro_energy': hydro_energy.round(2),
    'wind_energy': wind_energy.round(2),
    'solar_energy': solar_energy.round(2),

    # Emissions
    'co2_emissions': co2_emissions.round(2),
    'ghg_emissions': ghg_emissions.round(2),
    'emissions_per_capita': emissions_per_capita.round(3),
    'emissions_intensity': emissions_intensity.round(3),

    # Prices
    'oil_price_usd_barrel': oil_price.round(2),
    'gas_price_usd_mmbtu': gas_price.round(2),
    'coal_price_usd_ton': coal_price.round(2),
    'electricity_price_cents_kwh': electricity_price.round(2),
    'solar_cost_usd_watt': solar_cost.round(2),
    'wind_cost_usd_watt': wind_cost.round(2),

    # Environmental
    'air_quality_index': air_quality.round(2),
    'renewables_investment_gdp': renewables_investment.round(2),
    'energy_rd_gdp': energy_rd.round(3),
    'carbon_price_usd_tonne': carbon_price.round(2),

    # Employment
    'fossil_employment_thousands': fossil_employment.round(0),
    'renewable_employment_thousands': renewable_employment.round(0),

    # Energy security
    'import_dependency_pct': import_dependency.round(2),
    'energy_diversity_index': diversity_index.round(3),

    # Distributional
    'avg_energy_burden_pct': avg_energy_burden.round(2),
    'low_income_energy_burden_pct': low_income_energy_burden.round(2),
    'energy_poverty_rate': energy_poverty_rate.round(2),

    # Regional
    'rural_renewables_share': rural_renewables_share.round(2),
    'urban_renewables_share': urban_renewables_share.round(2),
})

# Save to CSV
data.to_csv('/home/user/Python-learning/datasets/energy_environment_data.csv', index=False)

print(f"Generated energy & environment dataset with {len(data)} observations")
print(f"Years: {data['year'].min()} to {data['year'].max()}\n")

print("Energy transition indicators (1970 vs 2024):")
print(f"Renewables share: {data['renewables_share'].iloc[0]:.1f}% → {data['renewables_share'].iloc[-1]:.1f}%")
print(f"Coal share: {data['coal_share'].iloc[0]:.1f}% → {data['coal_share'].iloc[-1]:.1f}%")
print(f"Energy intensity: {data['energy_intensity'].iloc[0]:.3f} → {data['energy_intensity'].iloc[-1]:.3f}")
print(f"Emissions intensity: {data['emissions_intensity'].iloc[0]:.3f} → {data['emissions_intensity'].iloc[-1]:.3f}")

print(f"\n2024 statistics:")
print(f"Solar cost: ${data['solar_cost_usd_watt'].iloc[-1]:.2f}/W (from ${data['solar_cost_usd_watt'].iloc[0]:.2f}/W in 1970)")
print(f"Renewable employment: {data['renewable_employment_thousands'].iloc[-1]:.0f}k")
print(f"Fossil employment: {data['fossil_employment_thousands'].iloc[-1]:.0f}k")
print(f"Carbon price: ${data['carbon_price_usd_tonne'].iloc[-1]:.2f}/tonne")
print(f"Energy poverty rate: {data['energy_poverty_rate'].iloc[-1]:.1f}%")
