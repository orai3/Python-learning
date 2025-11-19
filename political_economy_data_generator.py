"""
Institutional Political Economy Dataset Generator
Generates realistic synthetic data for 40 countries over 50 years (1974-2023)
Covers: Varieties of Capitalism, Power Resources Theory, Financialization, Neoliberalism

Theory-grounded with realistic correlations and historical dynamics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats

np.random.seed(42)

# Country configurations with regime types
COUNTRIES = {
    # Liberal Market Economies (LME)
    'USA': {'regime': 'LME', 'region': 'North America', 'income': 'high'},
    'United Kingdom': {'regime': 'LME', 'region': 'Europe', 'income': 'high'},
    'Canada': {'regime': 'LME', 'region': 'North America', 'income': 'high'},
    'Australia': {'regime': 'LME', 'region': 'Oceania', 'income': 'high'},
    'Ireland': {'regime': 'LME', 'region': 'Europe', 'income': 'high'},
    'New Zealand': {'regime': 'LME', 'region': 'Oceania', 'income': 'high'},

    # Coordinated Market Economies (CME)
    'Germany': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Sweden': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Denmark': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Norway': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Finland': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Netherlands': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Austria': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Belgium': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},
    'Switzerland': {'regime': 'CME', 'region': 'Europe', 'income': 'high'},

    # Mediterranean (Mixed Market Economies)
    'Italy': {'regime': 'MME', 'region': 'Europe', 'income': 'high'},
    'Spain': {'regime': 'MME', 'region': 'Europe', 'income': 'high'},
    'Portugal': {'regime': 'MME', 'region': 'Europe', 'income': 'high'},
    'Greece': {'regime': 'MME', 'region': 'Europe', 'income': 'high'},

    # Post-Socialist Transition Economies
    'Poland': {'regime': 'Transition', 'region': 'Europe', 'income': 'high'},
    'Czech Republic': {'regime': 'Transition', 'region': 'Europe', 'income': 'high'},
    'Hungary': {'regime': 'Transition', 'region': 'Europe', 'income': 'high'},
    'Estonia': {'regime': 'Transition', 'region': 'Europe', 'income': 'high'},
    'Slovenia': {'regime': 'Transition', 'region': 'Europe', 'income': 'high'},

    # East Asian Developmental States
    'Japan': {'regime': 'EAsia', 'region': 'Asia', 'income': 'high'},
    'South Korea': {'regime': 'EAsia', 'region': 'Asia', 'income': 'high'},
    'Taiwan': {'regime': 'EAsia', 'region': 'Asia', 'income': 'high'},
    'Singapore': {'regime': 'EAsia', 'region': 'Asia', 'income': 'high'},

    # Latin American
    'Brazil': {'regime': 'LatAm', 'region': 'Latin America', 'income': 'upper-middle'},
    'Argentina': {'regime': 'LatAm', 'region': 'Latin America', 'income': 'upper-middle'},
    'Chile': {'regime': 'LatAm', 'region': 'Latin America', 'income': 'high'},
    'Mexico': {'regime': 'LatAm', 'region': 'Latin America', 'income': 'upper-middle'},
    'Colombia': {'regime': 'LatAm', 'region': 'Latin America', 'income': 'upper-middle'},

    # Statist/Hybrid
    'France': {'regime': 'Statist', 'region': 'Europe', 'income': 'high'},
    'China': {'regime': 'Statist', 'region': 'Asia', 'income': 'upper-middle'},
    'India': {'regime': 'Developing', 'region': 'Asia', 'income': 'lower-middle'},
    'South Africa': {'regime': 'Developing', 'region': 'Africa', 'income': 'upper-middle'},
    'Turkey': {'regime': 'Developing', 'region': 'Europe/Asia', 'income': 'upper-middle'},
    'Indonesia': {'regime': 'Developing', 'region': 'Asia', 'income': 'lower-middle'},
    'Thailand': {'regime': 'Developing', 'region': 'Asia', 'income': 'upper-middle'},
}

# Time period: 1974-2023 (annual data)
START_YEAR = 1974
END_YEAR = 2023
YEARS = list(range(START_YEAR, END_YEAR + 1))
N_YEARS = len(YEARS)

# Regime baseline characteristics (mean values)
REGIME_BASELINES = {
    'LME': {
        'labor_coord': 0.25, 'union_density': 0.20, 'employment_protection': 0.30,
        'vocational_training': 0.30, 'stakeholder_governance': 0.25,
        'left_cabinet_share': 0.15, 'welfare_generosity': 0.35,
        'financial_sector_gdp': 0.12, 'financialization': 0.70,
        'neoliberalism': 0.75, 'trade_openness': 0.60, 'labor_flexibility': 0.75
    },
    'CME': {
        'labor_coord': 0.80, 'union_density': 0.55, 'employment_protection': 0.70,
        'vocational_training': 0.75, 'stakeholder_governance': 0.80,
        'left_cabinet_share': 0.45, 'welfare_generosity': 0.75,
        'financial_sector_gdp': 0.08, 'financialization': 0.35,
        'neoliberalism': 0.35, 'trade_openness': 0.70, 'labor_flexibility': 0.35
    },
    'MME': {
        'labor_coord': 0.50, 'union_density': 0.40, 'employment_protection': 0.65,
        'vocational_training': 0.40, 'stakeholder_governance': 0.50,
        'left_cabinet_share': 0.35, 'welfare_generosity': 0.55,
        'financial_sector_gdp': 0.09, 'financialization': 0.45,
        'neoliberalism': 0.50, 'trade_openness': 0.50, 'labor_flexibility': 0.50
    },
    'Transition': {
        'labor_coord': 0.35, 'union_density': 0.30, 'employment_protection': 0.45,
        'vocational_training': 0.50, 'stakeholder_governance': 0.40,
        'left_cabinet_share': 0.25, 'welfare_generosity': 0.45,
        'financial_sector_gdp': 0.07, 'financialization': 0.50,
        'neoliberalism': 0.65, 'trade_openness': 0.65, 'labor_flexibility': 0.60
    },
    'EAsia': {
        'labor_coord': 0.60, 'union_density': 0.25, 'employment_protection': 0.55,
        'vocational_training': 0.65, 'stakeholder_governance': 0.70,
        'left_cabinet_share': 0.15, 'welfare_generosity': 0.40,
        'financial_sector_gdp': 0.10, 'financialization': 0.55,
        'neoliberalism': 0.45, 'trade_openness': 0.80, 'labor_flexibility': 0.50
    },
    'LatAm': {
        'labor_coord': 0.40, 'union_density': 0.35, 'employment_protection': 0.50,
        'vocational_training': 0.35, 'stakeholder_governance': 0.35,
        'left_cabinet_share': 0.30, 'welfare_generosity': 0.40,
        'financial_sector_gdp': 0.08, 'financialization': 0.55,
        'neoliberalism': 0.60, 'trade_openness': 0.55, 'labor_flexibility': 0.65
    },
    'Statist': {
        'labor_coord': 0.55, 'union_density': 0.45, 'employment_protection': 0.60,
        'vocational_training': 0.55, 'stakeholder_governance': 0.55,
        'left_cabinet_share': 0.40, 'welfare_generosity': 0.60,
        'financial_sector_gdp': 0.09, 'financialization': 0.50,
        'neoliberalism': 0.45, 'trade_openness': 0.50, 'labor_flexibility': 0.45
    },
    'Developing': {
        'labor_coord': 0.30, 'union_density': 0.25, 'employment_protection': 0.40,
        'vocational_training': 0.30, 'stakeholder_governance': 0.30,
        'left_cabinet_share': 0.25, 'welfare_generosity': 0.30,
        'financial_sector_gdp': 0.06, 'financialization': 0.40,
        'neoliberalism': 0.60, 'trade_openness': 0.60, 'labor_flexibility': 0.70
    }
}


def generate_country_timeseries(country, regime, n_years):
    """
    Generate realistic time series for a country with:
    - Path dependence (AR process)
    - Critical junctures (structural breaks)
    - Institutional complementarities (correlations)
    """

    baselines = REGIME_BASELINES[regime]
    data = {}

    # Critical junctures (year indices)
    neoliberal_turn = 1980 - START_YEAR  # ~1980
    financialization_accel = 1995 - START_YEAR  # Mid-1990s
    financial_crisis = 2008 - START_YEAR  # 2008

    # Time trend for financialization and neoliberalism
    t = np.arange(n_years)

    # === VARIETIES OF CAPITALISM INDICATORS ===

    # Labor market coordination (slow-moving institution)
    labor_coord_base = baselines['labor_coord']
    # Slight decline post-1980 for most regimes except CME
    if regime == 'CME':
        trend = -0.001 * (t > neoliberal_turn) * (t - neoliberal_turn)
    else:
        trend = -0.003 * (t > neoliberal_turn) * (t - neoliberal_turn)

    data['labor_market_coordination'] = np.clip(
        labor_coord_base + trend + np.random.normal(0, 0.03, n_years).cumsum() * 0.01,
        0, 1
    )

    # Union density (declining trend post-1980, except Nordic)
    union_base = baselines['union_density']
    if regime == 'CME' and country in ['Sweden', 'Denmark', 'Finland']:
        decline_rate = -0.0015
    elif regime == 'CME':
        decline_rate = -0.002
    else:
        decline_rate = -0.004

    trend = decline_rate * (t > neoliberal_turn) * (t - neoliberal_turn)
    data['union_density'] = np.clip(
        union_base + trend + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.05, 0.95
    )

    # Employment protection legislation
    epl_base = baselines['employment_protection']
    # Some deregulation post-1980, especially LME
    if regime == 'LME':
        trend = -0.002 * (t > neoliberal_turn) * (t - neoliberal_turn)
    else:
        trend = -0.001 * (t > neoliberal_turn) * (t - neoliberal_turn)

    data['employment_protection'] = np.clip(
        epl_base + trend + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.1, 0.9
    )

    # Vocational training system strength
    voc_base = baselines['vocational_training']
    data['vocational_training'] = np.clip(
        voc_base + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.1, 0.9
    )

    # Corporate governance (stakeholder orientation)
    gov_base = baselines['stakeholder_governance']
    # Shift toward shareholder value post-1990
    if regime in ['LME', 'Transition']:
        trend = -0.003 * (t > financialization_accel) * (t - financialization_accel)
    else:
        trend = -0.001 * (t > financialization_accel) * (t - financialization_accel)

    data['stakeholder_governance'] = np.clip(
        gov_base + trend + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.1, 0.9
    )

    # Interfirm cooperation
    interfirm_base = 0.7 if regime in ['CME', 'EAsia'] else 0.3
    data['interfirm_cooperation'] = np.clip(
        interfirm_base + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.1, 0.9
    )

    # === POWER RESOURCES THEORY ===

    # Left party cabinet share (volatile, political cycles)
    left_base = baselines['left_cabinet_share']
    # Political cycles (8-12 year periods)
    cycle = 0.15 * np.sin(2 * np.pi * t / 10 + np.random.uniform(0, 2*np.pi))
    data['left_cabinet_share'] = np.clip(
        left_base + cycle + np.random.normal(0, 0.08, n_years),
        0, 1
    )

    # Welfare state generosity (correlated with left power, union strength)
    welfare_base = baselines['welfare_generosity']
    # Slight retrenchment post-1980
    trend = -0.002 * (t > neoliberal_turn) * (t - neoliberal_turn)
    # Correlation with union density
    welfare_union_effect = 0.3 * (data['union_density'] - union_base)

    data['welfare_generosity'] = np.clip(
        welfare_base + trend + welfare_union_effect + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.2, 0.9
    )

    # Strike activity (declining over time, spikes during crises)
    strike_base = 50 if regime == 'CME' else (30 if regime == 'MME' else 15)
    crisis_spikes = np.zeros(n_years)
    crisis_spikes[neoliberal_turn:neoliberal_turn+3] = 30
    crisis_spikes[financial_crisis:financial_crisis+2] = 25

    data['strike_days_per_1000'] = np.maximum(
        strike_base * np.exp(-0.02 * t) + crisis_spikes + np.random.gamma(2, 5, n_years),
        0
    )

    # Labor decommodification index (Esping-Andersen style)
    data['decommodification_index'] = np.clip(
        0.4 * data['welfare_generosity'] +
        0.3 * data['union_density'] +
        0.3 * data['employment_protection'] +
        np.random.normal(0, 0.03, n_years),
        0, 1
    )

    # === FINANCIALIZATION ===

    # Financial sector share of GDP (rising trend)
    fin_gdp_base = baselines['financial_sector_gdp']
    # Accelerating growth 1980-2007, slight decline post-2008
    trend = np.where(
        t < financial_crisis,
        0.002 * (t > neoliberal_turn) * (t - neoliberal_turn),
        0.002 * (financial_crisis - neoliberal_turn) - 0.001 * (t - financial_crisis)
    )

    data['financial_sector_gdp_share'] = np.clip(
        fin_gdp_base + trend + np.random.normal(0, 0.005, n_years).cumsum() * 0.01,
        0.03, 0.25
    )

    # Household debt to income ratio
    hh_debt_base = 0.8 if regime in ['LME', 'CME'] else 0.5
    # Rapid rise 1990-2007
    trend = 0.03 * (t > financialization_accel) * (t - financialization_accel)
    trend = np.where(t > financial_crisis,
                     trend[financial_crisis] - 0.01 * (t - financial_crisis),
                     trend)

    data['household_debt_to_income'] = np.clip(
        hh_debt_base + trend + np.random.normal(0, 0.05, n_years).cumsum() * 0.02,
        0.2, 3.0
    )

    # Corporate debt to GDP
    corp_debt_base = 0.6 if regime == 'EAsia' else 0.4
    trend = 0.015 * (t > financialization_accel) * (t - financialization_accel)

    data['corporate_debt_to_gdp'] = np.clip(
        corp_debt_base + trend + np.random.normal(0, 0.03, n_years).cumsum() * 0.02,
        0.2, 2.0
    )

    # Stock market capitalization to GDP
    stock_base = 0.8 if regime == 'LME' else 0.4
    trend = 0.025 * (t > financialization_accel) * (t - financialization_accel)
    # Crashes in 2001, 2008
    crashes = np.zeros(n_years)
    crashes[2001 - START_YEAR] = -0.3
    crashes[financial_crisis] = -0.4

    stock_series = stock_base + trend + np.random.normal(0, 0.05, n_years).cumsum() * 0.03
    for i in range(n_years):
        if crashes[i] < 0:
            stock_series[i:] += crashes[i]
        if i > 0 and crashes[i-1] < 0:
            # Partial recovery
            stock_series[i:] += abs(crashes[i-1]) * 0.3

    data['stock_market_cap_to_gdp'] = np.clip(stock_series, 0.1, 2.5)

    # Non-financial corporate financial income ratio
    nfc_fin_base = 0.10 if regime == 'LME' else 0.05
    trend = 0.004 * (t > financialization_accel) * (t - financialization_accel)

    data['nfc_financial_income_ratio'] = np.clip(
        nfc_fin_base + trend + np.random.normal(0, 0.01, n_years).cumsum() * 0.01,
        0.02, 0.4
    )

    # Financial deregulation index
    dereg_base = 0.7 if regime == 'LME' else 0.3
    # Step changes: 1980s deregulation, 1990s further, slight re-reg post-2008
    dereg = np.full(n_years, dereg_base)
    dereg[neoliberal_turn:] += 0.15
    dereg[financialization_accel:] += 0.1
    dereg[financial_crisis:] -= 0.05
    dereg += np.random.normal(0, 0.02, n_years).cumsum() * 0.01

    data['financial_deregulation'] = np.clip(dereg, 0, 1)

    # Composite financialization index
    data['financialization_index'] = (
        0.25 * (data['financial_sector_gdp_share'] / 0.15) +
        0.20 * (data['household_debt_to_income'] / 1.5) +
        0.20 * (data['stock_market_cap_to_gdp'] / 1.0) +
        0.20 * (data['nfc_financial_income_ratio'] / 0.2) +
        0.15 * data['financial_deregulation']
    ) / 5

    # === NEOLIBERALISM ===

    # Trade openness (exports + imports / GDP)
    trade_base = baselines['trade_openness']
    # Globalization trend
    trend = 0.004 * t
    # Slight reversal post-2008
    trend = np.where(t > financial_crisis,
                     trend[financial_crisis] - 0.001 * (t - financial_crisis),
                     trend)

    data['trade_openness'] = np.clip(
        trade_base + trend + np.random.normal(0, 0.03, n_years).cumsum() * 0.02,
        0.2, 2.0
    )

    # Capital account openness (Chinn-Ito style)
    cap_base = 0.7 if regime in ['LME', 'CME'] else 0.3
    # Liberalization post-1980
    cap_open = np.full(n_years, cap_base)
    cap_open[neoliberal_turn:] += 0.01 * (t[neoliberal_turn:] - neoliberal_turn)
    cap_open += np.random.normal(0, 0.02, n_years).cumsum() * 0.01

    data['capital_account_openness'] = np.clip(cap_open, 0, 1)

    # Privatization index (cumulative)
    priv_base = 0.2
    # Wave of privatizations 1980s-1990s
    priv_rate = 0.015 if regime in ['LME', 'Transition'] else 0.008
    priv_trend = priv_rate * (t > neoliberal_turn) * (t - neoliberal_turn)

    data['privatization_index'] = np.clip(
        priv_base + priv_trend + np.random.normal(0, 0.01, n_years).cumsum() * 0.01,
        0, 1
    )

    # Labor market flexibility (inverse of protection + coordination)
    data['labor_market_flexibility'] = np.clip(
        1 - 0.5 * data['employment_protection'] - 0.5 * data['labor_market_coordination'] +
        np.random.normal(0, 0.03, n_years),
        0, 1
    )

    # Top marginal tax rate (declining post-1980)
    tax_base = 0.65 if regime == 'CME' else (0.50 if regime == 'Statist' else 0.40)
    tax_trend = -0.003 * (t > neoliberal_turn) * (t - neoliberal_turn)

    data['top_tax_rate'] = np.clip(
        tax_base + tax_trend + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.20, 0.90
    )

    # Social spending as % GDP
    social_base = baselines['welfare_generosity'] * 30  # Convert to %
    social_trend = -0.05 * (t > neoliberal_turn) * (t - neoliberal_turn)
    # But increases during crises
    crisis_increase = np.zeros(n_years)
    crisis_increase[financial_crisis:financial_crisis+5] = 2.0

    data['social_spending_gdp'] = np.clip(
        social_base + social_trend + crisis_increase + np.random.normal(0, 0.3, n_years).cumsum() * 0.1,
        5, 35
    )

    # Composite neoliberalism index
    data['neoliberalism_index'] = (
        0.20 * data['trade_openness'] / 1.0 +
        0.20 * data['capital_account_openness'] +
        0.15 * data['privatization_index'] +
        0.20 * data['labor_market_flexibility'] +
        0.15 * (1 - data['top_tax_rate']) +
        0.10 * (1 - data['social_spending_gdp'] / 30)
    )

    # === ADDITIONAL INSTITUTIONAL INDICATORS ===

    # Wage bargaining centralization
    data['wage_bargaining_level'] = np.clip(
        data['labor_market_coordination'] * 0.8 +
        data['union_density'] * 0.2 +
        np.random.normal(0, 0.03, n_years),
        0, 1
    )

    # Active labor market policy spending
    data['almp_spending_gdp'] = np.clip(
        (data['social_spending_gdp'] / 20) * (0.5 if regime == 'CME' else 0.2) +
        np.random.normal(0, 0.05, n_years).cumsum() * 0.02,
        0, 3
    )

    # Product market regulation
    pmr_base = 0.7 if regime in ['CME', 'Statist'] else 0.4
    pmr_trend = -0.004 * (t > neoliberal_turn) * (t - neoliberal_turn)

    data['product_market_regulation'] = np.clip(
        pmr_base + pmr_trend + np.random.normal(0, 0.02, n_years).cumsum() * 0.01,
        0.1, 0.9
    )

    # Wage share of GDP
    wage_share_base = 0.65 if regime == 'CME' else 0.58
    # Declining trend (Kaleckian power shift)
    wage_trend = -0.002 * t
    # Correlation with union density
    wage_union_effect = 0.15 * (data['union_density'] - union_base)

    data['wage_share_gdp'] = np.clip(
        wage_share_base + wage_trend + wage_union_effect + np.random.normal(0, 0.01, n_years).cumsum() * 0.01,
        0.45, 0.75
    )

    # Economic policy orientation (-1 = heterodox, +1 = orthodox)
    policy_base = 0.5 if regime == 'LME' else -0.2
    policy_shift = 0.01 * (t > neoliberal_turn) * (t - neoliberal_turn)

    data['policy_orthodoxy'] = np.clip(
        policy_base + policy_shift +
        0.3 * (data['left_cabinet_share'] - left_base) * -1 +  # Left -> heterodox
        np.random.normal(0, 0.05, n_years).cumsum() * 0.02,
        -1, 1
    )

    return data


def generate_full_dataset():
    """Generate complete dataset for all countries and years"""

    all_data = []

    for country, info in COUNTRIES.items():
        regime = info['regime']
        print(f"Generating data for {country} ({regime})...")

        country_data = generate_country_timeseries(country, regime, N_YEARS)

        # Create dataframe for this country
        for i, year in enumerate(YEARS):
            row = {
                'country': country,
                'year': year,
                'regime_type': regime,
                'region': info['region'],
                'income_level': info['income']
            }

            # Add all indicators for this year
            for indicator, values in country_data.items():
                row[indicator] = values[i]

            all_data.append(row)

    df = pd.DataFrame(all_data)

    # Add some derived indicators

    # Institutional complementarity measures
    # CME complementarity: high coord + high vocational training + high stakeholder gov
    df['cme_complementarity'] = (
        df['labor_market_coordination'] * 0.33 +
        df['vocational_training'] * 0.33 +
        df['stakeholder_governance'] * 0.34
    )

    # LME complementarity: high flexibility + high stock market + low coordination
    df['lme_complementarity'] = (
        df['labor_market_flexibility'] * 0.4 +
        (df['stock_market_cap_to_gdp'] / 2).clip(0, 1) * 0.3 +
        (1 - df['labor_market_coordination']) * 0.3
    )

    # Power resources composite
    df['power_resources_index'] = (
        df['union_density'] * 0.35 +
        df['left_cabinet_share'] * 0.30 +
        df['decommodification_index'] * 0.35
    )

    # Regulation school periodization
    # Golden Age (high wage share, low financialization, high labor power)
    df['golden_age_score'] = (
        df['wage_share_gdp'] * 0.4 +
        (1 - df['financialization_index']) * 0.3 +
        df['power_resources_index'] * 0.3
    )

    # Neoliberal regime score
    df['neoliberal_regime_score'] = (
        df['neoliberalism_index'] * 0.5 +
        df['financialization_index'] * 0.3 +
        (1 - df['power_resources_index']) * 0.2
    )

    return df


if __name__ == "__main__":
    print("=" * 80)
    print("INSTITUTIONAL POLITICAL ECONOMY DATASET GENERATOR")
    print("=" * 80)
    print(f"\nGenerating data for {len(COUNTRIES)} countries over {N_YEARS} years ({START_YEAR}-{END_YEAR})")
    print(f"Total observations: {len(COUNTRIES) * N_YEARS:,}")
    print("\nRegime types:")
    regime_counts = {}
    for c, info in COUNTRIES.items():
        regime = info['regime']
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    for regime, count in sorted(regime_counts.items()):
        print(f"  {regime}: {count} countries")

    print("\n" + "=" * 80)
    print("GENERATING DATA...")
    print("=" * 80 + "\n")

    df = generate_full_dataset()

    # Save to CSV
    output_file = '/home/user/Python-learning/political_economy_dataset.csv'
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nSaved to: {output_file}")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print("\nColumn categories:")
    print("  - Varieties of Capitalism indicators: 6")
    print("  - Power Resources Theory metrics: 5")
    print("  - Financialization measures: 7")
    print("  - Neoliberalism indicators: 8")
    print("  - Additional institutional measures: 9")
    print("  - Composite indices: 7")

    print("\nSample statistics (latest year, 2023):")
    latest = df[df['year'] == 2023]

    print("\nNeoliberalism Index by regime:")
    print(latest.groupby('regime_type')['neoliberalism_index'].mean().sort_values(ascending=False))

    print("\nFinancialization Index by regime:")
    print(latest.groupby('regime_type')['financialization_index'].mean().sort_values(ascending=False))

    print("\nPower Resources Index by regime:")
    print(latest.groupby('regime_type')['power_resources_index'].mean().sort_values(ascending=False))

    print("\n" + "=" * 80)
    print("Dataset ready for analysis!")
    print("=" * 80)
