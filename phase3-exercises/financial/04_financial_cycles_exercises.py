"""
Financial Analysis Exercises: Minsky Moments & Credit Cycles
Heterodox Economics Focus

Exercises cover financial instability analysis from Post-Keynesian perspective,
emphasizing Minsky's Financial Instability Hypothesis and credit cycle dynamics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXERCISE 1: Minsky's Financial Instability Hypothesis - Simulation
# ============================================================================
# THEORY: Minsky argued capitalist finance is inherently unstable. Stability
# breeds instability as risk-taking escalates from hedge → speculative → Ponzi.
# Reference: Minsky (1977, 1986, 1992)
# ============================================================================

def exercise_1_minsky_simulation():
    """
    Problem: Simulate Minsky's three financing regimes and identify
    'Minsky moments' when fragility triggers crisis.

    Reference: Minsky (1992), Keen (2013) on modeling Minsky
    """
    print("=" * 80)
    print("EXERCISE 1: Minsky's Financial Instability Hypothesis")
    print("=" * 80)

    # Simulation parameters
    np.random.seed(42)
    periods = 200  # Quarters (50 years)

    # SOLUTION: Build Minsky-style model
    # Key variables:
    # - Asset prices (endogenous boom-bust)
    # - Leverage ratio (debt/equity)
    # - Interest coverage (profits/interest payments)
    # - Investment rate

    # Initial conditions
    asset_price = np.zeros(periods)
    leverage = np.zeros(periods)
    interest_coverage = np.zeros(periods)
    investment_rate = np.zeros(periods)
    financing_regime = np.zeros(periods)  # 0=hedge, 1=speculative, 2=Ponzi

    # Starting values (conservative financing)
    asset_price[0] = 100
    leverage[0] = 0.3  # Low debt
    interest_coverage[0] = 5.0  # Profits >> interest
    investment_rate[0] = 0.05

    # Parameters
    r_safe = 0.03  # Safe interest rate
    profit_rate_trend = 0.06  # Trend profit rate

    # Simulate dynamics
    for t in range(1, periods):
        # Memory of stability → overconfidence
        stability_memory = np.mean(asset_price[max(0, t-20):t]) if t > 1 else asset_price[0]
        stability_factor = asset_price[t-1] / stability_memory if stability_memory > 0 else 1.0

        # Rising prices → optimism → more leverage
        if stability_factor > 1.02:  # Recent price gains
            leverage_increase = 0.02 * (stability_factor - 1)
        else:
            leverage_increase = -0.01  # Deleveraging

        leverage[t] = np.clip(leverage[t-1] + leverage_increase + np.random.normal(0, 0.01),
                              0.1, 0.9)

        # Profit rate with cyclicality
        cycle_component = 0.03 * np.sin(2 * np.pi * t / 40)  # ~10 year cycle
        profit_rate = profit_rate_trend + cycle_component + np.random.normal(0, 0.01)

        # Interest rate rises with leverage and asset prices (Minsky's financial fragility)
        interest_rate = r_safe + 0.05 * leverage[t] + 0.02 * np.log(asset_price[t-1] / 100)

        # Interest coverage ratio
        interest_burden = interest_rate * leverage[t]
        interest_coverage[t] = profit_rate / interest_burden if interest_burden > 0 else 10.0

        # Classify financing regime (Minsky's taxonomy)
        if interest_coverage[t] > 1.5:
            financing_regime[t] = 0  # HEDGE: Profits cover interest + principal
        elif interest_coverage[t] > 1.0:
            financing_regime[t] = 1  # SPECULATIVE: Profits cover interest, not principal
        else:
            financing_regime[t] = 2  # PONZI: Profits don't cover interest (borrowing to pay interest)

        # Investment rate depends on optimism and financing conditions
        optimism = stability_factor - 1
        financial_constraint = max(0, 1 - leverage[t] / 0.8)  # Constrained at high leverage

        investment_rate[t] = 0.05 + 0.1 * optimism * financial_constraint + np.random.normal(0, 0.01)
        investment_rate[t] = np.clip(investment_rate[t], 0, 0.2)

        # Asset price dynamics (fundamental + momentum + fragility)
        fundamental_value = 100 * (1 + profit_rate_trend) ** (t / 4)  # Quarterly compounding

        # Momentum (extrapolative expectations)
        if t > 4:
            momentum = 0.5 * (asset_price[t-1] - asset_price[t-4])
        else:
            momentum = 0

        # Fragility effect (Ponzi regime triggers crash)
        if financing_regime[t] == 2 and np.random.rand() < 0.3:  # Minsky moment!
            fragility_shock = -0.3 * asset_price[t-1]
        else:
            fragility_shock = 0

        # Asset price equation
        asset_price[t] = (0.7 * fundamental_value +  # Some anchor to fundamentals
                         0.3 * asset_price[t-1] +    # Persistence
                         momentum +                   # Momentum/bubble
                         fragility_shock +            # Crisis
                         np.random.normal(0, 5))      # Noise

        asset_price[t] = max(asset_price[t], 20)  # Floor

    # Create DataFrame
    df = pd.DataFrame({
        'quarter': np.arange(periods),
        'year': 1970 + np.arange(periods) / 4,
        'asset_price': asset_price,
        'leverage': leverage,
        'interest_coverage': interest_coverage,
        'investment_rate': investment_rate,
        'regime': financing_regime
    })

    # Identify Minsky moments (sharp price drops from Ponzi regime)
    df['price_change'] = df['asset_price'].pct_change()
    df['minsky_moment'] = ((df['regime'] == 2) & (df['price_change'] < -0.15))

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Minsky's Financial Instability Hypothesis: Simulation",
                 fontsize=14, fontweight='bold')

    # Plot 1: Asset prices with regime coloring
    regime_colors = {0: '#4CAF50', 1: '#FF9800', 2: '#F44336'}
    regime_names = {0: 'Hedge', 1: 'Speculative', 2: 'Ponzi'}

    for regime_val in [0, 1, 2]:
        mask = df['regime'] == regime_val
        axes[0, 0].scatter(df.loc[mask, 'year'], df.loc[mask, 'asset_price'],
                          c=regime_colors[regime_val], label=regime_names[regime_val],
                          s=20, alpha=0.6)

    # Mark Minsky moments
    minsky_mask = df['minsky_moment']
    axes[0, 0].scatter(df.loc[minsky_mask, 'year'], df.loc[minsky_mask, 'asset_price'],
                      c='black', marker='v', s=200, label='Minsky Moment',
                      edgecolors='yellow', linewidths=2, zorder=5)

    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Asset Price Index')
    axes[0, 0].set_title('Asset Prices by Financing Regime')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Leverage ratio over time
    axes[0, 1].plot(df['year'], df['leverage'], linewidth=2, color='#2196F3')
    axes[0, 1].axhline(y=0.5, color='orange', linestyle='--',
                      label='Moderate leverage', alpha=0.7)
    axes[0, 1].axhline(y=0.7, color='red', linestyle='--',
                      label='High leverage', alpha=0.7)

    # Shade Minsky moments
    for idx in df[minsky_mask].index:
        axes[0, 1].axvspan(df.loc[idx, 'year'] - 0.5, df.loc[idx, 'year'] + 0.5,
                          alpha=0.3, color='red')

    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Leverage (Debt/Equity)')
    axes[0, 1].set_title('Leverage Dynamics')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Interest coverage ratio (Minsky's key indicator)
    axes[1, 0].plot(df['year'], df['interest_coverage'], linewidth=2, color='#9C27B0')
    axes[1, 0].axhline(y=1.5, color='green', linestyle='--',
                      label='Hedge threshold', alpha=0.7)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--',
                      label='Ponzi threshold', alpha=0.7)

    # Shade regime periods
    axes[1, 0].fill_between(df['year'], 0, 1.0, alpha=0.2, color='red', label='Ponzi zone')
    axes[1, 0].fill_between(df['year'], 1.0, 1.5, alpha=0.2, color='orange', label='Speculative zone')

    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Interest Coverage Ratio')
    axes[1, 0].set_title('Interest Coverage (Profits / Interest Payments)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 8])

    # Plot 4: Regime composition over time (rolling window)
    window = 20  # 5-year rolling window
    regime_shares = []
    years_roll = []

    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        hedge_pct = (window_data['regime'] == 0).sum() / window * 100
        spec_pct = (window_data['regime'] == 1).sum() / window * 100
        ponzi_pct = (window_data['regime'] == 2).sum() / window * 100

        regime_shares.append([hedge_pct, spec_pct, ponzi_pct])
        years_roll.append(df.loc[i, 'year'])

    regime_shares = np.array(regime_shares)

    axes[1, 1].fill_between(years_roll, 0, regime_shares[:, 0],
                            label='Hedge', alpha=0.7, color='#4CAF50')
    axes[1, 1].fill_between(years_roll, regime_shares[:, 0],
                            regime_shares[:, 0] + regime_shares[:, 1],
                            label='Speculative', alpha=0.7, color='#FF9800')
    axes[1, 1].fill_between(years_roll, regime_shares[:, 0] + regime_shares[:, 1], 100,
                            label='Ponzi', alpha=0.7, color='#F44336')

    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Regime Composition (%)')
    axes[1, 1].set_title('Evolution of Financing Regimes (5-year window)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase3-exercises/financial/ex1_minsky_simulation.png',
                dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: phase3-exercises/financial/ex1_minsky_simulation.png")

    # Calculate statistics
    print("\nSIMULATION STATISTICS:")
    print("=" * 80)
    print(f"Total periods: {periods} quarters ({periods/4:.0f} years)")
    print(f"\nFinancing regime distribution:")
    for regime_val, name in regime_names.items():
        pct = (df['regime'] == regime_val).sum() / periods * 100
        print(f"  {name}: {pct:.1f}%")

    print(f"\nMinsky moments identified: {df['minsky_moment'].sum()}")
    print(f"Crisis frequency: {df['minsky_moment'].sum() / (periods/4):.2f} per decade")

    print(f"\nLeverage statistics:")
    print(f"  Mean: {df['leverage'].mean():.3f}")
    print(f"  Std: {df['leverage'].std():.3f}")
    print(f"  Max: {df['leverage'].max():.3f}")

    print(f"\nAsset price statistics:")
    print(f"  Mean: {df['asset_price'].mean():.1f}")
    print(f"  Std: {df['asset_price'].std():.1f}")
    print(f"  Max drawdown: {(df['asset_price'].max() - df['asset_price'].min()) / df['asset_price'].max() * 100:.1f}%")

    # ECONOMIC INTERPRETATION
    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION:")
    print("=" * 80)
    print("\n1. Minsky's Three Financing Regimes:")
    print("   a) HEDGE FINANCE: Conservative, stable")
    print("      - Profits cover both interest AND principal repayment")
    print("      - Self-sustaining, no need to roll over debt")
    print("      - Typical of post-crisis caution")
    print("   b) SPECULATIVE FINANCE: Moderate risk")
    print("      - Profits cover interest but NOT principal")
    print("      - Must roll over/refinance debt")
    print("      - Vulnerable to interest rate rises or profit declines")
    print("   c) PONZI FINANCE: Fragile, unstable")
    print("      - Profits don't even cover interest payments")
    print("      - Must borrow more to pay interest (debt compounds)")
    print("      - Requires ever-rising asset prices to sustain")
    print("      - Named after Charles Ponzi's pyramid scheme")

    print("\n2. The Financial Instability Hypothesis:")
    print("   'Stability is destabilizing' - Minsky's key insight")
    print("   Mechanism:")
    print("   ① Tranquil period → Profits stable, risks seem low")
    print("   ② Memory of stability → Lenders/borrowers become overconfident")
    print("   ③ Risk-taking escalates → Leverage rises, margins compress")
    print("   ④ System shifts: Hedge → Speculative → Ponzi")
    print("   ⑤ Fragility builds → Minor shock can trigger crisis")
    print("   ⑥ MINSKY MOMENT: Sudden recognition of insolvency")
    print("   ⑦ Fire sales, deleveraging, crisis spreads")

    print("\n3. Simulation Patterns:")
    print("   - Early periods: Conservative (post-crisis)")
    print("   - Middle periods: Gradual risk escalation")
    print("   - Late periods: Ponzi finance proliferates")
    print("   - Minsky moments: Sharp corrections")
    print("   - Cycle repeats: Each crisis resets to caution, then gradual buildup")

    print("\n4. Real-World Examples:")
    print("   - 1980s S&L Crisis: Real estate speculation, Ponzi schemes")
    print("   - 2000 Dot-com Bubble: Tech stocks, negative earnings")
    print("   - 2008 Financial Crisis: Subprime mortgages, securitization")
    print("     * Shadow banking heavily leveraged")
    print("     * Housing prices unsustainable")
    print("     * Many mortgages were Ponzi (interest-only, teaser rates)")
    print("   - 2020-2021: Crypto, meme stocks, SPACs")

    print("\n5. Policy Implications:")
    print("   Mainstream view: Markets self-correct, crises are exogenous shocks")
    print("   Minsky view: Instability is ENDOGENOUS, built into finance")
    print("   Therefore:")
    print("   - Regulation crucial (not 'distortion')")
    print("   - Countercyclical policy needed:")
    print("     * Tighten lending standards in booms (lean against wind)")
    print("     * Support in busts (lender of last resort)")
    print("   - Prudential regulation (capital requirements, leverage limits)")
    print("   - Combat 'money manager capitalism' (short-termism)")
    print("   - Consider alternatives: Public banking, cooperative finance")

    print("\n6. Heterodox Perspective:")
    print("   - Rejects efficient markets hypothesis")
    print("   - Instability is normal, not aberration")
    print("   - Finance is not neutral 'veil' but driver of real economy")
    print("   - Distributional conflicts matter (who benefits from bubble?)")
    print("   - Historical & institutional context crucial (path dependence)")

    return df


# ============================================================================
# EXERCISE 2: Credit Cycle Analysis - Spectral Decomposition
# ============================================================================
# THEORY: Credit cycles drive business cycles (Post-Keynesian/Austrian view).
# Spectral analysis identifies cyclical components in credit data.
# Reference: Schumpeter (1939), Kydland & Prescott (1990), Borio (2014)
# ============================================================================

def exercise_2_credit_cycle_analysis():
    """
    Problem: Extract credit cycles using spectral analysis and correlate
    with real economy variables.

    Reference: Borio (2014) on financial cycles, Drehmann et al (2012)
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Credit Cycle Analysis - Spectral Decomposition")
    print("=" * 80)

    # Generate synthetic credit and GDP data
    np.random.seed(42)
    periods = 240  # 60 years quarterly data

    t = np.arange(periods)
    quarters = t
    years = 1960 + t / 4

    # Credit-to-GDP ratio with multiple cyclical components
    # Long financial cycle (~15-20 years, Borio 2014)
    financial_cycle = 15 * np.sin(2 * np.pi * t / 60)

    # Medium business cycle (~8-10 years)
    business_cycle = 8 * np.sin(2 * np.pi * t / 32)

    # Short inventory cycle (~3-4 years, Kitchin cycle)
    inventory_cycle = 3 * np.sin(2 * np.pi * t / 14)

    # Trend (financialization - rising credit)
    trend = 50 + 0.15 * t

    # Credit-to-GDP ratio
    credit_gdp = trend + financial_cycle + business_cycle + inventory_cycle + np.random.normal(0, 2, periods)

    # GDP growth rate (affected by credit with lag)
    gdp_growth = 3 + 0.1 * financial_cycle + 0.2 * business_cycle + np.random.normal(0, 1, periods)

    # Asset prices (correlated with credit)
    asset_prices = 100 * np.exp(0.03 * t / 4 + (financial_cycle + business_cycle) / 50)

    # Create DataFrame
    df = pd.DataFrame({
        'quarter': quarters,
        'year': years,
        'credit_gdp': credit_gdp,
        'gdp_growth': gdp_growth,
        'asset_prices': asset_prices,
        'financial_cycle_true': financial_cycle,
        'business_cycle_true': business_cycle
    })

    # SOLUTION: Spectral analysis and cycle extraction
    def spectral_analysis(series):
        """
        Perform spectral analysis using FFT

        Args:
            series: Time series data

        Returns:
            frequencies, power spectrum
        """
        # Detrend (remove linear trend)
        detrended = signal.detrend(series)

        # FFT
        fft_vals = np.fft.fft(detrended)
        power = np.abs(fft_vals) ** 2

        # Frequencies (convert to years)
        n = len(series)
        freq = np.fft.fftfreq(n, d=0.25)  # Quarterly data

        # Keep only positive frequencies
        pos_mask = freq > 0
        freq_pos = freq[pos_mask]
        power_pos = power[pos_mask]

        # Convert to period (years)
        periods_years = 1 / freq_pos

        return periods_years, power_pos, detrended

    # Perform spectral analysis on credit-to-GDP
    periods_array, power, credit_detrended = spectral_analysis(df['credit_gdp'].values)

    # Find dominant cycles (peaks in power spectrum)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(power, prominence=np.max(power) * 0.1)

    dominant_periods = periods_array[peaks]
    dominant_powers = power[peaks]

    # Sort by power
    sorted_idx = np.argsort(dominant_powers)[::-1]
    dominant_periods = dominant_periods[sorted_idx]
    dominant_powers = dominant_powers[sorted_idx]

    print("\nDOMINANT CYCLES IDENTIFIED:")
    print("=" * 80)
    for i, (period, power_val) in enumerate(zip(dominant_periods[:5], dominant_powers[:5]), 1):
        cycle_type = ""
        if 15 <= period <= 25:
            cycle_type = " (Financial/Kuznets cycle)"
        elif 7 <= period <= 11:
            cycle_type = " (Juglar/Business cycle)"
        elif 3 <= period <= 5:
            cycle_type = " (Kitchin/Inventory cycle)"

        print(f"{i}. Period: {period:.1f} years, Power: {power_val:.0f}{cycle_type}")

    # Band-pass filter to extract specific cycles
    def bandpass_filter(data, low_period, high_period, sample_rate=4):
        """
        Extract specific cyclical component using band-pass filter

        Args:
            data: Time series
            low_period, high_period: Period range in years
            sample_rate: Samples per year (4 for quarterly)

        Returns:
            Filtered cycle
        """
        # Convert periods to frequencies
        low_freq = 1 / (high_period * sample_rate)
        high_freq = 1 / (low_period * sample_rate)

        # Butterworth band-pass filter
        sos = signal.butter(4, [low_freq, high_freq], btype='band', output='sos',
                           fs=sample_rate)
        filtered = signal.sosfilt(sos, data)

        return filtered

    # Extract different cycle components
    financial_cycle_ext = bandpass_filter(credit_detrended, 12, 25)  # 12-25 year financial cycle
    business_cycle_ext = bandpass_filter(credit_detrended, 6, 12)   # 6-12 year business cycle

    df['financial_cycle_extracted'] = financial_cycle_ext
    df['business_cycle_extracted'] = business_cycle_ext

    # Calculate turning points (peaks and troughs)
    def find_turning_points(series, window=8):
        """Find local maxima and minima"""
        peaks_idx, _ = find_peaks(series, distance=window)
        troughs_idx, _ = find_peaks(-series, distance=window)
        return peaks_idx, troughs_idx

    fin_peaks, fin_troughs = find_turning_points(financial_cycle_ext)
    bus_peaks, bus_troughs = find_turning_points(business_cycle_ext)

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 14))
    fig.suptitle('Credit Cycle Analysis: Spectral Decomposition',
                 fontsize=14, fontweight='bold')

    # Plot 1: Credit-to-GDP ratio with trend
    axes[0, 0].plot(df['year'], df['credit_gdp'], label='Observed', linewidth=1, alpha=0.7)
    # Fit polynomial trend
    trend_coeffs = np.polyfit(df['quarter'], df['credit_gdp'], 2)
    trend_fit = np.polyval(trend_coeffs, df['quarter'])
    axes[0, 0].plot(df['year'], trend_fit, 'r--', label='Trend', linewidth=2)
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Credit-to-GDP Ratio (%)')
    axes[0, 0].set_title('Credit-to-GDP Ratio: Raw Data')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Power spectrum
    axes[0, 1].plot(periods_array, power, linewidth=1.5)
    axes[0, 1].scatter(dominant_periods[:5], dominant_powers[:5],
                      color='red', s=100, zorder=5, marker='o',
                      edgecolors='black', linewidths=2, label='Dominant cycles')
    axes[0, 1].set_xlabel('Period (years)')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].set_title('Power Spectrum (Periodogram)')
    axes[0, 1].set_xlim([0, 40])
    axes[0, 1].axvline(x=15, color='green', linestyle='--', alpha=0.5, label='Financial cycle')
    axes[0, 1].axvline(x=8, color='blue', linestyle='--', alpha=0.5, label='Business cycle')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Extracted financial cycle
    axes[1, 0].plot(df['year'], financial_cycle_ext, label='Extracted (12-25yr)', linewidth=2)
    axes[1, 0].plot(df['year'], df['financial_cycle_true'], '--',
                   label='True (synthetic)', alpha=0.6, linewidth=2)
    axes[1, 0].scatter(df['year'].iloc[fin_peaks], financial_cycle_ext[fin_peaks],
                      color='green', marker='^', s=100, label='Peaks', zorder=5)
    axes[1, 0].scatter(df['year'].iloc[fin_troughs], financial_cycle_ext[fin_troughs],
                      color='red', marker='v', s=100, label='Troughs', zorder=5)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Deviation from Trend')
    axes[1, 0].set_title('Financial Cycle (12-25 years)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Extracted business cycle
    axes[1, 1].plot(df['year'], business_cycle_ext, label='Extracted (6-12yr)', linewidth=2)
    axes[1, 1].plot(df['year'], df['business_cycle_true'], '--',
                   label='True (synthetic)', alpha=0.6, linewidth=2)
    axes[1, 1].scatter(df['year'].iloc[bus_peaks], business_cycle_ext[bus_peaks],
                      color='green', marker='^', s=100, label='Peaks', zorder=5)
    axes[1, 1].scatter(df['year'].iloc[bus_troughs], business_cycle_ext[bus_troughs],
                      color='red', marker='v', s=100, label='Troughs', zorder=5)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Deviation from Trend')
    axes[1, 1].set_title('Business Cycle (6-12 years)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Correlation with GDP growth
    # Calculate rolling correlation
    window = 20
    rolling_corr = df['credit_gdp'].rolling(window).corr(df['gdp_growth'])

    axes[2, 0].scatter(df['credit_gdp'], df['gdp_growth'], alpha=0.5, s=20)
    axes[2, 0].set_xlabel('Credit-to-GDP Ratio (%)')
    axes[2, 0].set_ylabel('GDP Growth Rate (%)')
    axes[2, 0].set_title('Credit vs GDP Growth')
    # Fit line
    coeffs = np.polyfit(df['credit_gdp'], df['gdp_growth'], 1)
    fit_line = np.polyval(coeffs, df['credit_gdp'])
    axes[2, 0].plot(df['credit_gdp'], fit_line, 'r--',
                   label=f'Correlation: {df["credit_gdp"].corr(df["gdp_growth"]):.3f}',
                   linewidth=2)
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Rolling correlation
    axes[2, 1].plot(df['year'], rolling_corr, linewidth=2)
    axes[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2, 1].set_xlabel('Year')
    axes[2, 1].set_ylabel('Correlation Coefficient')
    axes[2, 1].set_title(f'Rolling Correlation: Credit vs GDP Growth ({window}Q window)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim([-1, 1])

    plt.tight_layout()
    plt.savefig('phase3-exercises/financial/ex2_credit_cycles.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: phase3-exercises/financial/ex2_credit_cycles.png")

    # Calculate cycle statistics
    print("\nCYCLE STATISTICS:")
    print("=" * 80)
    if len(fin_peaks) > 1:
        fin_peak_years = df['year'].iloc[fin_peaks].values
        fin_periods = np.diff(fin_peak_years)
        print(f"Financial cycle (peak-to-peak):")
        print(f"  Mean period: {np.mean(fin_periods):.1f} years")
        print(f"  Std: {np.std(fin_periods):.1f} years")
        print(f"  Number of complete cycles: {len(fin_periods)}")

    if len(bus_peaks) > 1:
        bus_peak_years = df['year'].iloc[bus_peaks].values
        bus_periods = np.diff(bus_peak_years)
        print(f"\nBusiness cycle (peak-to-peak):")
        print(f"  Mean period: {np.mean(bus_periods):.1f} years")
        print(f"  Std: {np.std(bus_periods):.1f} years")
        print(f"  Number of complete cycles: {len(bus_periods)}")

    # ECONOMIC INTERPRETATION
    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION:")
    print("=" * 80)
    print("\n1. Multiple Overlapping Cycles:")
    print("   Economic fluctuations aren't single 'business cycle' but multiple:")
    print("   a) Kitchin (inventory) cycle: ~3-4 years")
    print("      - Short fluctuations in inventory investment")
    print("   b) Juglar (business/fixed investment) cycle: ~7-11 years")
    print("      - Classic 'business cycle' of expansions/recessions")
    print("      - Driven by fixed capital investment dynamics")
    print("   c) Kuznets (infrastructure/financial) cycle: ~15-25 years")
    print("      - Long waves in infrastructure, demographics, finance")
    print("      - Credit/asset price cycles (Borio 2014)")
    print("   d) Kondratieff wave: ~45-60 years (not shown here)")
    print("      - Technological paradigm shifts")

    print("\n2. The Financial Cycle (Borio, Drehmann):")
    print("   Key findings from BIS research:")
    print("   - Financial cycles are LONGER than business cycles (15-20 vs 8-10 years)")
    print("   - Characterized by joint fluctuations in credit, asset prices, leverage")
    print("   - Peaks often followed by financial crises")
    print("   - Not well-captured by standard macro models (pre-2008)")

    print("\n3. Credit-Growth Relationship:")
    print("   Correlation is positive but time-varying:")
    print("   - In expansion: Credit fuels growth (investment, consumption)")
    print("   - Near peak: Credit builds fragility (Minsky)")
    print("   - In crisis: Deleveraging depresses growth (debt deflation)")
    print("   - 'Finance matters': Not neutral veil over real economy")

    print("\n4. Implications for Macro Policy:")
    print("   Traditional view (pre-2008):")
    print("   - Focus on stabilizing inflation & output (Taylor rule)")
    print("   - Finance assumed stable/self-regulating")
    print("   Post-crisis view:")
    print("   - Need to monitor financial cycle indicators:")
    print("     * Credit-to-GDP gaps")
    print("     * Asset price deviations from trend")
    print("     * Leverage ratios")
    print("   - Macroprudential policy crucial:")
    print("     * Countercyclical capital buffers")
    print("     * Loan-to-value caps")
    print("     * Debt-service-to-income limits")
    print("   - Monetary policy dilemma:")
    print("     * Low rates may fuel financial cycles")
    print("     * 'Leaning against the wind' vs 'cleaning up after'")

    print("\n5. Heterodox Perspective:")
    print("   Post-Keynesian/Circuitist insights:")
    print("   - Credit creates purchasing power (loans create deposits)")
    print("   - Bank lending drives cycles, not just real factors")
    print("   - Distributional effects: Who gets credit? At what terms?")
    print("   - Regulatory regime shapes cycle amplitude")
    print("   - Historical specificity: Each cycle has unique features")
    print("   Contrast with RBC/DSGE:")
    print("   - Mainstream: Cycles driven by technology shocks, supply-side")
    print("   - Heterodox: Cycles driven by demand, finance, institutions")

    print("\n6. Turning Point Analysis:")
    print("   Identifying peaks/troughs crucial for:")
    print("   - Policy timing (when to tighten/ease)")
    print("   - Crisis early warning (divergence from fundamentals)")
    print("   - Regime classification (expansion vs contraction)")
    print("   BUT: Real-time identification is hard!")
    print("   - Data revisions, end-point problems")
    print("   - Different indicators give different signals")
    print("   - Political pressure to ignore warning signs")

    return df, dominant_periods, financial_cycle_ext, business_cycle_ext


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
            "title": "Challenge 1: Keen's Minsky Model",
            "description": "Implement Steve Keen's full dynamical systems model of Minsky. "
                          "3 differential equations for wages, employment, debt. Explore "
                          "parameter sensitivity and chaos. Can stable equilibria exist?",
            "skills": "Differential equations, numerical integration, chaos theory",
            "reference": "Keen (1995, 2013)"
        },
        {
            "title": "Challenge 2: Credit-to-GDP Gap (BIS Methodology)",
            "description": "Implement Basel III credit-to-GDP gap using HP filter with λ=400,000. "
                          "Compare to alternative trends (Hamilton, polynomial). Test as "
                          "early warning indicator for financial crises using historical data.",
            "skills": "HP filter, crisis prediction, ROC curves",
            "reference": "Drehmann et al (2011), BIS (2010)"
        },
        {
            "title": "Challenge 3: Financial Conditions Index",
            "description": "Construct FCI combining credit spreads, term spreads, stock prices, "
                          "exchange rates, volatility. Use PCA to extract common factor. "
                          "Does it predict GDP better than traditional monetary indicators?",
            "skills": "PCA, index construction, forecasting",
            "reference": "Hatzius et al (2010), Brave & Butters (2011)"
        },
        {
            "title": "Challenge 4: Debt-Deflation Dynamics",
            "description": "Model Irving Fisher's debt-deflation theory. Falling prices increase "
                          "real debt burden → bankruptcies → further deflation. Simulate spiral. "
                          "When does economy stabilize vs collapse?",
            "skills": "Dynamic modeling, feedback loops, stability analysis",
            "reference": "Fisher (1933), Eggertsson & Krugman (2012)"
        },
        {
            "title": "Challenge 5: Sectoral Financial Balances",
            "description": "Implement Godley's sectoral balances identity: (S-I) + (T-G) + (M-X) = 0. "
                          "Analyze how private, government, foreign balances co-move. "
                          "Identify unsustainable configurations (e.g., twin deficits).",
            "skills": "Accounting identities, SFC modeling, sustainability",
            "reference": "Godley & Lavoie (2007)"
        },
        {
            "title": "Challenge 6: Agent-Based Crisis Model",
            "description": "Build simple ABM of financial crisis: banks, firms, households. "
                          "Network of loans. Simulate default contagion through balance sheets. "
                          "How does network structure affect systemic risk?",
            "skills": "Agent-based modeling, networks, Monte Carlo",
            "reference": "Gai & Kapadia (2010), Battiston et al (2012)"
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
    print("FINANCIAL ANALYSIS EXERCISES - HETERODOX ECONOMICS")
    print("=" * 80)
    print("\nThis module contains exercises on financial instability:")
    print("1. Minsky's Financial Instability Hypothesis (simulation)")
    print("2. Credit cycle analysis (spectral decomposition)")
    print("\nEach exercise includes:")
    print("  ✓ Economic problem grounded in Post-Keynesian finance theory")
    print("  ✓ Complete Python implementation")
    print("  ✓ Dynamic simulations and spectral analysis")
    print("  ✓ Heterodox economic interpretation")
    print("  ✓ Policy implications")
    print("  ✓ Extension challenges")
    print("\n" + "=" * 80)

    # Run exercises
    minsky_df = exercise_1_minsky_simulation()
    credit_df, dominant_periods, fin_cycle, bus_cycle = exercise_2_credit_cycle_analysis()
    extension_challenges()

    print("\n" + "=" * 80)
    print("ALL EXERCISES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run this file: python 04_financial_cycles_exercises.py")
    print("2. Examine the Minsky simulation and credit cycle visualizations")
    print("3. Apply spectral analysis to real credit data (BIS, Fed)")
    print("4. Attempt the extension challenges")
    print("\nKey Takeaway:")
    print("Financial instability is endogenous, not just bad luck or exogenous shocks.")
    print("Credit cycles are longer and more consequential than conventional business cycles.")
    print("Heterodox economics provides essential tools for understanding financial crises.")
    print("=" * 80 + "\n")
