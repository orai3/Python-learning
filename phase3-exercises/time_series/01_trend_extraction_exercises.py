"""
Time Series Analysis Exercises: Trend Extraction & Structural Breaks
Heterodox Economics Focus

These exercises cover techniques for analyzing economic time series with emphasis on
Post-Keynesian and institutional approaches to structural change and regime shifts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXERCISE 1: Basic Trend Extraction - Hodrick-Prescott Filter
# ============================================================================
# THEORY: The HP filter is commonly used in mainstream economics but criticized
# by heterodox economists (e.g., Hamilton 2018) for introducing spurious cycles.
# Understanding it helps critique mainstream "business cycle" analysis.
# ============================================================================

def exercise_1_basic_hp_filter():
    """
    Problem: Extract trend from GDP data using HP filter and critique the approach
    from a Post-Keynesian perspective.

    Reference: Hodrick & Prescott (1997), Hamilton (2018) critique
    """
    print("=" * 80)
    print("EXERCISE 1: Hodrick-Prescott Filter - Extracting Trends")
    print("=" * 80)

    # Generate synthetic GDP data with trend, cycle, and random components
    np.random.seed(42)
    t = np.linspace(0, 100, 400)  # Quarterly data for 100 years

    # Trend (long-run growth)
    trend = 1000 * np.exp(0.02 * t)

    # Cyclical component (business cycles)
    cycle = 50 * np.sin(2 * np.pi * t / 20) + 30 * np.sin(2 * np.pi * t / 8)

    # Random shocks
    noise = np.random.normal(0, 20, len(t))

    # Observed GDP
    gdp = trend + cycle + noise

    # Create DataFrame
    df = pd.DataFrame({
        'quarter': range(len(t)),
        'gdp': gdp,
        'true_trend': trend,
        'true_cycle': cycle
    })

    # SOLUTION: Implement HP Filter
    def hp_filter(y, lambda_param=1600):
        """
        Hodrick-Prescott filter implementation

        Args:
            y: Time series data
            lambda_param: Smoothing parameter (1600 for quarterly data)

        Returns:
            trend: Trend component
            cycle: Cyclical component
        """
        n = len(y)

        # Create second difference matrix
        I = np.eye(n)
        D2 = np.zeros((n-2, n))
        for i in range(n-2):
            D2[i, i:i+3] = [1, -2, 1]

        # HP filter formula: trend = (I + λD'D)^(-1) y
        trend = np.linalg.inv(I + lambda_param * D2.T @ D2) @ y
        cycle = y - trend

        return trend, cycle

    # Apply HP filter with different lambda values
    lambdas = [400, 1600, 6400]  # Low, standard, high smoothing

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('HP Filter with Different Smoothing Parameters', fontsize=14, fontweight='bold')

    for idx, lambda_val in enumerate(lambdas):
        hp_trend, hp_cycle = hp_filter(df['gdp'].values, lambda_val)

        # Plot trends
        axes[idx, 0].plot(df['quarter'], df['gdp'], alpha=0.5, label='Observed GDP', linewidth=1)
        axes[idx, 0].plot(df['quarter'], hp_trend, label=f'HP Trend (λ={lambda_val})', linewidth=2)
        axes[idx, 0].plot(df['quarter'], df['true_trend'], label='True Trend',
                         linestyle='--', linewidth=2)
        axes[idx, 0].set_title(f'Trend Extraction (λ={lambda_val})')
        axes[idx, 0].set_xlabel('Quarter')
        axes[idx, 0].set_ylabel('GDP')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)

        # Plot cycles
        axes[idx, 1].plot(df['quarter'], hp_cycle, label=f'HP Cycle (λ={lambda_val})', linewidth=1.5)
        axes[idx, 1].plot(df['quarter'], df['true_cycle'], label='True Cycle',
                         linestyle='--', linewidth=1.5, alpha=0.7)
        axes[idx, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[idx, 1].set_title(f'Cyclical Component (λ={lambda_val})')
        axes[idx, 1].set_xlabel('Quarter')
        axes[idx, 1].set_ylabel('Deviation from Trend')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase3-exercises/time_series/ex1_hp_filter.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: phase3-exercises/time_series/ex1_hp_filter.png")

    # ECONOMIC INTERPRETATION
    print("\nECONOMIC INTERPRETATION:")
    print("-" * 80)
    print("1. The HP filter is widely used but has significant limitations:")
    print("   - Assumes a smooth, deterministic trend (unrealistic for capitalist economies)")
    print("   - Can create spurious cyclical dynamics (Hamilton 2018)")
    print("   - Treats trend/cycle as independent (ignores hysteresis effects)")
    print("\n2. Post-Keynesian Critique:")
    print("   - Real economies don't have a 'natural' smooth trend")
    print("   - Demand shocks can have permanent effects (path dependence)")
    print("   - Focus should be on institutional changes, not mechanical filtering")
    print("\n3. Lambda Parameter Sensitivity:")
    print("   - Higher λ → smoother trend, larger cycles")
    print("   - Lower λ → trend follows data more closely, smaller cycles")
    print("   - Standard λ=1600 for quarterly data is arbitrary")

    # Calculate and display statistics
    for lambda_val in lambdas:
        _, hp_cycle = hp_filter(df['gdp'].values, lambda_val)
        print(f"\nλ={lambda_val}: Cycle std dev = {np.std(hp_cycle):.2f}, "
              f"Cycle/GDP ratio = {np.std(hp_cycle)/np.mean(df['gdp'])*100:.2f}%")

    return df


# ============================================================================
# EXERCISE 2: Structural Breaks - Chow Test for Regime Changes
# ============================================================================
# THEORY: Heterodox economics emphasizes institutional change and regime shifts.
# The Chow test helps identify when economic relationships fundamentally changed.
# Reference: Chow (1960), Post-Keynesian analysis of neoliberal transition
# ============================================================================

def exercise_2_structural_breaks():
    """
    Problem: Test for structural breaks in the wage share of GDP, relevant for
    identifying regime changes (e.g., post-1980 neoliberal shift).

    Reference: Chow (1960), Kaleckian analysis of distributional regimes
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Structural Breaks - Chow Test")
    print("=" * 80)

    # Generate synthetic wage share data with structural break
    np.random.seed(42)
    n = 200  # Quarterly data, 50 years

    # Pre-break period (1970s): Higher wage share, stable
    t1 = np.arange(80)
    wage_share_1 = 65 + 0.05 * t1 + np.random.normal(0, 1.5, len(t1))

    # Post-break period (1980s onward): Lower wage share, declining
    t2 = np.arange(120)
    wage_share_2 = 68 - 0.12 * t2 + np.random.normal(0, 1.5, len(t2))

    # Combine
    wage_share = np.concatenate([wage_share_1, wage_share_2])
    quarters = np.arange(len(wage_share))

    df = pd.DataFrame({
        'quarter': quarters,
        'wage_share': wage_share,
        'year': 1970 + quarters / 4
    })

    # SOLUTION: Implement Chow Test
    def chow_test(y, x, breakpoint):
        """
        Chow test for structural break at given breakpoint

        Args:
            y: Dependent variable
            x: Independent variable (time trend)
            breakpoint: Index of suspected break

        Returns:
            F-statistic, p-value, interpretation
        """
        # Full sample regression
        X_full = np.column_stack([np.ones(len(x)), x])
        beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
        resid_full = y - X_full @ beta_full
        rss_full = np.sum(resid_full ** 2)

        # Pre-break regression
        X_pre = np.column_stack([np.ones(breakpoint), x[:breakpoint]])
        beta_pre = np.linalg.lstsq(X_pre, y[:breakpoint], rcond=None)[0]
        resid_pre = y[:breakpoint] - X_pre @ beta_pre
        rss_pre = np.sum(resid_pre ** 2)

        # Post-break regression
        X_post = np.column_stack([np.ones(len(x) - breakpoint), x[breakpoint:]])
        beta_post = np.linalg.lstsq(X_post, y[breakpoint:], rcond=None)[0]
        resid_post = y[breakpoint:] - X_post @ beta_post
        rss_post = np.sum(resid_post ** 2)

        # Chow F-statistic
        rss_separate = rss_pre + rss_post
        k = 2  # Number of parameters
        n = len(y)

        F_stat = ((rss_full - rss_separate) / k) / (rss_separate / (n - 2*k))

        # P-value
        p_value = 1 - stats.f.cdf(F_stat, k, n - 2*k)

        return F_stat, p_value, beta_pre, beta_post

    # Test for break at quarter 80 (around 1990)
    breakpoint = 80
    F_stat, p_value, beta_pre, beta_post = chow_test(
        df['wage_share'].values,
        df['quarter'].values,
        breakpoint
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Structural Break Analysis: Wage Share of GDP', fontsize=14, fontweight='bold')

    # Plot 1: Full time series with break
    axes[0, 0].plot(df['year'], df['wage_share'], 'o', alpha=0.5, markersize=3, label='Observed')
    axes[0, 0].axvline(x=df['year'].iloc[breakpoint], color='red', linestyle='--',
                       linewidth=2, label='Suspected Break (1990)')

    # Fit lines pre and post break
    x_pre = df['quarter'].iloc[:breakpoint].values
    x_post = df['quarter'].iloc[breakpoint:].values
    y_pre_fit = beta_pre[0] + beta_pre[1] * x_pre
    y_post_fit = beta_post[0] + beta_post[1] * x_post

    axes[0, 0].plot(df['year'].iloc[:breakpoint], y_pre_fit, 'b-', linewidth=2,
                    label=f'Pre-break trend (slope={beta_pre[1]:.4f})')
    axes[0, 0].plot(df['year'].iloc[breakpoint:], y_post_fit, 'g-', linewidth=2,
                    label=f'Post-break trend (slope={beta_post[1]:.4f})')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Wage Share (%)')
    axes[0, 0].set_title('Wage Share with Structural Break')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Test multiple potential breakpoints
    test_range = range(40, 160)
    f_statistics = []
    p_values = []

    for bp in test_range:
        F, p, _, _ = chow_test(df['wage_share'].values, df['quarter'].values, bp)
        f_statistics.append(F)
        p_values.append(p)

    axes[0, 1].plot([df['year'].iloc[i] for i in test_range], f_statistics, 'b-', linewidth=2)
    axes[0, 1].axhline(y=stats.f.ppf(0.95, 2, len(df) - 4), color='red',
                       linestyle='--', label='5% Critical Value')
    axes[0, 1].set_xlabel('Potential Break Year')
    axes[0, 1].set_ylabel('Chow F-Statistic')
    axes[0, 1].set_title('F-Statistics Across Potential Break Points')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: P-values
    axes[1, 0].plot([df['year'].iloc[i] for i in test_range], p_values, 'g-', linewidth=2)
    axes[1, 0].axhline(y=0.05, color='red', linestyle='--', label='5% Significance Level')
    axes[1, 0].set_xlabel('Potential Break Year')
    axes[1, 0].set_ylabel('P-Value')
    axes[1, 0].set_title('P-Values Across Potential Break Points')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Residuals analysis
    residuals_pre = df['wage_share'].iloc[:breakpoint] - (beta_pre[0] + beta_pre[1] * df['quarter'].iloc[:breakpoint])
    residuals_post = df['wage_share'].iloc[breakpoint:] - (beta_post[0] + beta_post[1] * df['quarter'].iloc[breakpoint:])

    axes[1, 1].scatter(df['year'].iloc[:breakpoint], residuals_pre, alpha=0.5,
                      s=20, label='Pre-break residuals', color='blue')
    axes[1, 1].scatter(df['year'].iloc[breakpoint:], residuals_post, alpha=0.5,
                      s=20, label='Post-break residuals', color='green')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].axvline(x=df['year'].iloc[breakpoint], color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Regression Residuals')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase3-exercises/time_series/ex2_structural_breaks.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: phase3-exercises/time_series/ex2_structural_breaks.png")

    # ECONOMIC INTERPRETATION
    print("\nCHOW TEST RESULTS:")
    print("-" * 80)
    print(f"F-statistic: {F_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significant at 5% level: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"\nPre-break trend (1970-1990): {beta_pre[1]:.4f} pp/quarter")
    print(f"Post-break trend (1990-2020): {beta_post[1]:.4f} pp/quarter")
    print(f"Change in trend: {beta_post[1] - beta_pre[1]:.4f} pp/quarter")

    print("\nECONOMIC INTERPRETATION:")
    print("-" * 80)
    print("1. Structural Break Context:")
    print("   - 1980s: Neoliberal transition (Reagan/Thatcher era)")
    print("   - Weakening of labor unions, financialization")
    print("   - Shift from wage-led to profit-led growth regime")
    print("\n2. Kaleckian Perspective:")
    print("   - Wage share decline reflects power shift from labor to capital")
    print("   - This is NOT a 'natural' market outcome but institutional change")
    print("   - Has implications for aggregate demand (wage-led vs profit-led)")
    print("\n3. Policy Implications:")
    print("   - Trend changes reflect policy choices, not economic laws")
    print("   - Different institutional arrangements → different outcomes")
    print("   - Possibility of regime change back toward higher wage share")

    return df, breakpoint, F_stat, p_value


# ============================================================================
# EXERCISE 3: Rolling Window Analysis - Detecting Time-Varying Relationships
# ============================================================================
# THEORY: Post-Keynesian economics emphasizes that economic relationships are
# not stable parameters but change with institutional context.
# ============================================================================

def exercise_3_rolling_window_analysis():
    """
    Problem: Analyze time-varying relationship between capacity utilization
    and investment rate using rolling window regressions.

    Reference: Kaleckian investment theory, institutionalist analysis
    """
    print("\n" + "=" * 80)
    print("EXERCISE 3: Rolling Window Analysis - Time-Varying Relationships")
    print("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n = 200

    # Capacity utilization (relatively stable)
    capacity_util = 75 + 10 * np.sin(2 * np.pi * np.arange(n) / 40) + np.random.normal(0, 2, n)

    # Investment rate with changing sensitivity to capacity utilization
    # Early period: Strong accelerator effect (β ≈ 0.8)
    # Later period: Weaker effect due to financialization (β ≈ 0.3)
    beta_early = 0.8
    beta_late = 0.3

    # Smooth transition in beta
    beta_t = beta_early + (beta_late - beta_early) * (1 / (1 + np.exp(-0.1 * (np.arange(n) - 100))))

    investment_rate = 5 + beta_t * (capacity_util - 75) + np.random.normal(0, 1.5, n)

    df = pd.DataFrame({
        'quarter': np.arange(n),
        'year': 1970 + np.arange(n) / 4,
        'capacity_util': capacity_util,
        'investment_rate': investment_rate,
        'true_beta': beta_t
    })

    # SOLUTION: Rolling window regression
    def rolling_regression(x, y, window=40):
        """
        Perform rolling window OLS regression

        Args:
            x: Independent variable
            y: Dependent variable
            window: Window size

        Returns:
            DataFrame with rolling coefficients
        """
        results = []

        for i in range(window, len(x) + 1):
            # Get window data
            x_window = x[i-window:i]
            y_window = y[i-window:i]

            # Demean for clarity (coefficient = sensitivity)
            x_dm = x_window - np.mean(x_window)

            # OLS regression
            X = np.column_stack([np.ones(len(x_dm)), x_dm])
            beta = np.linalg.lstsq(X, y_window, rcond=None)[0]

            # Standard errors
            residuals = y_window - X @ beta
            rss = np.sum(residuals ** 2)
            var_beta = rss / (len(x_window) - 2) * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(var_beta))

            # R-squared
            tss = np.sum((y_window - np.mean(y_window)) ** 2)
            r_squared = 1 - rss / tss

            results.append({
                'end_quarter': i - 1,
                'intercept': beta[0],
                'slope': beta[1],
                'se_slope': se[1],
                'r_squared': r_squared
            })

        return pd.DataFrame(results)

    # Perform rolling regression
    window_size = 40  # 10 years of quarterly data
    rolling_results = rolling_regression(
        df['capacity_util'].values,
        df['investment_rate'].values,
        window=window_size
    )

    # Calculate confidence intervals
    rolling_results['ci_lower'] = rolling_results['slope'] - 1.96 * rolling_results['se_slope']
    rolling_results['ci_upper'] = rolling_results['slope'] + 1.96 * rolling_results['se_slope']

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Rolling Window Analysis: Capacity Utilization → Investment',
                 fontsize=14, fontweight='bold')

    # Plot 1: Time-varying coefficient
    valid_years = df['year'].iloc[rolling_results['end_quarter']].values
    axes[0, 0].plot(valid_years, rolling_results['slope'], 'b-', linewidth=2,
                    label='Estimated β (rolling)')
    axes[0, 0].fill_between(valid_years,
                             rolling_results['ci_lower'],
                             rolling_results['ci_upper'],
                             alpha=0.3, label='95% Confidence Interval')
    axes[0, 0].plot(df['year'].iloc[window_size-1:],
                    df['true_beta'].iloc[window_size-1:],
                    'r--', linewidth=2, label='True β')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Investment Sensitivity to Capacity Utilization')
    axes[0, 0].set_title('Time-Varying Accelerator Effect')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: R-squared over time
    axes[0, 1].plot(valid_years, rolling_results['r_squared'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('R-squared')
    axes[0, 1].set_title('Model Fit Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Plot 3: Scatter plots for two periods
    mid_point = len(df) // 2

    axes[1, 0].scatter(df['capacity_util'].iloc[:mid_point],
                       df['investment_rate'].iloc[:mid_point],
                       alpha=0.6, s=30, label='1970-1995')
    # Fit line
    X_early = np.column_stack([np.ones(mid_point),
                                df['capacity_util'].iloc[:mid_point] - df['capacity_util'].iloc[:mid_point].mean()])
    beta_early_est = np.linalg.lstsq(X_early, df['investment_rate'].iloc[:mid_point], rcond=None)[0]
    x_range = np.linspace(df['capacity_util'].min(), df['capacity_util'].max(), 100)
    y_early = beta_early_est[0] + beta_early_est[1] * (x_range - df['capacity_util'].iloc[:mid_point].mean())
    axes[1, 0].plot(x_range, y_early, 'b-', linewidth=2,
                    label=f'β = {beta_early_est[1]:.3f}')
    axes[1, 0].set_xlabel('Capacity Utilization (%)')
    axes[1, 0].set_ylabel('Investment Rate (%)')
    axes[1, 0].set_title('Early Period (1970-1995)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(df['capacity_util'].iloc[mid_point:],
                       df['investment_rate'].iloc[mid_point:],
                       alpha=0.6, s=30, color='green', label='1995-2020')
    # Fit line
    X_late = np.column_stack([np.ones(len(df) - mid_point),
                               df['capacity_util'].iloc[mid_point:] - df['capacity_util'].iloc[mid_point:].mean()])
    beta_late_est = np.linalg.lstsq(X_late, df['investment_rate'].iloc[mid_point:], rcond=None)[0]
    y_late = beta_late_est[0] + beta_late_est[1] * (x_range - df['capacity_util'].iloc[mid_point:].mean())
    axes[1, 1].plot(x_range, y_late, 'g-', linewidth=2,
                    label=f'β = {beta_late_est[1]:.3f}')
    axes[1, 1].set_xlabel('Capacity Utilization (%)')
    axes[1, 1].set_ylabel('Investment Rate (%)')
    axes[1, 1].set_title('Late Period (1995-2020)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase3-exercises/time_series/ex3_rolling_window.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: phase3-exercises/time_series/ex3_rolling_window.png")

    # ECONOMIC INTERPRETATION
    print("\nROLLING WINDOW RESULTS:")
    print("-" * 80)
    print(f"Early period β (1970s-1980s): {rolling_results['slope'].iloc[0]:.3f}")
    print(f"Late period β (2010s-2020s): {rolling_results['slope'].iloc[-1]:.3f}")
    print(f"Decline in accelerator effect: {rolling_results['slope'].iloc[-1] - rolling_results['slope'].iloc[0]:.3f}")

    print("\nECONOMIC INTERPRETATION:")
    print("-" * 80)
    print("1. Weakening Accelerator Mechanism:")
    print("   - Investment less responsive to capacity utilization over time")
    print("   - Suggests structural change in investment behavior")
    print("\n2. Kaleckian/Post-Keynesian Explanation:")
    print("   - Financialization: Profits increasingly channeled to financial assets")
    print("   - Shareholder value maximization: Dividends/buybacks vs investment")
    print("   - Weakened industrial base: Less need for capacity expansion")
    print("\n3. Implications:")
    print("   - Harder to stimulate investment through demand policies")
    print("   - Need for institutional changes (not just monetary/fiscal policy)")
    print("   - Supports argument for industrial policy, public investment")
    print("\n4. Methodological Point:")
    print("   - Fixed coefficient models miss these crucial structural changes")
    print("   - Heterodox emphasis on historical specificity is validated")

    return df, rolling_results


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
            "title": "Challenge 1: Multiple Structural Breaks",
            "description": "Implement Bai-Perron test for multiple unknown break points. "
                          "Apply to profit share data to identify: (1) Post-war accord period, "
                          "(2) Neoliberal transition, (3) Post-2008 regime.",
            "skills": "Sequential F-tests, BIC model selection",
            "reference": "Bai & Perron (1998, 2003)"
        },
        {
            "title": "Challenge 2: Hamilton Filter Alternative",
            "description": "Implement Hamilton's (2018) alternative to HP filter using regression "
                          "on past values. Compare results for GDP trend extraction. Does it better "
                          "preserve low-frequency movements?",
            "skills": "OLS regression, lag operators",
            "reference": "Hamilton (2018)"
        },
        {
            "title": "Challenge 3: State-Space Models",
            "description": "Use Kalman filter to model time-varying NAIRU (non-accelerating "
                          "inflation rate of unemployment). Compare to fixed NAIRU assumption. "
                          "Critique from hysteresis perspective.",
            "skills": "Kalman filtering, state-space representation",
            "reference": "Stock & Watson (1998), Ball (2009) on hysteresis"
        },
        {
            "title": "Challenge 4: Regime-Switching Models",
            "description": "Implement Markov-switching model for growth regimes (high/low growth). "
                          "Do regime switches coincide with policy changes or financial crises?",
            "skills": "EM algorithm, Markov chains, maximum likelihood",
            "reference": "Hamilton (1989), Psaradakis & Sola (1998)"
        },
        {
            "title": "Challenge 5: Wavelet Analysis",
            "description": "Apply wavelet decomposition to analyze time-frequency relationships "
                          "between financial variables and real economy. Do correlations vary "
                          "at different time horizons?",
            "skills": "Wavelet transforms, time-frequency analysis",
            "reference": "Aguiar-Conraria & Soares (2011)"
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
    print("TIME SERIES ANALYSIS EXERCISES - HETERODOX ECONOMICS")
    print("=" * 80)
    print("\nThis module contains exercises progressing from basic to advanced:")
    print("1. HP Filter (with heterodox critique)")
    print("2. Structural breaks (Chow test)")
    print("3. Rolling window analysis (time-varying relationships)")
    print("\nEach exercise includes:")
    print("  ✓ Economic problem grounded in heterodox theory")
    print("  ✓ Complete Python implementation")
    print("  ✓ Visualizations")
    print("  ✓ Economic interpretation from heterodox perspective")
    print("  ✓ Extension challenges")
    print("\n" + "=" * 80)

    # Run exercises
    df_gdp = exercise_1_basic_hp_filter()
    df_wages, breakpoint, F_stat, p_value = exercise_2_structural_breaks()
    df_investment, rolling_results = exercise_3_rolling_window_analysis()
    extension_challenges()

    print("\n" + "=" * 80)
    print("ALL EXERCISES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run this file: python 01_trend_extraction_exercises.py")
    print("2. Examine the generated visualizations")
    print("3. Try the extension challenges")
    print("4. Apply these methods to real economic data")
    print("\nData saved for further analysis:")
    print(f"  - GDP data: {len(df_gdp)} observations")
    print(f"  - Wage share data: {len(df_wages)} observations")
    print(f"  - Investment data: {len(df_investment)} observations")
    print("=" * 80 + "\n")
