"""
Input-Output Analysis Exercises: Leontief & Sraffa Models
Heterodox Economics Focus

Exercises cover Input-Output economics from structuralist and
Sraffian perspectives. Essential for analyzing interdependencies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# EXERCISE 1: Leontief Input-Output Model - Basic Multipliers
# ============================================================================
# THEORY: Leontief I-O shows inter-industry linkages. Essential for
# understanding propagation of demand shocks and structural change.
# Reference: Leontief (1936, 1986), Miller & Blair (2009)
# ============================================================================

def exercise_1_leontief_multipliers():
    """
    Problem: Build simple 3-sector I-O table, calculate Leontief inverse,
    analyze multiplier effects of demand shocks.

    Reference: Leontief (1986), Miller & Blair (2009)
    """
    print("=" * 80)
    print("EXERCISE 1: Leontief Input-Output Model & Multipliers")
    print("=" * 80)

    # SOLUTION: Construct I-O table for 3-sector economy
    sectors = ['Agriculture', 'Manufacturing', 'Services']

    # Transactions table (intermediate inputs, $ millions)
    # Rows: Outputs FROM sector i
    # Columns: Inputs TO sector j
    transactions = np.array([
        [20, 100, 50],    # Agr sells to: Agr(20), Mfg(100), Svc(50)
        [50, 200, 300],   # Mfg sells to: Agr(50), Mfg(200), Svc(300)
        [30, 100, 100]    # Svc sells to: Agr(30), Mfg(100), Svc(100)
    ])

    # Final demand (consumption, investment, govt, exports - imports)
    final_demand = np.array([130, 350, 270])  # By sector

    # Total output (intermediate + final)
    total_output = transactions.sum(axis=1) + final_demand

    print("\nINPUT-OUTPUT TABLE:")
    print("=" * 80)
    io_table = pd.DataFrame(
        transactions,
        index=sectors,
        columns=sectors
    )
    io_table['Final Demand'] = final_demand
    io_table['Total Output'] = total_output
    print(io_table)

    # Calculate technical coefficients matrix (A matrix)
    # a_ij = input from i per unit output of j
    A = transactions / total_output[np.newaxis, :]

    print("\nTECHNICAL COEFFICIENTS MATRIX (A):")
    print("=" * 80)
    A_df = pd.DataFrame(A, index=sectors, columns=sectors)
    print(A_df.round(4))

    # Calculate Leontief inverse: L = (I - A)^(-1)
    I = np.eye(len(sectors))
    L = np.linalg.inv(I - A)

    print("\nLEONTIEF INVERSE MATRIX (I-A)^(-1):")
    print("=" * 80)
    L_df = pd.DataFrame(L, index=sectors, columns=sectors)
    print(L_df.round(4))

    print("\nINTERPRETATION:")
    print(f"Element L[i,j] = total output from sector i needed to deliver")
    print(f"one unit of final demand in sector j (direct + indirect)")

    # Output multipliers (column sums of L)
    output_multipliers = L.sum(axis=0)
    print("\nOUTPUT MULTIPLIERS (column sums):")
    for sector, mult in zip(sectors, output_multipliers):
        print(f"  {sector}: {mult:.3f}")
    print("Interpretation: $1 of final demand → $mult of total output (all sectors)")

    # Simulate demand shock: +$100M in manufacturing final demand
    demand_shock = np.array([0, 100, 0])
    output_change = L @ demand_shock

    print("\nDEMAND SHOCK SIMULATION:")
    print("=" * 80)
    print("Shock: +$100M final demand in Manufacturing")
    print("\nOutput changes by sector:")
    for sector, change in zip(sectors, output_change):
        print(f"  {sector}: +${change:.2f}M ({change/100:.1%} of shock)")
    print(f"\nTotal output change: +${output_change.sum():.2f}M")
    print(f"Multiplier effect: {output_change.sum()/100:.2f}x")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Input-Output Analysis: Leontief Model', fontsize=14, fontweight='bold')

    # Plot 1: Technical coefficients heatmap
    im1 = axes[0, 0].imshow(A, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_xticks(range(len(sectors)))
    axes[0, 0].set_yticks(range(len(sectors)))
    axes[0, 0].set_xticklabels(sectors, rotation=45, ha='right')
    axes[0, 0].set_yticklabels(sectors)
    axes[0, 0].set_title('Technical Coefficients Matrix (A)')
    axes[0, 0].set_xlabel('Purchasing Sector')
    axes[0, 0].set_ylabel('Supplying Sector')

    for i in range(len(sectors)):
        for j in range(len(sectors)):
            axes[0, 0].text(j, i, f'{A[i, j]:.3f}', ha='center', va='center')

    plt.colorbar(im1, ax=axes[0, 0], label='Coefficient Value')

    # Plot 2: Leontief inverse heatmap
    im2 = axes[0, 1].imshow(L, cmap='Blues', aspect='auto')
    axes[0, 1].set_xticks(range(len(sectors)))
    axes[0, 1].set_yticks(range(len(sectors)))
    axes[0, 1].set_xticklabels(sectors, rotation=45, ha='right')
    axes[0, 1].set_yticklabels(sectors)
    axes[0, 1].set_title('Leontief Inverse Matrix (I-A)^-1')
    axes[0, 1].set_xlabel('Final Demand Sector')
    axes[0, 1].set_ylabel('Output Sector')

    for i in range(len(sectors)):
        for j in range(len(sectors)):
            axes[0, 1].text(j, i, f'{L[i, j]:.3f}', ha='center', va='center')

    plt.colorbar(im2, ax=axes[0, 1], label='Multiplier Value')

    # Plot 3: Output multipliers bar chart
    axes[1, 0].bar(sectors, output_multipliers, color=['#8BC34A', '#FF9800', '#2196F3'],
                   alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Output Multiplier')
    axes[1, 0].set_title('Output Multipliers by Sector')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, (sector, mult) in enumerate(zip(sectors, output_multipliers)):
        axes[1, 0].text(i, mult + 0.05, f'{mult:.2f}', ha='center', fontweight='bold')

    # Plot 4: Demand shock propagation
    x_pos = np.arange(len(sectors))
    axes[1, 1].bar(x_pos, output_change, color=['#8BC34A', '#FF9800', '#2196F3'],
                   alpha=0.7, edgecolor='black')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(sectors)
    axes[1, 1].set_ylabel('Output Change ($M)')
    axes[1, 1].set_title('Impact of $100M Manufacturing Demand Shock')
    axes[1, 1].axhline(y=100, color='red', linestyle='--', label='Initial shock')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, change in enumerate(output_change):
        axes[1, 1].text(i, change + 2, f'${change:.1f}M', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('phase3-exercises/input_output/ex1_leontief_multipliers.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")

    print("\n" + "=" * 80)
    print("HETERODOX INTERPRETATION:")
    print("=" * 80)
    print("1. Structuralist Perspective:")
    print("   - Economy is web of interdependencies, not atomistic agents")
    print("   - Demand propagates through input-output linkages")
    print("   - Some sectors more 'central' (high multipliers)")
    print("\n2. Policy Implications:")
    print("   - Strategic sectors: High backward/forward linkages")
    print("   - Industrial policy: Support key nodes in I-O network")
    print("   - Multiplier effects justify fiscal stimulus")
    print("\n3. Sraffian Extension:")
    print("   - Can analyze price formation, not just quantities")
    print("   - Income distribution affects relative prices")
    print("   - Contradicts marginal productivity theory")

    return A, L, output_multipliers


# ============================================================================
# EXERCISE 2: Goodwin Growth Cycle - Class Struggle Model
# ============================================================================
# THEORY: Goodwin (1967) models cyclical dynamics from conflict between
# labor (wages) and capital (profits). Predator-prey dynamics.
# Reference: Goodwin (1967), Goodwin (1990)
# ============================================================================

def exercise_2_goodwin_cycle():
    """
    Problem: Simulate Goodwin's growth cycle model showing endogenous
    fluctuations in employment and wage share.

    Reference: Goodwin (1967) "A Growth Cycle"
    """
    print("\n" + "=" * 80)
    print("EXERCISE 2: Goodwin Growth Cycle - Class Struggle Dynamics")
    print("=" * 80)

    # Model parameters
    alpha = 0.05  # Labor productivity growth rate
    beta = 0.03   # Labor force growth rate
    nu = 3.0      # Output-capital ratio
    kappa = 0.2   # Worker bargaining power (Phillips curve slope)
    sigma = 0.8   # Capitalists' saving rate

    # Initial conditions
    v0 = 0.65     # Initial wage share (labor's share of output)
    u0 = 0.93     # Initial employment rate (1 - unemployment rate)

    # Time span
    T = 200       # Years
    dt = 0.1      # Time step

    # Goodwin equations (differential):
    # dv/dt = v * [kappa * (u - u*) - alpha]  (wage share dynamics)
    # du/dt = u * [sigma * nu * (1-v) - alpha - beta]  (employment dynamics)
    # where u* is 'natural' unemployment (set to 0 for simplicity)

    def goodwin_dynamics(state, t, params):
        v, u = state
        alpha, beta, nu, kappa, sigma = params

        dv_dt = v * (kappa * u - alpha)
        du_dt = u * (sigma * nu * (1 - v) - alpha - beta)

        return [dv_dt, du_dt]

    # Simulate using Euler method
    time = np.arange(0, T, dt)
    n_steps = len(time)

    v = np.zeros(n_steps)
    u = np.zeros(n_steps)
    v[0] = v0
    u[0] = u0

    params = (alpha, beta, nu, kappa, sigma)

    for i in range(1, n_steps):
        dv_dt, du_dt = goodwin_dynamics([v[i-1], u[i-1]], time[i-1], params)
        v[i] = v[i-1] + dv_dt * dt
        u[i] = u[i-1] + du_dt * dt

        # Bound values (economic constraints)
        v[i] = np.clip(v[i], 0.3, 0.95)
        u[i] = np.clip(u[i], 0.5, 0.999)

    # Calculate implied variables
    profit_share = 1 - v
    unemployment_rate = 1 - u
    growth_rate = sigma * nu * profit_share  # g = s * r

    print("\nGOODWIN MODEL PARAMETERS:")
    print("=" * 80)
    print(f"Labor productivity growth (α): {alpha:.1%}")
    print(f"Labor force growth (β): {beta:.1%}")
    print(f"Output-capital ratio (ν): {nu:.2f}")
    print(f"Bargaining power (κ): {kappa:.2f}")
    print(f"Capitalists' saving rate (σ): {sigma:.1%}")

    print("\nCYCLE CHARACTERISTICS:")
    print("=" * 80)
    # Find period by autocorrelation
    from scipy.signal import find_peaks
    peaks_v, _ = find_peaks(v, distance=20)
    if len(peaks_v) > 1:
        periods_v = np.diff(time[peaks_v])
        avg_period = np.mean(periods_v)
        print(f"Average cycle period: {avg_period:.1f} years")
        print(f"Number of cycles: {len(peaks_v) - 1:.0f}")

    print(f"\nWage share range: {v.min():.1%} - {v.max():.1%}")
    print(f"Employment rate range: {u.min():.1%} - {u.max():.1%}")
    print(f"Implied unemployment: {(1-u.min())*100:.1f}% - {(1-u.max())*100:.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Goodwin Growth Cycle: Class Struggle Dynamics",
                 fontsize=14, fontweight='bold')

    # Plot 1: Phase diagram (limit cycle)
    axes[0, 0].plot(u, v, linewidth=1.5, color='#2196F3')
    axes[0, 0].plot(u[0], v[0], 'go', markersize=12, label='Start', zorder=5)
    axes[0, 0].plot(u[-1], v[-1], 'ro', markersize=12, label='End', zorder=5)

    # Add arrows to show direction
    arrow_points = np.arange(0, len(u), len(u)//10)
    for idx in arrow_points[:-1]:
        axes[0, 0].annotate('', xy=(u[idx+50], v[idx+50]), xytext=(u[idx], v[idx]),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    axes[0, 0].set_xlabel('Employment Rate (u)')
    axes[0, 0].set_ylabel('Wage Share (v)')
    axes[0, 0].set_title('Phase Diagram: Limit Cycle')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Time series - wage vs profit share
    axes[0, 1].plot(time, v, label='Wage Share', linewidth=2, color='#2196F3')
    axes[0, 1].plot(time, profit_share, label='Profit Share', linewidth=2, color='#F44336')
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Time (years)')
    axes[0, 1].set_ylabel('Share of Output')
    axes[0, 1].set_title('Functional Income Distribution Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])

    # Plot 3: Employment rate and growth
    ax3a = axes[1, 0]
    ax3b = ax3a.twinx()

    ax3a.plot(time, u * 100, label='Employment Rate', linewidth=2, color='#4CAF50')
    ax3b.plot(time, growth_rate * 100, label='Growth Rate', linewidth=2,
             color='#FF9800', linestyle='--')

    ax3a.set_xlabel('Time (years)')
    ax3a.set_ylabel('Employment Rate (%)', color='#4CAF50')
    ax3b.set_ylabel('Growth Rate (%)', color='#FF9800')
    ax3a.set_title('Employment and Growth Dynamics')
    ax3a.grid(True, alpha=0.3)

    lines1, labels1 = ax3a.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot 4: Quadrant analysis
    # Divide cycle into four phases
    v_center = np.median(v)
    u_center = np.median(u)

    colors_quad = np.where(v > v_center,
                          np.where(u > u_center, '#4CAF50', '#2196F3'),  # High v
                          np.where(u > u_center, '#FF9800', '#F44336'))  # Low v

    axes[1, 1].scatter(u, v, c=colors_quad, s=2, alpha=0.5)
    axes[1, 1].axvline(x=u_center, color='black', linestyle='--', linewidth=1)
    axes[1, 1].axhline(y=v_center, color='black', linestyle='--', linewidth=1)

    # Label quadrants
    axes[1, 1].text(u.max()*0.98, v.max()*0.98, 'I: High w, High u\n↑ Inflation',
                   ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='#4CAF50', alpha=0.5))
    axes[1, 1].text(u.min()*1.02, v.max()*0.98, 'II: High w, Low u\n↓ Profit, ↓ Invest',
                   ha='left', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='#2196F3', alpha=0.5))
    axes[1, 1].text(u.min()*1.02, v.min()*1.02, 'III: Low w, Low u\n↑ Profit',
                   ha='left', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='#F44336', alpha=0.5))
    axes[1, 1].text(u.max()*0.98, v.min()*1.02, 'IV: Low w, High u\n↑ Wage pressure',
                   ha='right', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='#FF9800', alpha=0.5))

    axes[1, 1].set_xlabel('Employment Rate (u)')
    axes[1, 1].set_ylabel('Wage Share (v)')
    axes[1, 1].set_title('Cycle Phases (Goodwin Quadrants)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('phase3-exercises/goodwin_cycles/ex2_goodwin_cycle.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved")

    print("\n" + "=" * 80)
    print("ECONOMIC INTERPRETATION - GOODWIN CYCLE:")
    print("=" * 80)
    print("\nPhase I (High employment, High wage share):")
    print("  - Workers have bargaining power → wages rise")
    print("  - Profit share squeezed → Investment slows")
    print("  - Employment peaks → Transition to Phase II")

    print("\nPhase II (Low employment falling, High wage share):")
    print("  - Low profits → Weak investment → Employment falls")
    print("  - Unemployment rises → Worker power weakens")
    print("  - Wage share begins to fall → Transition to Phase III")

    print("\nPhase III (Low employment, Low wage share):")
    print("  - High profit share → Investment recovers")
    print("  - But employment still low → Wages suppressed")
    print("  - Profits fuel accumulation → Transition to Phase IV")

    print("\nPhase IV (High employment, Low wage share):")
    print("  - High profits → Strong investment → Employment rises")
    print("  - Labor market tightens → Workers gain leverage")
    print("  - Wages start rising → Back to Phase I")

    print("\nKEY INSIGHTS:")
    print("1. Endogenous cycles from class conflict (not external shocks)")
    print("2. No stable equilibrium - perpetual oscillation")
    print("3. Distribution and employment co-evolve")
    print("4. 'Reserve army of unemployed' regulates wages (Marx)")
    print("5. Predator-prey math (wolves-rabbits) applied to economics")

    print("\nPOLICY IMPLICATIONS:")
    print("- Phillips curve emerges endogenously")
    print("- No 'natural rate' of unemployment (NAIRU critique)")
    print("- Policies affecting distribution alter cycle dynamics")
    print("- Relevance: Profit-led vs wage-led growth debate")

    return time, v, u, profit_share


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INPUT-OUTPUT & GOODWIN CYCLE EXERCISES")
    print("=" * 80)
    print("\nExercises:")
    print("1. Leontief Input-Output multipliers")
    print("2. Goodwin growth cycle (class struggle)")
    print("=" * 80)

    # Run exercises
    A, L, multipliers = exercise_1_leontief_multipliers()
    time, wage_share, employment, profit_share = exercise_2_goodwin_cycle()

    print("\n" + "=" * 80)
    print("ALL EXERCISES COMPLETED!")
    print("=" * 80)
