"""
Experiment 1: BOCPD Validation on Synthetic LLM Safety Series.

Uses pre-generated synthetic safety rate time series with known ground-truth
changepoints at steps 50, 100, 150.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Add repo root to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bocpd_core import (
    BocpdGaussian, NormalInverseGamma, run_bocpd,
    cusum_test, moving_average_detector,
    permutation_test_kl, detection_metrics, kl_divergence_empirical
)

import random
random.seed(42)
np.random.seed(42)


def load_synthetic_data():
    data_path = Path(__file__).parent.parent / 'datasets' / 'welllog' / 'synthetic_llm_safety_series.json'
    with open(data_path) as f:
        d = json.load(f)
    series = np.array(d['safety_rate_sequence'])
    true_cps = d['true_phase_boundaries']
    phases = d['phases']
    print(f"Loaded series: {len(series)} steps, phases={phases}")
    print(f"True changepoints: {true_cps}")
    phase_means_str = [f"{series[s:e].mean():.3f}" for s, e in zip([0]+true_cps, true_cps+[len(series)])]
    print(f"Phase means: {phase_means_str}")
    return series, true_cps, phases


def run_ruptures_pelt(series: np.ndarray, n_bkps: int = 3):
    """Run ruptures PELT offline CPD as a baseline."""
    try:
        import ruptures as rpt
        model = rpt.Pelt(model='rbf', min_size=5, jump=1)
        model.fit(series.reshape(-1, 1))
        result = model.predict(pen=1.0)
        # result contains end of each segment, convert to changepoints
        bkps = [r - 1 for r in result[:-1]]
        return bkps
    except Exception as e:
        print(f"ruptures PELT failed: {e}")
        return []


def aggregate_metrics_analysis(series: np.ndarray, true_cps: list):
    """Compute aggregate metrics that might 'miss' changepoints."""
    n = len(series)

    # Global mean and std
    global_mean = np.mean(series)
    global_std = np.std(series)

    # Mean per phase (with true boundaries)
    phase_means = {}
    boundaries = [0] + true_cps + [n]
    for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        phase_means[f'phase_{i+1}'] = np.mean(series[s:e])

    # Test if 'null result': is global mean change across first/last 50 steps significant?
    first_half = series[:50]
    last_half = series[150:]
    delta = np.mean(last_half) - np.mean(first_half)
    pooled_se = np.sqrt(np.var(first_half)/len(first_half) + np.var(last_half)/len(last_half))
    z_score = delta / pooled_se if pooled_se > 0 else 0

    # What about comparing pre-misalignment (steps 0-50) vs misalignment (steps 100-150)?
    pre = series[:50]
    mis = series[100:150]
    delta_mis = np.mean(mis) - np.mean(pre)
    pooled_se_mis = np.sqrt(np.var(pre)/len(pre) + np.var(mis)/len(mis))
    z_mis = delta_mis / pooled_se_mis if pooled_se_mis > 0 else 0

    return {
        'global_mean': global_mean,
        'global_std': global_std,
        'phase_means': phase_means,
        'first_vs_last_delta': delta,
        'first_vs_last_zscore': z_score,
        'aligned_vs_misaligned_delta': delta_mis,
        'aligned_vs_misaligned_zscore': z_mis,
    }


def plot_experiment1(series, true_cps, cp_probs_bocpd, detected_bocpd,
                     cusum_vals, detected_cusum,
                     ma_vals, detected_ma,
                     pelt_bkps, phases, outdir):
    """Create comprehensive visualization for Experiment 1."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    n = len(series)
    steps = np.arange(n)
    colors = ['#2196F3', '#FF9800', '#F44336', '#4CAF50']

    # Panel 1: Safety rate time series with true changepoints
    ax1 = fig.add_subplot(gs[0, :])
    boundaries = [0] + true_cps + [n]
    for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax1.axvspan(s, e, alpha=0.15, color=colors[i % len(colors)], label=phases[i] if i < len(phases) else f'Phase {i+1}')
    ax1.plot(steps, series, 'k-', linewidth=0.8, alpha=0.8, label='Safety rate')
    # Moving average
    window = 10
    ma_smooth = np.convolve(series, np.ones(window)/window, mode='same')
    ax1.plot(steps, ma_smooth, 'b--', linewidth=1.5, alpha=0.7, label=f'MA({window})')
    for cp in true_cps:
        ax1.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Safety Rate')
    ax1.set_title('Synthetic LLM Safety Rate Time Series (True CPs: red dashed)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: BOCPD changepoint probability
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(steps, cp_probs_bocpd, 'b-', linewidth=1.5, label='BOCPD CP prob')
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.8, label='Threshold (0.5)')
    for cp in true_cps:
        ax2.axvline(x=cp, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='True CP' if cp == true_cps[0] else '')
    for d in detected_bocpd:
        ax2.axvline(x=d, color='green', linestyle=':', linewidth=1.5, alpha=0.8, label='Detected CP' if d == detected_bocpd[0] else '')
    ax2.set_xlim(0, n)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('P(changepoint)')
    ax2.set_title('BOCPD: Changepoint Probability')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: CUSUM
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(steps, cusum_vals, 'purple', linewidth=1.5, label='CUSUM statistic')
    ax3.axhline(y=5.0, color='orange', linestyle='--', linewidth=1, label='Threshold (5.0)')
    for cp in true_cps:
        ax3.axvline(x=cp, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    for d in detected_cusum:
        ax3.axvline(x=d, color='green', linestyle=':', linewidth=1.5, alpha=0.8)
    ax3.set_xlim(0, n)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('CUSUM Statistic')
    ax3.set_title('CUSUM Sequential Test')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Comparison table (text plot)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    # Compute metrics
    metrics_bocpd = detection_metrics(detected_bocpd, true_cps, tolerance=5)
    metrics_cusum = detection_metrics(detected_cusum, true_cps, tolerance=5)
    metrics_ma = detection_metrics(detected_ma, true_cps, tolerance=5)
    metrics_pelt = detection_metrics(pelt_bkps, true_cps, tolerance=5)

    table_data = [
        ['Method', 'Detected CPs', 'F1', 'Precision', 'Recall', 'Avg Delay'],
        ['BOCPD', str(detected_bocpd[:5]), f"{metrics_bocpd['f1']:.2f}",
         f"{metrics_bocpd['precision']:.2f}", f"{metrics_bocpd['recall']:.2f}",
         f"{metrics_bocpd['avg_delay']:.1f}" if metrics_bocpd['avg_delay'] != float('inf') else 'N/A'],
        ['CUSUM', str(detected_cusum[:5]), f"{metrics_cusum['f1']:.2f}",
         f"{metrics_cusum['precision']:.2f}", f"{metrics_cusum['recall']:.2f}",
         f"{metrics_cusum['avg_delay']:.1f}" if metrics_cusum['avg_delay'] != float('inf') else 'N/A'],
        ['MA+2σ', str(detected_ma[:5]), f"{metrics_ma['f1']:.2f}",
         f"{metrics_ma['precision']:.2f}", f"{metrics_ma['recall']:.2f}",
         f"{metrics_ma['avg_delay']:.1f}" if metrics_ma['avg_delay'] != float('inf') else 'N/A'],
        ['PELT', str(pelt_bkps[:5]), f"{metrics_pelt['f1']:.2f}",
         f"{metrics_pelt['precision']:.2f}", f"{metrics_pelt['recall']:.2f}",
         f"{metrics_pelt['avg_delay']:.1f}" if metrics_pelt['avg_delay'] != float('inf') else 'N/A'],
    ]

    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                       loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    ax4.set_title('Detection Performance Comparison', fontsize=10, pad=15)

    # Panel 5: Phase KL divergences
    ax5 = fig.add_subplot(gs[2, 1])
    boundaries = [0] + true_cps + [n]
    kl_labels = []
    kl_vals = []
    for i in range(len(boundaries) - 2):
        s1, e1 = boundaries[i], boundaries[i+1]
        s2, e2 = boundaries[i+1], boundaries[i+2]
        kl = kl_divergence_empirical(series[s1:e1], series[s2:e2])
        kl_labels.append(f'Phase{i+1}→{i+2}')
        kl_vals.append(kl)

    bars = ax5.bar(kl_labels, kl_vals, color=['#FF9800', '#F44336', '#4CAF50'], alpha=0.8)
    ax5.set_ylabel('KL Divergence')
    ax5.set_title('KL Divergence Between Adjacent Phases')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, kl_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Experiment 1: BOCPD vs. Baselines on Synthetic LLM Safety Series',
                  fontsize=13, fontweight='bold', y=0.98)

    outpath = Path(outdir) / 'experiment1_synthetic.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {outpath}")
    return metrics_bocpd, metrics_cusum, metrics_ma, metrics_pelt


def main():
    outdir = Path(__file__).parent.parent / 'figures'
    outdir.mkdir(exist_ok=True)
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 1: BOCPD Validation on Synthetic Safety Series")
    print("=" * 60)

    series, true_cps, phases = load_synthetic_data()

    # Configure BOCPD: hazard_rate = 0.02 (expect CP every ~50 steps)
    # Wide prior (kappa=1, beta=0.05): uncertain about mean and variance before observations.
    # Using mu0=0.5 (neutral center for [0,1] safety rates) with weak precision.
    # This ensures P(x_t | fresh prior) is non-negligible when distribution shifts.
    prior = NormalInverseGamma(mu0=0.5, kappa=1.0, alpha=1.0, beta=0.05)
    hazard_rate = 0.02  # 1 expected changepoint per 50 steps

    print(f"\nRunning BOCPD (hazard_rate={hazard_rate:.3f})...")
    R_list, cp_probs, detected_bocpd = run_bocpd(series, hazard_rate=hazard_rate, prior=prior)
    print(f"BOCPD detected changepoints at steps: {detected_bocpd}")

    print("\nRunning CUSUM baseline...")
    cusum_vals, detected_cusum = cusum_test(series, threshold=4.0)
    print(f"CUSUM detected: {detected_cusum}")

    print("\nRunning Moving Average baseline...")
    ma_vals, detected_ma = moving_average_detector(series, window=10, threshold_sigma=2.0)
    print(f"MA+2σ detected: {detected_ma}")

    print("\nRunning ruptures PELT offline CPD...")
    pelt_bkps = run_ruptures_pelt(series, n_bkps=3)
    print(f"PELT detected: {pelt_bkps}")

    print("\n--- Detection Metrics ---")
    for name, detected in [('BOCPD', detected_bocpd), ('CUSUM', detected_cusum),
                             ('MA+2σ', detected_ma), ('PELT', pelt_bkps)]:
        m = detection_metrics(detected, true_cps, tolerance=5)
        print(f"{name:10s}: F1={m['f1']:.3f}, P={m['precision']:.3f}, R={m['recall']:.3f}, "
              f"delay={m['avg_delay']:.1f}, FP={m['fp']}")

    print("\n--- Aggregate Metrics Analysis ---")
    agg = aggregate_metrics_analysis(series, true_cps)
    print(f"Global mean: {agg['global_mean']:.4f} ± {agg['global_std']:.4f}")
    print(f"Phase means: {agg['phase_means']}")
    print(f"First 50 vs Last 50 steps: Δ={agg['first_vs_last_delta']:.4f}, z={agg['first_vs_last_zscore']:.2f}")
    print(f"Aligned vs Misaligned: Δ={agg['aligned_vs_misaligned_delta']:.4f}, z={agg['aligned_vs_misaligned_zscore']:.2f}")

    print("\n--- KL Divergence & Permutation Tests ---")
    boundaries = [0] + true_cps + [len(series)]
    kl_results = []
    for i in range(len(boundaries) - 2):
        s1, e1 = boundaries[i], boundaries[i+1]
        s2, e2 = boundaries[i+1], boundaries[i+2]
        kl, pval = permutation_test_kl(series, boundaries[i+1], n_permutations=1000)
        print(f"Phase {i+1}→{i+2}: KL={kl:.4f}, p={pval:.4f} ({'significant' if pval < 0.05 else 'not significant'})")
        kl_results.append({'transition': f'phase_{i+1}_to_{i+2}', 'kl': kl, 'p_value': pval})

    print("\n--- Generating Visualization ---")
    metrics_bocpd, metrics_cusum, metrics_ma, metrics_pelt = plot_experiment1(
        series, true_cps, cp_probs, detected_bocpd,
        cusum_vals, detected_cusum, ma_vals, detected_ma,
        pelt_bkps, phases, outdir
    )

    # Save results
    import json as jsonlib
    results = {
        'experiment': 'synthetic_validation',
        'series_length': len(series),
        'true_changepoints': true_cps,
        'phases': phases,
        'methods': {
            'bocpd': {
                'detected_cps': detected_bocpd,
                'metrics': metrics_bocpd,
                'hazard_rate': hazard_rate,
            },
            'cusum': {
                'detected_cps': detected_cusum,
                'metrics': metrics_cusum,
            },
            'moving_average': {
                'detected_cps': detected_ma,
                'metrics': metrics_ma,
            },
            'ruptures_pelt': {
                'detected_cps': pelt_bkps,
                'metrics': metrics_pelt,
            }
        },
        'aggregate_metrics': agg,
        'kl_divergence_tests': kl_results,
    }

    def convert_for_json(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_for_json(v) for v in obj]
        return obj

    with open(results_dir / 'experiment1_results.json', 'w') as f:
        jsonlib.dump(convert_for_json(results), f, indent=2)
    print(f"Results saved to: {results_dir / 'experiment1_results.json'}")

    return results


if __name__ == '__main__':
    main()
