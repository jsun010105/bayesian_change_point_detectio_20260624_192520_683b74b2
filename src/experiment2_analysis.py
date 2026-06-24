"""
Experiment 2 Analysis: Apply BOCPD to real LLM behavioral data.

Uses cached LLM responses and applies multiple BOCPD variants to detect
behavioral phase transitions. The key signals are:
  - Response length (continuous Gaussian): clearly shows phase shifts
  - Refusal binary (Beta-Bernoulli): also shows phase structure

This script tests the hypothesis that BOCPD detects latent phase shifts
masked by aggregate metrics.
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

random_seed = 42
np.random.seed(random_seed)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.bocpd_core import (
    NormalInverseGamma, run_bocpd, run_bocpd_bernoulli,
    cusum_test, moving_average_detector,
    permutation_test_kl, detection_metrics, kl_divergence_empirical
)


def load_data():
    cache_path = Path(__file__).parent.parent / 'results' / 'llm_responses_cache.json'
    with open(cache_path) as f:
        data = json.load(f)
    return data['results']


def extract_signals(results):
    """Extract behavioral signals for BOCPD analysis."""
    n = len(results)
    phases = np.array([r['phase'] for r in results])
    response_lengths = np.array([float(r['response_length']) for r in results])
    refusal_binary = np.array([float(r['refusal']) for r in results])

    # Compliance signal: 1 if long response (helpful), 0 if short/refusal
    # For harmful prompts: short/refusal = aligned; for safe prompts: long = aligned
    # We use raw response_length as the primary continuous signal
    # We use refusal_binary as the secondary binary signal

    return {
        'phases': phases,
        'response_lengths': response_lengths,
        'refusal_binary': refusal_binary,
        'n': n,
    }


def run_all_detectors(signals, phase_boundaries, hazard_rate=0.02):
    """Run BOCPD and baselines on multiple signals."""
    response_lengths = signals['response_lengths']
    refusal_binary = signals['refusal_binary']
    n = signals['n']

    # Signal 1: Response length (Gaussian BOCPD)
    # Use wide prior centered at moderate length; kappa=1, alpha=1 for uncertainty
    prior_length = NormalInverseGamma(mu0=100.0, kappa=1.0, alpha=1.0, beta=5000.0)
    R1, cp_probs_length, detected_length = run_bocpd(
        response_lengths, hazard_rate=hazard_rate, prior=prior_length
    )

    # Signal 2: Refusal binary (Beta-Bernoulli BOCPD)
    R2, cp_probs_refusal, detected_refusal = run_bocpd_bernoulli(
        refusal_binary, hazard_rate=hazard_rate, alpha0=1.0, beta0=1.0
    )

    # Signal 3: Normalized length (Gaussian BOCPD with normalized values)
    norm_lengths = (response_lengths - response_lengths.mean()) / (response_lengths.std() + 1e-10)
    prior_norm = NormalInverseGamma(mu0=0.0, kappa=1.0, alpha=1.0, beta=1.0)
    R3, cp_probs_norm, detected_norm = run_bocpd(
        norm_lengths, hazard_rate=hazard_rate, prior=prior_norm
    )

    # CUSUM on response lengths
    cusum_vals, detected_cusum = cusum_test(response_lengths, threshold=4.0)

    # Moving average on response lengths
    ma_vals, detected_ma = moving_average_detector(response_lengths, window=10, threshold_sigma=2.0)

    return {
        'response_length': {
            'cp_probs': cp_probs_length, 'detected': detected_length
        },
        'refusal_binary': {
            'cp_probs': cp_probs_refusal, 'detected': detected_refusal
        },
        'norm_length': {
            'cp_probs': cp_probs_norm, 'detected': detected_norm
        },
        'cusum': {
            'cusum_vals': cusum_vals, 'detected': detected_cusum
        },
        'moving_average': {
            'ma_vals': ma_vals, 'detected': detected_ma
        },
    }


def aggregate_metric_analysis(signals, phase_boundaries):
    """Show that aggregate metrics would give misleading results."""
    lengths = signals['response_lengths']
    n = signals['n']

    # What does aggregate analysis say?
    global_mean = lengths.mean()
    global_std = lengths.std()

    # An aggregate approach: compare first 80 steps vs last 80 steps
    first_half = lengths[:80]
    last_half = lengths[80:]
    delta = last_half.mean() - first_half.mean()
    t_stat, p_val = stats.ttest_ind(first_half, last_half)

    # Compare first 40 (aligned) vs steps 80-120 (misaligned)
    aligned = lengths[:40]
    misaligned = lengths[80:120]
    t_stat2, p_val2 = stats.ttest_ind(aligned, misaligned)

    # Phase means
    phase_means = {}
    boundaries = [0] + phase_boundaries + [n]
    for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        phase_means[f'phase_{i+1}'] = float(lengths[s:e].mean())

    return {
        'global_mean': float(global_mean),
        'global_std': float(global_std),
        'first_half_mean': float(first_half.mean()),
        'last_half_mean': float(last_half.mean()),
        'first_vs_last_delta': float(delta),
        'first_vs_last_ttest': {'t_stat': float(t_stat), 'p_value': float(p_val)},
        'aligned_vs_misaligned_ttest': {'t_stat': float(t_stat2), 'p_value': float(p_val2)},
        'phase_means': phase_means,
    }


def plot_results(signals, detectors, phase_boundaries, agg_metrics, model_name, outdir):
    """Comprehensive visualization of experiment 2 analysis."""
    lengths = signals['response_lengths']
    refusal = signals['refusal_binary']
    phases = signals['phases']
    n = signals['n']
    steps = np.arange(n)

    phase_colors = ['#2196F3', '#FF9800', '#F44336', '#4CAF50']
    phase_labels = ['Phase 1\n(Safe/Benign)', 'Phase 2\n(Borderline)',
                    'Phase 3\n(Harmful)', 'Phase 4\n(Recovery)']
    boundaries_full = [0] + phase_boundaries + [n]

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    true_cps = phase_boundaries

    # ─── Panel 1: Response length time series ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for i, (s, e) in enumerate(zip(boundaries_full[:-1], boundaries_full[1:])):
        ax1.axvspan(s, e, alpha=0.12, color=phase_colors[i])
        ax1.text((s+e)/2, 265, phase_labels[i], ha='center', va='bottom',
                 fontsize=8, color=phase_colors[i], fontweight='bold')

    ax1.scatter(steps, lengths, c=[phase_colors[int(p)-1] for p in phases],
                alpha=0.5, s=18, zorder=3)
    # Rolling mean of lengths
    ma_smooth = np.convolve(lengths, np.ones(10)/10, mode='same')
    ax1.plot(steps, ma_smooth, 'k-', linewidth=2, alpha=0.9, label='Rolling mean (w=10)')

    for cp in true_cps:
        ax1.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.7,
                    label='True phase boundary' if cp == true_cps[0] else '')

    detected_l = detectors['response_length']['detected']
    for d in detected_l:
        ax1.axvline(x=d, color='darkgreen', linestyle=':', linewidth=2.5, alpha=0.9,
                    label='BOCPD detected CP' if d == detected_l[0] else '')

    ax1.axhline(y=agg_metrics['global_mean'], color='purple', linestyle='-.',
                linewidth=1.5, alpha=0.6, label=f'Global mean ({agg_metrics["global_mean"]:.0f} words)')

    ax1.set_xlim(0, n)
    ax1.set_ylim(0, 270)
    ax1.set_xlabel('Query Step (proxy for training step)')
    ax1.set_ylabel('Response Length (words)')
    ax1.set_title(f'LLM Response Length as Behavioral Signal\n'
                  f'Model: {model_name} | 4 phases × 40 queries | '
                  f'Global mean ({agg_metrics["global_mean"]:.0f}) masks phase dynamics', fontsize=10)
    ax1.legend(loc='center right', fontsize=7)
    ax1.grid(True, alpha=0.3)

    # ─── Panel 2: BOCPD on response length ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    cp_probs_l = detectors['response_length']['cp_probs']
    ax2.plot(steps, cp_probs_l, 'b-', linewidth=1.5, label='BOCPD P(CP | length)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.2, label='Threshold 0.5')
    ax2.fill_between(steps, 0, cp_probs_l, alpha=0.2, color='blue')
    for cp in true_cps:
        ax2.axvline(x=cp, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    for d in detected_l:
        ax2.axvline(x=d, color='darkgreen', linestyle=':', linewidth=2, alpha=0.9)
    ax2.set_xlim(0, n)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('P(changepoint)')
    ax2.set_title('BOCPD on Response Length\n(Gaussian NIG likelihood)')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ─── Panel 3: BOCPD on refusal binary ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    cp_probs_r = detectors['refusal_binary']['cp_probs']
    detected_r = detectors['refusal_binary']['detected']
    ax3.plot(steps, cp_probs_r, 'g-', linewidth=1.5, label='BOCPD P(CP | refusal)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.2, label='Threshold 0.5')
    ax3.fill_between(steps, 0, cp_probs_r, alpha=0.2, color='green')
    for cp in true_cps:
        ax3.axvline(x=cp, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    for d in detected_r:
        ax3.axvline(x=d, color='darkgreen', linestyle=':', linewidth=2, alpha=0.9)
    ax3.set_xlim(0, n)
    ax3.set_ylim(-0.02, 1.05)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('P(changepoint)')
    ax3.set_title('BOCPD on Refusal Rate\n(Beta-Bernoulli likelihood)')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ─── Panel 4: CUSUM baseline ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    cusum_vals = detectors['cusum']['cusum_vals']
    detected_cusum = detectors['cusum']['detected']
    ax4.plot(steps, cusum_vals, 'purple', linewidth=1.5)
    ax4.axhline(y=4.0, color='orange', linestyle='--', linewidth=1.2, label='Threshold 4.0')
    for cp in true_cps:
        ax4.axvline(x=cp, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    for d in detected_cusum[:6]:
        ax4.axvline(x=d, color='green', linestyle=':', linewidth=1.5, alpha=0.6)
    ax4.set_xlim(0, n)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('CUSUM Statistic')
    ax4.set_title(f'CUSUM Baseline on Response Length\n'
                  f'({len(detected_cusum)} total detections — many false positives)')
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ─── Panel 5: Aggregate metrics vs phase-level breakdown ────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    phase_means_v = list(agg_metrics['phase_means'].values())
    x_pos = np.arange(4)
    bars = ax5.bar(x_pos, phase_means_v, color=phase_colors, alpha=0.8, width=0.6)
    ax5.axhline(y=agg_metrics['global_mean'], color='black', linestyle='--',
                linewidth=2.5, label=f'Global mean = {agg_metrics["global_mean"]:.0f}')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(['Phase 1\n(Safe)', 'Phase 2\n(Border)',
                          'Phase 3\n(Harmful)', 'Phase 4\n(Recovery)'])
    ax5.set_ylabel('Mean Response Length (words)')
    ax5.set_title('Aggregate vs Phase-Level Metrics\n'
                  'Global mean masks dramatic behavioral shifts')
    ax5.legend(fontsize=8)
    ax5.set_ylim(0, 270)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, phase_means_v):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f'{val:.0f}', ha='center', fontsize=10, fontweight='bold')

    # ─── Panel 6: KL divergences between phases ─────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    kl_vals = []
    kl_labels = []
    p_vals = []
    for i in range(len(boundaries_full) - 2):
        s1, e1 = boundaries_full[i], boundaries_full[i+1]
        s2, e2 = boundaries_full[i+1], boundaries_full[i+2]
        kl, pval = permutation_test_kl(lengths, boundaries_full[i+1], n_permutations=500)
        kl_vals.append(kl)
        kl_labels.append(f'Phase {i+1}→{i+2}')
        p_vals.append(pval)
        print(f"  {kl_labels[-1]}: KL={kl:.3f}, p={pval:.4f}")

    bar_colors = ['#4CAF50' if p < 0.05 else '#FF9800' for p in p_vals]
    ax6_bars = ax6.bar(kl_labels, kl_vals, color=bar_colors, alpha=0.85)
    ax6.set_ylabel('KL Divergence (response length distributions)')
    ax6.set_title('KL Divergence Between Adjacent Phase Distributions\n'
                  'Green = statistically significant (p<0.05), Orange = not significant')
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val, p in zip(ax6_bars, kl_vals, p_vals):
        sig_str = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'KL={val:.2f}\np={p:.3f} {sig_str}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Experiment 2: Real LLM Behavioral Phase Analysis\n'
                 f'BOCPD detects latent phase transitions masked by aggregate metrics',
                 fontsize=12, fontweight='bold', y=1.01)

    outpath = Path(outdir) / 'experiment2_analysis.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure: {outpath}")
    return kl_vals, p_vals, kl_labels


def main():
    outdir = Path(__file__).parent.parent / 'figures'
    results_dir = Path(__file__).parent.parent / 'results'
    model_name = "meta-llama/llama-3.1-8b-instruct"

    print("=" * 70)
    print("EXPERIMENT 2 ANALYSIS: BOCPD on Real LLM Behavioral Data")
    print("=" * 70)

    results = load_data()
    signals = extract_signals(results)
    n = signals['n']
    phase_boundaries = [40, 80, 120]
    true_cps = phase_boundaries

    print(f"\nLoaded {n} responses across 4 phases")
    print("\n--- Response Length by Phase ---")
    for phase in [1, 2, 3, 4]:
        phase_rs = [r for r in results if r['phase'] == phase]
        lengths = [r['response_length'] for r in phase_rs]
        refusals = [r['refusal'] for r in phase_rs]
        print(f"  Phase {phase}: mean_len={np.mean(lengths):.1f}, std={np.std(lengths):.1f}, "
              f"refusal_rate={np.mean(refusals):.3f}")

    print("\nRunning changepoint detectors...")
    detectors = run_all_detectors(signals, phase_boundaries)

    print("\n--- Detected Changepoints ---")
    print(f"  BOCPD (length):    {detectors['response_length']['detected']}")
    print(f"  BOCPD (refusal):   {detectors['refusal_binary']['detected']}")
    print(f"  BOCPD (norm_len):  {detectors['norm_length']['detected']}")
    print(f"  CUSUM:             {detectors['cusum']['detected'][:8]}")
    print(f"  MA+2σ:             {detectors['moving_average']['detected']}")
    print(f"  True boundaries:   {true_cps}")

    print("\n--- Detection Metrics (tolerance=8) ---")
    for name, det_key in [('BOCPD (length)', 'response_length'),
                            ('BOCPD (refusal)', 'refusal_binary'),
                            ('CUSUM', 'cusum'), ('MA+2σ', 'moving_average')]:
        detected = detectors[det_key]['detected']
        m = detection_metrics(detected, true_cps, tolerance=8)
        print(f"  {name:20s}: F1={m['f1']:.3f}, P={m['precision']:.3f}, "
              f"R={m['recall']:.3f}, FP={m['fp']}, detected={len(detected)}")

    print("\n--- Aggregate Metric Analysis ---")
    agg = aggregate_metrics_analysis(signals, phase_boundaries)
    print(f"  Global mean length: {agg['global_mean']:.1f} ± {agg['global_std']:.1f}")
    print(f"  Phase means: {agg['phase_means']}")
    print(f"  First half vs Last half: Δ={agg['first_vs_last_delta']:.1f}, "
          f"t={agg['first_vs_last_ttest']['t_stat']:.2f}, "
          f"p={agg['first_vs_last_ttest']['p_value']:.4f}")
    print(f"  Aligned vs Misaligned: t={agg['aligned_vs_misaligned_ttest']['t_stat']:.2f}, "
          f"p={agg['aligned_vs_misaligned_ttest']['p_value']:.4f}")
    print(f"  Note: Global mean ({agg['global_mean']:.0f}) masks phase means of "
          f"{[f\"{v:.0f}\" for v in agg['phase_means'].values()]}")

    print("\n--- KL Divergences Between Phases ---")
    kl_vals, p_vals, kl_labels = plot_results(
        signals, detectors, phase_boundaries, agg, model_name, outdir
    )

    # Permutation tests at BOCPD-detected changepoints
    print("\n--- Permutation Tests at BOCPD Detected CPs ---")
    kl_cp_results = []
    lengths = signals['response_lengths']
    for cp in detectors['response_length']['detected']:
        kl, pval = permutation_test_kl(lengths, cp, n_permutations=500)
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
        print(f"  CP at step {cp}: KL={kl:.4f}, p={pval:.4f} {sig}")
        kl_cp_results.append({'step': int(cp), 'kl': kl, 'p_value': pval, 'significant': pval < 0.05})

    # Save results
    def convert_json(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: convert_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_json(v) for v in obj]
        return obj

    results_out = {
        'model': model_name,
        'phase_boundaries': phase_boundaries,
        'detectors': {
            'bocpd_length': {
                'detected_cps': [int(d) for d in detectors['response_length']['detected']],
                'metrics': detection_metrics(detectors['response_length']['detected'],
                                              true_cps, tolerance=8),
                'signal': 'response_length_gaussian',
            },
            'bocpd_refusal': {
                'detected_cps': [int(d) for d in detectors['refusal_binary']['detected']],
                'metrics': detection_metrics(detectors['refusal_binary']['detected'],
                                              true_cps, tolerance=8),
                'signal': 'refusal_beta_bernoulli',
            },
            'cusum': {
                'detected_cps': [int(d) for d in detectors['cusum']['detected']],
                'metrics': detection_metrics(detectors['cusum']['detected'], true_cps, tolerance=8),
            },
            'moving_average': {
                'detected_cps': [int(d) for d in detectors['moving_average']['detected']],
                'metrics': detection_metrics(detectors['moving_average']['detected'],
                                              true_cps, tolerance=8),
            },
        },
        'aggregate_metrics': agg,
        'kl_between_phases': [
            {'transition': label, 'kl': kl, 'p_value': p}
            for label, kl, p in zip(kl_labels, kl_vals, p_vals)
        ],
        'kl_at_detected_cps': kl_cp_results,
    }

    with open(results_dir / 'experiment2_analysis_results.json', 'w') as f:
        json.dump(convert_json(results_out), f, indent=2)
    print(f"\nResults saved: {results_dir / 'experiment2_analysis_results.json'}")

    return results_out


if __name__ == '__main__':
    main()
