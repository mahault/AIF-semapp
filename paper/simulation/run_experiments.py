"""
run_experiments.py — Run all three experiments and generate publication figures.

Produces:
  figures/exp1_epistemic_foraging.pdf
  figures/exp2_multiwave_cooperation.pdf
  figures/exp3_nested_free_energy.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import os
import sys

from semfabric_sim import (
    run_experiment_1, run_experiment_2, run_experiment_3,
    GRID_N, N_CELLS, idx_to_rc, rc_to_idx, entropy_H,
    UNPROCESSED, PARTIAL, ENRICHED
)

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

COLORS = {
    'epistemic': '#2166AC',    # blue
    'pragmatic': '#D6604D',    # orange-red
    'shared': '#4393C3',       # light blue
    'diff_A': '#1A9850',       # green
    'diff_C': '#E6AB02',       # gold
    'diff_D': '#7570B3',       # purple
    'adversarial': '#D73027',  # red
    'fabric': '#762A83',       # purple
    'region': '#F1A340',       # amber
    'sign': '#999999',         # gray
}

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

T = 100
SEED = 42


# ============================================================
# Experiment 1 Figure: Four-panel
# ============================================================
def plot_experiment_1(results):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ep = results['epistemic']
    pr = results['pragmatic']
    t_range = np.arange(1, len(ep['fabric_F']) + 1)

    # (a) Domain belief entropy
    ax = axes[0, 0]
    ax.plot(t_range, ep['domain_entropy'], color=COLORS['epistemic'],
            label='Epistemic', linewidth=1.5)
    ax.plot(t_range, pr['domain_entropy'], color=COLORS['pragmatic'],
            label='Pragmatic', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('$H[q(\\mathrm{domain})]$')
    ax.set_title('(a) Domain belief entropy')
    ax.legend()
    ax.set_ylim(bottom=-0.05)

    # (b) Fabric free energy
    ax = axes[0, 1]
    ax.plot(t_range, ep['fabric_F'], color=COLORS['epistemic'],
            label='Epistemic', linewidth=1.5)
    ax.plot(t_range, pr['fabric_F'], color=COLORS['pragmatic'],
            label='Pragmatic', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('$\\mathcal{F}_{\\mathrm{material}}$')
    ax.set_title('(b) Material free energy')
    ax.legend()

    # (c) Trajectory heatmap (epistemic)
    ax = axes[1, 0]
    visit_count = np.zeros((GRID_N, GRID_N))
    for pos in ep['positions']:
        r, c = idx_to_rc(pos)
        visit_count[r, c] += 1
    im = ax.imshow(visit_count, cmap='Blues', interpolation='nearest')
    ax.set_title('(c) Visit frequency (epistemic)')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Visits')

    # (d) Enrichment spread over time
    ax = axes[1, 1]
    ax.plot(np.arange(len(ep['enrichment_count'])), ep['enrichment_count'],
            color=COLORS['epistemic'], label='Epistemic', linewidth=1.5)
    ax.plot(np.arange(len(pr['enrichment_count'])), pr['enrichment_count'],
            color=COLORS['pragmatic'], label='Pragmatic', linewidth=1.5, linestyle='--')
    ax.axhline(y=N_CELLS, color='gray', linestyle=':', alpha=0.5, label=f'Max ({N_CELLS})')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Enriched cells')
    ax.set_title('(d) Enrichment progress')
    ax.legend()

    fig.suptitle('Experiment 1: Single Wave Epistemic Foraging', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'exp1_epistemic_foraging.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Experiment 2 Figure: 2-row layout for 5 conditions
# ============================================================
def plot_experiment_2(results):
    conditions = ['shared', 'diff_A', 'diff_C', 'diff_D', 'adversarial']
    labels = ['Shared', 'Diff-A\n(expertise)', 'Diff-C\n(preference)',
              'Diff-D\n(prior)', 'Adversarial']
    short_labels = ['Shared', 'Diff-A', 'Diff-C', 'Diff-D', 'Adversarial']

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 10, hspace=0.40, wspace=0.8)

    # Top row: two panels spanning 5 columns each
    ax_F = fig.add_subplot(gs[0, :5])
    ax_E = fig.add_subplot(gs[0, 5:])

    # Bottom row: five heatmaps (2 columns each)
    axes_hm = [fig.add_subplot(gs[1, i*2:(i+1)*2]) for i in range(5)]

    # (a) F_material over time
    for cond, label in zip(conditions, short_labels):
        h = results[cond]
        t_range = np.arange(1, len(h['fabric_F']) + 1)
        ax_F.plot(t_range, h['fabric_F'], color=COLORS[cond],
                  label=label, linewidth=1.5)
    ax_F.set_xlabel('Timestep')
    ax_F.set_ylabel('$\\mathcal{F}_{\\mathrm{material}}$')
    ax_F.set_title('(a) Material free energy')
    ax_F.legend(fontsize=8)

    # (b) Enrichment count over time
    for cond, label in zip(conditions, short_labels):
        h = results[cond]
        ax_E.plot(np.arange(len(h['enrichment_count'])), h['enrichment_count'],
                  color=COLORS[cond], label=label, linewidth=1.5)
    ax_E.axhline(y=N_CELLS, color='gray', linestyle=':', alpha=0.5)
    ax_E.set_xlabel('Timestep')
    ax_E.set_ylabel('Enriched cells')
    ax_E.set_title('(b) Enrichment progress')
    ax_E.legend(fontsize=8)

    # Bottom row: final enrichment state heatmaps
    cmap = ListedColormap(['#f0f0f0', '#a6d96a', '#1a9641'])
    from matplotlib.patches import Patch

    for i, (cond, label) in enumerate(zip(conditions, labels)):
        ax = axes_hm[i]
        final_grid = results[cond]['fabric_states'][-1]
        im = ax.imshow(final_grid, cmap=cmap, vmin=0, vmax=2,
                       interpolation='nearest', aspect='equal')
        ax.set_title(label, fontsize=9, fontweight='bold')
        ax.set_xticks(range(GRID_N))
        ax.set_yticks(range(GRID_N))
        ax.tick_params(labelsize=7)

        # Show enrichment count in corner
        n_enr = results[cond]['enrichment_count'][-1]
        ax.text(0.02, 0.98, f'E={n_enr}', transform=ax.transAxes,
                fontsize=8, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Shared legend for enrichment states
    legend_elements = [
        Patch(facecolor='#f0f0f0', edgecolor='gray', label='Unprocessed'),
        Patch(facecolor='#a6d96a', edgecolor='gray', label='Partial'),
        Patch(facecolor='#1a9641', edgecolor='gray', label='Enriched'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9)

    fig.suptitle('Experiment 2: Isolating Generative Model Component Contributions',
                 fontsize=13, fontweight='bold')
    out = os.path.join(FIG_DIR, 'exp2_multiwave_cooperation.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Experiment 3 Figure: Three-panel
# ============================================================
def plot_experiment_3(results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    t_range = np.arange(1, len(results['fabric_F']) + 1)

    # (a) Sign-level free energy
    ax = axes[0]
    for i, s in enumerate(results['tracked']):
        r, c = idx_to_rc(s)
        ax.plot(t_range, results['sign_F'][s], alpha=0.6, linewidth=1.0,
                label=f'Sign ({r},{c})')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('$F_{\\mathrm{sign}}$')
    ax.set_title('(a) Sign-level free energy')
    ax.legend(fontsize=7, ncol=2)

    # (b) Regional free energy (2x2 blocks)
    ax = axes[1]
    region_colors = ['#D73027', '#FC8D59', '#91BFDB', '#4575B4']
    for i, r_key in enumerate(results['regions']):
        r0, c0 = r_key
        ax.plot(t_range, results['regional_F'][r_key], color=region_colors[i],
                linewidth=1.5, label=f'Region ({r0},{c0})')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('$F_{\\mathrm{region}}$')
    ax.set_title('(b) Regional free energy (2$\\times$2)')
    ax.legend(fontsize=8)

    # (c) Fabric-level free energy
    ax = axes[2]
    ax.plot(t_range, results['fabric_F'], color=COLORS['fabric'],
            linewidth=2.0, label='$\\mathcal{F}_{\\mathrm{material}}$')
    # Add mean of regional F for comparison
    mean_regional = np.mean(
        [results['regional_F'][r] for r in results['regions']], axis=0)
    ax.plot(t_range, mean_regional, color=COLORS['region'],
            linewidth=1.5, linestyle='--', label='Mean regional $F$')
    mean_sign = np.mean(
        [results['sign_F'][s] for s in results['tracked']], axis=0)
    ax.plot(t_range, mean_sign, color=COLORS['sign'],
            linewidth=1.0, linestyle=':', label='Mean sign $F$')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Free energy')
    ax.set_title('(c) Multi-scale free energy')
    ax.legend(fontsize=8)

    fig.suptitle('Experiment 3: Nested Free Energy Minimization Across Scales',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'exp3_nested_free_energy.pdf')
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"  Saved: {out}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Semiotic Fabric Simulation — Generating Publication Figures")
    print("=" * 60)
    print()

    print(f"Parameters: T={T}, seed={SEED}, grid={GRID_N}x{GRID_N}")
    print(f"Output: {os.path.abspath(FIG_DIR)}")
    print()

    print("Experiment 1: Single Wave Epistemic Foraging")
    r1 = run_experiment_1(T=T, seed=SEED)
    plot_experiment_1(r1)
    print(f"  Epistemic: F={r1['epistemic']['fabric_F'][-1]:.3f}, "
          f"enriched={r1['epistemic']['enrichment_count'][-1]}")
    print(f"  Pragmatic: F={r1['pragmatic']['fabric_F'][-1]:.3f}, "
          f"enriched={r1['pragmatic']['enrichment_count'][-1]}")
    print()

    print("Experiment 2: Multi-Wave Cooperation")
    r2 = run_experiment_2(T=T, seed=SEED)
    plot_experiment_2(r2)
    for c in ['shared', 'diff_A', 'diff_C', 'diff_D', 'adversarial']:
        print(f"  {c}: F={r2[c]['fabric_F'][-1]:.3f}, "
              f"enriched={r2[c]['enrichment_count'][-1]}, "
              f"coverage={r2[c]['coverage'][-1]:.2f}")
    print()

    print("Experiment 3: Nested Free Energy")
    r3 = run_experiment_3(T=T, seed=SEED)
    plot_experiment_3(r3)
    print(f"  Final fabric F: {r3['fabric_F'][-1]:.3f}")
    print()

    print("=" * 60)
    print("All figures saved successfully.")
    print("=" * 60)
