"""
run_experiments.py — Generate figures for the Semiotic Fabric paper
==================================================================

Three experiments using pymdp-based Process Waves on a 5×5 semiotic grid:
  Experiment 1 : Single-Wave epistemic foraging with learning
  Experiment 2 : Multi-Wave cooperation (isolating A, C, D components)
  Experiment 3 : Nested free energy across Markov blanket scales
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from semfabric_sim import (
    SemFabricEnv, Wave, fabric_free_energy, sign_free_energy,
    regional_free_energy, GRID_N, N_CELLS, MOVE_NAMES,
    GEOINT, SIGINT, OSINT, DOMAIN_NAMES,
    UNPROCESSED, PARTIAL, ENRICHED,
    idx_to_rc, rc_to_idx, entropy_H,
)

FIG_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures'))
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Simulation loop helper
# ============================================================
def run_wave(env, wave, pos, T=100, learning_interval=20):
    """Run a single Wave for T steps.  Returns metrics dict."""
    metrics = dict(
        fabric_F=[], domain_entropy=[], enrichment_count=[],
        positions=[], fabric_states=[],
    )
    obs = env.observe(pos)

    for t in range(T):
        # Perception + local action selection (observe / enrich)
        enrich_action, qs, q_pi, efe = wave.infer_and_act(pos, obs)

        # Belief propagation to neighbors
        wave.update_neighbor_beliefs(pos, obs[1])
        wave.propagate_domain_belief(pos, obs[2])

        # Spatial movement selection
        move, target = wave.select_movement(pos)

        # Execute in environment
        dom_belief = wave.cell_beliefs_dom[target]
        new_pos, new_obs, did_enrich = env.step(pos, move, enrich_action, dom_belief)
        wave.record_enrichment(new_pos, did_enrich)

        pos = new_pos
        obs = new_obs

        # Online learning
        if wave.learn and (t + 1) % learning_interval == 0:
            wave.do_learning_update()
            wave.reset_history()

        # Record metrics
        metrics['fabric_F'].append(fabric_free_energy(env))
        mean_ent = np.mean([entropy_H(wave.cell_beliefs_dom[k]) for k in range(N_CELLS)])
        metrics['domain_entropy'].append(mean_ent)
        metrics['enrichment_count'].append(env.n_enriched())
        metrics['positions'].append(pos)
        metrics['fabric_states'].append(env.grid())

    return metrics, pos


def run_two_waves(env, wa, wb, pos_a, pos_b, T=100, learning_interval=20):
    """Run two Waves on the same environment for T steps."""
    metrics = dict(
        fabric_F=[], enrichment_count=[], coverage=[],
        positions_a=[], positions_b=[], fabric_states=[],
    )
    obs_a = env.observe(pos_a)
    obs_b = env.observe(pos_b)

    for t in range(T):
        # Wave A acts
        ea_a, _, _, _ = wa.infer_and_act(pos_a, obs_a)
        wa.update_neighbor_beliefs(pos_a, obs_a[1])
        wa.propagate_domain_belief(pos_a, obs_a[2])
        move_a, tgt_a = wa.select_movement(pos_a)
        dom_a = wa.cell_beliefs_dom[tgt_a]
        pos_a, obs_a, did_a = env.step(pos_a, move_a, ea_a, dom_a)
        wa.record_enrichment(pos_a, did_a)

        # Wave B acts
        ea_b, _, _, _ = wb.infer_and_act(pos_b, obs_b)
        wb.update_neighbor_beliefs(pos_b, obs_b[1])
        wb.propagate_domain_belief(pos_b, obs_b[2])
        move_b, tgt_b = wb.select_movement(pos_b)
        dom_b = wb.cell_beliefs_dom[tgt_b]
        pos_b, obs_b, did_b = env.step(pos_b, move_b, ea_b, dom_b)
        wb.record_enrichment(pos_b, did_b)

        # Cross-wave stigmergic coupling
        if did_a:
            from semfabric_sim import neighbors, PARTIAL
            for n in neighbors(pos_a):
                wb.cell_beliefs_sem[n, PARTIAL:] *= 1.1
                wb.cell_beliefs_sem[n] /= wb.cell_beliefs_sem[n].sum()
        if did_b:
            from semfabric_sim import neighbors, PARTIAL
            for n in neighbors(pos_b):
                wa.cell_beliefs_sem[n, PARTIAL:] *= 1.1
                wa.cell_beliefs_sem[n] /= wa.cell_beliefs_sem[n].sum()

        # Learning
        if wa.learn and (t + 1) % learning_interval == 0:
            wa.do_learning_update(); wa.reset_history()
        if wb.learn and (t + 1) % learning_interval == 0:
            wb.do_learning_update(); wb.reset_history()

        # Record
        metrics['fabric_F'].append(fabric_free_energy(env))
        metrics['enrichment_count'].append(env.n_enriched())
        metrics['coverage'].append(env.n_processed() / N_CELLS)
        metrics['positions_a'].append(pos_a)
        metrics['positions_b'].append(pos_b)
        metrics['fabric_states'].append(env.grid())

    return metrics


# ============================================================
# Experiment 1 : Single Wave Epistemic Foraging + Learning
# ============================================================
def experiment_1():
    print("=== Experiment 1: Single Wave Epistemic Foraging ===")
    T = 150
    seed = 42
    start = rc_to_idx(2, 2)

    results = {}
    for label, epistemic in [('epistemic', True), ('pragmatic', False)]:
        env = SemFabricEnv(seed=seed)
        wave = Wave(epistemic=epistemic, seed=seed, learn=True, gamma=8.0)
        m, _ = run_wave(env, wave, start, T=T)
        results[label] = m
        print(f"  {label}: enriched={m['enrichment_count'][-1]}, "
              f"F_final={m['fabric_F'][-1]:.3f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ts = np.arange(T)

    # (a) Domain belief entropy
    ax = axes[0, 0]
    ax.plot(ts, results['epistemic']['domain_entropy'], 'b-', label='Epistemic (full EFE)', linewidth=1.5)
    ax.plot(ts, results['pragmatic']['domain_entropy'], 'r--', label='Pragmatic only', linewidth=1.5)
    ax.set_xlabel('Timestep'); ax.set_ylabel('Mean domain entropy')
    ax.set_title('(a) Domain belief convergence'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) Fabric-level VFE
    ax = axes[0, 1]
    ax.plot(ts, results['epistemic']['fabric_F'], 'b-', label='Epistemic', linewidth=1.5)
    ax.plot(ts, results['pragmatic']['fabric_F'], 'r--', label='Pragmatic', linewidth=1.5)
    ax.set_xlabel('Timestep'); ax.set_ylabel(r'$\mathcal{F}_{\mathrm{fabric}}$')
    ax.set_title(r'(b) Fabric-level VFE $\mathcal{F}_{\mathrm{fabric}}$'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) Visit heatmap
    ax = axes[1, 0]
    visits = np.zeros((GRID_N, GRID_N))
    for p in results['epistemic']['positions']:
        r, c = idx_to_rc(p)
        visits[r, c] += 1
    im = ax.imshow(visits, cmap='YlOrRd', interpolation='nearest')
    ax.set_title('(c) Visit frequency (epistemic)'); plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(GRID_N)); ax.set_yticks(range(GRID_N))

    # (d) Enrichment progress
    ax = axes[1, 1]
    ax.plot(ts, results['epistemic']['enrichment_count'], 'b-', label='Epistemic', linewidth=1.5)
    ax.plot(ts, results['pragmatic']['enrichment_count'], 'r--', label='Pragmatic', linewidth=1.5)
    ax.set_xlabel('Timestep'); ax.set_ylabel('Enriched cells')
    ax.set_title('(d) Enrichment progress'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'exp1_epistemic_foraging.{fmt}'), dpi=200)
    plt.close(fig)
    print("  Figure saved.")
    return results


# ============================================================
# Experiment 2 : Multi-Wave Cooperation
# ============================================================
def experiment_2():
    print("=== Experiment 2: Multi-Wave Cooperation ===")
    T = 100
    seed = 42
    pos_a_start = rc_to_idx(0, 0)
    pos_b_start = rc_to_idx(4, 4)

    conditions = {}

    # 1. Shared — identical generative models
    env = SemFabricEnv(heterogeneous=True, seed=seed)
    wa = Wave(seed=seed, learn=True)
    wb = Wave(seed=seed + 1, learn=True)
    conditions['shared'] = run_two_waves(env, wa, wb, pos_a_start, pos_b_start, T)

    # 2. Diff-A — perceptual expertise
    env = SemFabricEnv(heterogeneous=True, seed=seed)
    wa = Wave(seed=seed, expertise_dom=GEOINT, learn=True)
    wb = Wave(seed=seed + 1, expertise_dom=SIGINT, learn=True)
    conditions['diff_A'] = run_two_waves(env, wa, wb, pos_a_start, pos_b_start, T)

    # 3. Diff-C — preference differentiation
    env = SemFabricEnv(heterogeneous=True, seed=seed)
    wa = Wave(seed=seed, preference_dom=GEOINT, learn=True)
    wb = Wave(seed=seed + 1, preference_dom=SIGINT, learn=True)
    conditions['diff_C'] = run_two_waves(env, wa, wb, pos_a_start, pos_b_start, T)

    # 4. Diff-D — prior differentiation
    env = SemFabricEnv(heterogeneous=True, seed=seed)
    wa = Wave(seed=seed, prior_dom=GEOINT, learn=True)
    wb = Wave(seed=seed + 1, prior_dom=SIGINT, learn=True)
    conditions['diff_D'] = run_two_waves(env, wa, wb, pos_a_start, pos_b_start, T)

    # 5. Adversarial — misspecified model
    env = SemFabricEnv(heterogeneous=True, seed=seed)
    wa = Wave(seed=seed, learn=True)
    wb = Wave(seed=seed + 1, expertise_dom=OSINT, preference_dom=OSINT,
              prior_dom=OSINT, learn=True)
    conditions['adversarial'] = run_two_waves(env, wa, wb, pos_a_start, pos_b_start, T)

    for name, m in conditions.items():
        print(f"  {name}: enriched={m['enrichment_count'][-1]}, "
              f"F_final={m['fabric_F'][-1]:.3f}")

    # --- Figure ---
    fig = plt.figure(figsize=(14, 10))

    ts = np.arange(T)
    colors = {'shared': 'gray', 'diff_A': 'blue', 'diff_C': 'green',
              'diff_D': 'red', 'adversarial': 'orange'}
    labels = {'shared': 'Shared', 'diff_A': 'Diff-A (perception)',
              'diff_C': 'Diff-C (preference)', 'diff_D': 'Diff-D (prior)',
              'adversarial': 'Adversarial'}

    # Top-left: VFE
    ax1 = fig.add_subplot(2, 3, 1)
    for name in conditions:
        ax1.plot(ts, conditions[name]['fabric_F'], color=colors[name],
                 label=labels[name], linewidth=1.3)
    ax1.set_xlabel('Timestep'); ax1.set_ylabel(r'$\mathcal{F}_{\mathrm{fabric}}$')
    ax1.set_title(r'(a) Fabric VFE'); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    # Top-middle: Enrichment
    ax2 = fig.add_subplot(2, 3, 2)
    for name in conditions:
        ax2.plot(ts, conditions[name]['enrichment_count'], color=colors[name],
                 label=labels[name], linewidth=1.3)
    ax2.set_xlabel('Timestep'); ax2.set_ylabel('Enriched cells')
    ax2.set_title('(b) Enrichment progress'); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    # Top-right: Coverage
    ax3 = fig.add_subplot(2, 3, 3)
    for name in conditions:
        ax3.plot(ts, conditions[name]['coverage'], color=colors[name],
                 label=labels[name], linewidth=1.3)
    ax3.set_xlabel('Timestep'); ax3.set_ylabel('Coverage (processed/total)')
    ax3.set_title('(c) Fabric coverage'); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # Bottom: heatmaps for each condition
    cmap = ListedColormap(['#f0f0f0', '#ffd700', '#2ecc71'])
    for i, name in enumerate(['shared', 'diff_A', 'diff_C', 'diff_D', 'adversarial']):
        ax = fig.add_subplot(2, 5, 6 + i)
        grid = conditions[name]['fabric_states'][-1]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
        ax.set_title(labels[name], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'exp2_multiwave_cooperation.{fmt}'), dpi=200)
    plt.close(fig)
    print("  Figure saved.")
    return conditions


# ============================================================
# Experiment 3 : Nested Free Energy
# ============================================================
def experiment_3():
    print("=== Experiment 3: Nested Free Energy ===")
    T = 100
    seed = 42
    start = rc_to_idx(2, 2)

    env = SemFabricEnv(seed=seed)
    wave = Wave(seed=seed, prior_dom=GEOINT, learn=True)

    # Track specific cells and regions
    tracked_cells = [
        rc_to_idx(0, 0), rc_to_idx(1, 2), rc_to_idx(2, 4),
        rc_to_idx(3, 1), rc_to_idx(4, 3),
    ]
    tracked_regions = [(0, 0), (0, 2), (2, 0), (2, 2)]

    sign_F = {k: [] for k in tracked_cells}
    region_F = {r: [] for r in tracked_regions}
    fabric_F = []

    pos = start
    obs = env.observe(pos)

    for t in range(T):
        ea, qs, _, _ = wave.infer_and_act(pos, obs)
        wave.update_neighbor_beliefs(pos, obs[1])
        wave.propagate_domain_belief(pos, obs[2])
        move, target = wave.select_movement(pos)
        dom = wave.cell_beliefs_dom[target]
        pos, obs, did = env.step(pos, move, ea, dom)
        wave.record_enrichment(pos, did)

        if wave.learn and (t + 1) % 20 == 0:
            wave.do_learning_update()
            wave.reset_history()

        # Record at three scales
        for k in tracked_cells:
            sign_F[k].append(sign_free_energy(env.sem[k], env.quality[k]))
        for (r0, c0) in tracked_regions:
            region_F[(r0, c0)].append(regional_free_energy(env, r0, c0, size=2))
        fabric_F.append(fabric_free_energy(env))

    print(f"  F_final={fabric_F[-1]:.3f}, enriched={env.n_enriched()}")

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ts = np.arange(T)

    # (a) Sign-level
    ax = axes[0]
    cell_labels = ['(0,0)', '(1,2)', '(2,4)', '(3,1)', '(4,3)']
    for k, lbl in zip(tracked_cells, cell_labels):
        ax.plot(ts, sign_F[k], linewidth=1.0, label=f'Cell {lbl}')
    ax.set_xlabel('Timestep'); ax.set_ylabel('Sign-level VFE')
    ax.set_title('(a) Sign-level VFE'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) Regional
    ax = axes[1]
    for (r0, c0) in tracked_regions:
        ax.plot(ts, region_F[(r0, c0)], linewidth=1.2,
                label=f'Region ({r0},{c0})')
    ax.set_xlabel('Timestep'); ax.set_ylabel('Regional VFE')
    ax.set_title(r'(b) Regional VFE ($2\times 2$)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (c) Multi-scale comparison
    ax = axes[2]
    mean_sign = np.mean([sign_F[k] for k in tracked_cells], axis=0)
    mean_region = np.mean([region_F[r] for r in tracked_regions], axis=0)
    ax.plot(ts, mean_sign, 'b-', linewidth=1.2, label='Mean sign-level')
    ax.plot(ts, mean_region, 'g--', linewidth=1.2, label='Mean regional')
    ax.plot(ts, fabric_F, 'r-', linewidth=2.0, label=r'$\mathcal{F}_{\mathrm{fabric}}$')
    ax.set_xlabel('Timestep'); ax.set_ylabel('VFE')
    ax.set_title('(c) Multi-scale VFE'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    fig.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(os.path.join(FIG_DIR, f'exp3_nested_free_energy.{fmt}'), dpi=200)
    plt.close(fig)
    print("  Figure saved.")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Semiotic Fabric Active Inference — pymdp JAX Simulation")
    print("=" * 56)
    experiment_1()
    print()
    experiment_2()
    print()
    experiment_3()
    print()
    print("All experiments complete. Figures in:", os.path.abspath(FIG_DIR))
