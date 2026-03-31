"""
exp4_nested_vfe.py -- Experiment 4: Nested Free Energy (Cognitive Substrate)
===========================================================================

Claim: The fabric exhibits multi-scale self-organization -- the signature
of a cognitive system.

Run Experiments 1-3 on a single integrated fabric, track uncertainty
(posterior entropy serving as a VFE proxy) at 4 scales:
  Element-level -> Cluster-level -> Domain-level -> Fabric-level

Key prediction: fabric-level uncertainty decreases monotonically even when
individual elements show discrete jumps.
"""

import numpy as np
from common import (
    InferenceWave, discrete_vfe, log_s, entropy_H,
    setup_figure_style, save_figure, FIG_DIR,
)
from exp1_asd import (
    ArtifactGenerator,
    build_A_magic_1f, build_A_ext_1f, build_A_parse_diagnostic,
    build_A_parse_2f, _expand_A, build_A_size_1f,
    build_C as build_C_asd, build_B_2f as build_B_asd,
    build_D_2f as build_D_asd,
    N_STANDARDS, N_ACTIONS as N_ASD_ACTIONS, CLASSIFY, REQUEST_PARSE,
    N_PARSE,
)
from exp2_disambiguation import (
    SignGenerator, build_A_signifier, build_A_association,
    build_A_domain_cue, build_B as build_B_disambig,
    build_C as build_C_disambig, build_D as build_D_disambig,
    N_CONCEPTS, N_ASSOC, N_ACTIONS as N_DISAMBIG_ACTIONS,
    OBSERVE_MORE, COMMIT, MAX_STEPS,
)
from exp3_entity_resolution import (
    EntityFabric, build_A_feature, build_A_domsig,
    build_A_crosslink, build_A_temporal,
    build_B_entity, build_C_wave, build_D_wave,
    N_ENTITIES, N_DOMAIN_SOURCES, N_ACTIONS as N_ER_ACTIONS,
    N_STEPS as ER_STEPS, LINK_CONFIDENCE_THRESHOLD,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Integrated Fabric
# ============================================================
class IntegratedFabric:
    """Integrated fabric combining ASD, disambiguation, and entity resolution."""

    def __init__(self, n_artifacts=20, n_disambig=50, n_entity=30):
        self.n_artifacts = n_artifacts
        self.n_disambig = n_disambig
        self.n_entity = n_entity
        self.n_total = n_artifacts + n_disambig + n_entity

        self.element_uncertainty = np.zeros(self.n_total)
        self.element_uncertainty[:n_artifacts] = np.log(N_STANDARDS)
        self.element_uncertainty[n_artifacts:n_artifacts + n_disambig] = np.log(N_CONCEPTS)
        self.element_uncertainty[n_artifacts + n_disambig:] = np.log(N_ENTITIES)

        self.element_type = (
            ['artifact'] * n_artifacts +
            ['sign'] * n_disambig +
            ['entity'] * n_entity
        )

        self.n_clusters = 6 + 5 + 8
        self.element_cluster = np.zeros(self.n_total, dtype=int)

        self.n_domains = 4
        self.element_domain = np.zeros(self.n_total, dtype=int)
        self.element_domain[:n_artifacts] = 0
        self.element_domain[n_artifacts:n_artifacts + n_disambig] = 1

    def update(self, idx, uncertainty, cluster=None):
        self.element_uncertainty[idx] = uncertainty
        if cluster is not None:
            self.element_cluster[idx] = cluster

    def get_cluster_uncertainty(self):
        cluster_u = np.zeros(self.n_clusters)
        cluster_c = np.zeros(self.n_clusters)
        for i in range(self.n_total):
            c = self.element_cluster[i]
            cluster_u[c] += self.element_uncertainty[i]
            cluster_c[c] += 1
        cluster_c[cluster_c == 0] = 1
        return cluster_u / cluster_c

    def get_domain_uncertainty(self):
        domain_u = np.zeros(self.n_domains)
        domain_c = np.zeros(self.n_domains)
        for i in range(self.n_total):
            d = self.element_domain[i]
            domain_u[d] += self.element_uncertainty[i]
            domain_c[d] += 1
        domain_c[domain_c == 0] = 1
        return domain_u / domain_c

    def get_fabric_uncertainty(self):
        return float(np.mean(self.element_uncertainty))


# ============================================================
# Run experiment
# ============================================================
def run_experiment_4(seed=42):
    print("=== Experiment 4: Nested Free Energy ===")

    fabric = IntegratedFabric(n_artifacts=20, n_disambig=50, n_entity=30)

    # Generate all data first so we can pre-assign true clusters
    gen_asd = ArtifactGenerator(seed=seed, ambiguity_rate=0.30)
    artifacts = gen_asd.generate(n=fabric.n_artifacts)

    gen_signs = SignGenerator(seed=seed, n_polysemous=15, n_total=fabric.n_disambig)
    signs = gen_signs.generate()

    er_fabric = EntityFabric(seed=seed)
    base_idx = fabric.n_artifacts + fabric.n_disambig

    # Pre-assign TRUE clusters for all elements (not predicted — ground truth)
    # This ensures all clusters have members from step 0, so cluster-level
    # means decrease monotonically as uncertainty reduces.
    for i, art in enumerate(artifacts):
        fabric.element_cluster[i] = art['true_standard']
    for i, sign in enumerate(signs):
        fabric.element_cluster[fabric.n_artifacts + i] = 6 + sign['true_concept']
    for s in range(fabric.n_entity):
        fabric.element_cluster[base_idx + s] = 11 + er_fabric.sign_entity[s]
        dom = er_fabric.sign_domain[s]
        fabric.element_domain[base_idx + s] = 2 + min(dom, 1)

    # Tracking
    element_history = []
    cluster_history = []
    domain_history = []
    fabric_history = []

    # Record initial state
    element_history.append(fabric.element_uncertainty.copy())
    cluster_history.append(fabric.get_cluster_uncertainty())
    domain_history.append(fabric.get_domain_uncertainty())
    fabric_history.append(fabric.get_fabric_uncertainty())

    # === Phase 1: ASD (2-factor model) ===
    print("  Phase 1: Artifact Standard Detection...")

    asd_wave = InferenceWave(
        A_np=[_expand_A(build_A_magic_1f()), _expand_A(build_A_ext_1f()),
              build_A_parse_2f(), _expand_A(build_A_size_1f())],
        B_np=build_B_asd(), C_np=build_C_asd(), D_np=build_D_asd(),
        num_controls=[N_ASD_ACTIONS, 1], control_fac_idx=[0],
        learn=False, gamma=16.0, seed=seed,
        use_states_info_gain=True,
    )

    for i, art in enumerate(artifacts):
        obs = [art['obs_magic'], art['obs_ext'], N_PARSE - 1, art['obs_size']]
        qs_np, _, _, action = asd_wave.infer(obs)

        if action == REQUEST_PARSE:
            obs_parsed = [art['obs_magic'], art['obs_ext'],
                          art['obs_parse_if_requested'], art['obs_size']]
            emp_prior = [np.array([0.0, 1.0]), qs_np[1].copy()]
            qs_np, _, _, _ = asd_wave.infer(obs_parsed, empirical_prior=emp_prior)

        # Uncertainty from standard factor (factor 1)
        uncertainty = entropy_H(qs_np[1])
        fabric.update(i, uncertainty)

        element_history.append(fabric.element_uncertainty.copy())
        cluster_history.append(fabric.get_cluster_uncertainty())
        domain_history.append(fabric.get_domain_uncertainty())
        fabric_history.append(fabric.get_fabric_uncertainty())

    # === Phase 2: Disambiguation ===
    print("  Phase 2: Ambiguity Resolution...")
    A_sig = build_A_signifier()
    A_dmcue = build_A_domain_cue()
    B_dis = build_B_disambig()
    C_dis = build_C_disambig()
    D_dis = build_D_disambig()

    for i, sign in enumerate(signs):
        A_assoc_0 = build_A_association(step=0)
        dis_wave = InferenceWave(
            A_np=[A_sig, A_assoc_0, A_dmcue],
            B_np=B_dis, C_np=C_dis, D_np=D_dis,
            num_controls=[2, 1], control_fac_idx=[0],
            learn=False, gamma=6.0, seed=seed + i,
            use_states_info_gain=True,
        )

        obs = [sign['obs_sig'], N_ASSOC - 1, sign['obs_dmcue']]
        qs_np, _, _, _ = dis_wave.infer(obs)

        A_sig_u = np.ones_like(A_sig) / A_sig.shape[0]
        A_dmcue_u = np.ones_like(A_dmcue) / A_dmcue.shape[0]
        for step in range(MAX_STEPS):
            A_assoc_new = build_A_association(step + 1)
            dis_wave.A_np[0] = A_sig_u
            dis_wave.A_np[1] = A_assoc_new
            dis_wave.A_np[2] = A_dmcue_u
            obs = [sign['obs_sig'], sign['assoc_sequence'][step], sign['obs_dmcue']]
            qs_np, _, _, _ = dis_wave.infer(obs, empirical_prior=qs_np)

        uncertainty = entropy_H(qs_np[0])
        elem_idx = fabric.n_artifacts + i
        fabric.update(elem_idx, uncertainty)

        element_history.append(fabric.element_uncertainty.copy())
        cluster_history.append(fabric.get_cluster_uncertainty())
        domain_history.append(fabric.get_domain_uncertainty())
        fabric_history.append(fabric.get_fabric_uncertainty())

    # === Phase 3: Entity Resolution ===
    print("  Phase 3: Entity Resolution...")
    er_waves = []
    for dom in range(N_DOMAIN_SOURCES):
        wave = InferenceWave(
            A_np=[build_A_feature(domain_expertise=dom),
                  build_A_domsig(),
                  build_A_crosslink(link_available=True),
                  build_A_temporal()],
            B_np=build_B_entity(),
            C_np=build_C_wave(dom),
            D_np=build_D_wave(dom),
            num_controls=[N_ER_ACTIONS, 1], control_fac_idx=[0],
            learn=False, gamma=6.0, seed=seed + dom,
            use_states_info_gain=True,
        )
        er_waves.append(wave)

    rng = np.random.RandomState(seed)
    n_waves = len(er_waves)
    sign_posteriors = {}
    for step in range(ER_STEPS):
        for w_idx, wave in enumerate(er_waves):
            dom = w_idx
            domain_signs = np.where(er_fabric.sign_domain == dom)[0]
            if len(domain_signs) == 0:
                continue
            sign_idx = int(domain_signs[step % len(domain_signs)])

            other_domain_signs = np.where(er_fabric.sign_domain != dom)[0]
            if len(other_domain_signs) > 0:
                cycle_idx = (step * n_waves + w_idx) % len(other_domain_signs)
                other_sign = other_domain_signs[cycle_idx]
            else:
                other_sign = None

            obs = er_fabric.observe(
                sign_idx,
                wave.A_np[0], wave.A_np[1], wave.A_np[2], wave.A_np[3],
                rng, other_sign,
            )

            emp_prior = sign_posteriors.get(sign_idx, None)

            # Belief propagation through links (matches exp3)
            if emp_prior is not None:
                linked = np.where(er_fabric.link_state[sign_idx] > 0)[0]
                if len(linked) > 0:
                    linked_posts = []
                    for l_sign in linked[:3]:
                        l_post = sign_posteriors.get(int(l_sign), None)
                        if l_post is not None:
                            linked_posts.append(l_post[0])
                    if linked_posts:
                        avg_linked = np.mean(linked_posts, axis=0)
                        avg_linked /= avg_linked.sum() + 1e-16
                        entity_prior = 0.90 * emp_prior[0] + 0.10 * avg_linked
                        entity_prior /= entity_prior.sum() + 1e-16
                        emp_prior = [entity_prior, emp_prior[1].copy()]

            qs_np, _, _, action = wave.infer(obs, empirical_prior=emp_prior)
            sign_posteriors[sign_idx] = [q.copy() for q in qs_np]

            uncertainty = entropy_H(qs_np[0])
            pred = int(np.argmax(qs_np[0]))
            elem_idx = base_idx + sign_idx
            fabric.update(elem_idx, uncertainty)

            # Confidence-gated stigmergic coupling (matches exp3)
            if other_sign is not None and step >= 50:
                my_conf = float(qs_np[0][pred])
                other_post = sign_posteriors.get(int(other_sign), None)
                if other_post is not None:
                    other_pred = int(np.argmax(other_post[0]))
                    other_conf = float(other_post[0][other_pred])
                    if (my_conf > LINK_CONFIDENCE_THRESHOLD
                            and other_conf > LINK_CONFIDENCE_THRESHOLD
                            and pred == other_pred):
                        er_fabric.assert_link(sign_idx, int(other_sign),
                                              same=True)

        element_history.append(fabric.element_uncertainty.copy())
        cluster_history.append(fabric.get_cluster_uncertainty())
        domain_history.append(fabric.get_domain_uncertainty())
        fabric_history.append(fabric.get_fabric_uncertainty())

    element_history = np.array(element_history)
    cluster_history = np.array(cluster_history)
    domain_history = np.array(domain_history)
    fabric_history = np.array(fabric_history)

    print(f"  Fabric uncertainty: {fabric_history[0]:.3f} -> {fabric_history[-1]:.3f}")
    print(f"  Total processing steps: {len(fabric_history)}")

    return {
        'element_uncertainty': element_history,
        'cluster_uncertainty': cluster_history,
        'domain_uncertainty': domain_history,
        'fabric_uncertainty': fabric_history,
        'fabric': fabric,
        'n_artifacts': fabric.n_artifacts,
        'n_disambig': fabric.n_disambig,
        'n_entity': fabric.n_entity,
    }


# ============================================================
# Figure generation
# ============================================================
def plot_experiment_4(results):
    setup_figure_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    element_u = results['element_uncertainty']
    cluster_u = results['cluster_uncertainty']
    domain_u = results['domain_uncertainty']
    fabric_u = results['fabric_uncertainty']
    n_art = results['n_artifacts']
    n_dis = results['n_disambig']
    n_steps = len(fabric_u)
    ts = np.arange(n_steps)

    phase1_end = n_art + 1
    phase2_end = phase1_end + n_dis

    # (a) Element-level uncertainty
    ax = axes[0]
    art_indices = [0, 5, 10, 15]
    sign_indices = [n_art + 0, n_art + 15, n_art + 30]
    entity_indices = [n_art + n_dis + 0, n_art + n_dis + 10, n_art + n_dis + 20]

    for idx in art_indices:
        ax.plot(ts, element_u[:, idx], linewidth=0.8, alpha=0.6, color='steelblue')
    for idx in sign_indices:
        ax.plot(ts, element_u[:, idx], linewidth=0.8, alpha=0.6, color='coral')
    for idx in entity_indices:
        if idx < element_u.shape[1]:
            ax.plot(ts, element_u[:, idx], linewidth=0.8, alpha=0.6, color='mediumseagreen')

    ax.axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=phase2_end, color='gray', linestyle=':', alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=1.5, label='Artifact elements'),
        Line2D([0], [0], color='coral', linewidth=1.5, label='Sign elements'),
        Line2D([0], [0], color='mediumseagreen', linewidth=1.5, label='Entity elements'),
    ]
    ax.legend(handles=legend_elements, fontsize=7)
    ax.set_xlabel('Processing step')
    ax.set_ylabel(r'Element-level $H[q(s)]$')
    ax.set_title('(a) Element-level uncertainty (discrete jumps)')

    # (b) Cluster and domain level
    ax = axes[1]
    mean_cluster = cluster_u.mean(axis=1)
    mean_domain = domain_u.mean(axis=1)

    ax.plot(ts, mean_cluster, 'darkorange', linewidth=1.5, label='Mean cluster')
    ax.plot(ts, mean_domain, 'purple', linewidth=1.5, label='Mean domain')
    ax.axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=phase2_end, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Processing step')
    ax.set_ylabel(r'$H[q(s)]$')
    ax.set_title('(b) Cluster and domain level (intermediate smoothing)')
    ax.legend(fontsize=8)

    # (c) Multi-scale comparison
    ax = axes[2]
    mean_element = element_u.mean(axis=1)

    ax.plot(ts, mean_element, 'steelblue', linewidth=1.0, alpha=0.7, label='Mean element')
    ax.plot(ts, mean_cluster, 'darkorange', linewidth=1.2, alpha=0.8, label='Mean cluster')
    ax.plot(ts, mean_domain, 'purple', linewidth=1.5, label='Mean domain')
    ax.plot(ts, fabric_u, 'red', linewidth=2.5,
            label=r'$\mathcal{F}_{\mathrm{fabric}}$ (fabric-level)')

    ax.axvline(x=phase1_end, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=phase2_end, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Processing step')
    ax.set_ylabel(r'Uncertainty ($H[q(s)]$)')
    ax.set_title('(c) Multi-scale: progressively smoother at higher scales')
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_figure(fig, 'exp4_nested_vfe')
    return fig


if __name__ == '__main__':
    results = run_experiment_4()
    plot_experiment_4(results)
