"""
exp3_entity_resolution.py -- Experiment 3: Multi-Wave Entity Resolution
======================================================================

Claim: The PWF's shared fabric enables cooperative intelligence through
overlapping Markov blankets.

Three domain-specialized Waves cooperate through the fabric to resolve
entities across domains.
  - Fabric: 30 Signs from 3 domains (10 each), referring to 8 true entities
  - Hidden states: 2 factors per Wave -- entity_id (8), domain_source (3)
  - Observations: 4 modalities -- feature pattern, domain signature,
    cross-domain link, temporal correlation
  - Actions: 3 -- observe, link_same, link_different
  - Stigmergic coupling: confidence-gated link assertions propagate through
    the shared fabric to influence future observations
"""

import numpy as np
from common import (
    InferenceWave, discrete_vfe, log_s, entropy_H,
    setup_figure_style, save_figure, FIG_DIR,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
N_ENTITIES = 8
N_DOMAIN_SOURCES = 3
GEOINT, SIGINT, HUMINT = 0, 1, 2
DOMAIN_NAMES = ['GEOINT', 'SIGINT', 'HUMINT']

N_SIGNS_PER_DOMAIN = 10
N_SIGNS_TOTAL = 30

# Observations
N_FEATURE = 5    # pattern_A, pattern_B, pattern_C, pattern_D, ambiguous
N_DOMSIG = 4     # geoint_sig, sigint_sig, humint_sig, ambiguous
N_CROSSLINK = 3  # link_found, no_link, null
N_TEMPORAL = 4   # correlated, weakly_corr, uncorrelated, null

# Actions
OBSERVE, LINK_SAME, LINK_DIFFERENT = 0, 1, 2
N_ACTIONS = 3

# Which entities appear in which domain (domain knowledge)
DOMAIN_ENTITIES = {
    GEOINT: {0, 2, 3, 4, 5},
    SIGINT: {0, 1, 3, 6},
    HUMINT: {1, 2, 4, 7},
}

N_STEPS = 120

# Confidence threshold for link assertions
LINK_CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# Fabric
# ============================================================
class EntityFabric:
    """Fabric of 30 Signs across 3 domains, referring to 8 entities."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

        self.sign_entity = np.zeros(N_SIGNS_TOTAL, dtype=int)
        self.sign_domain = np.zeros(N_SIGNS_TOTAL, dtype=int)

        # Entity assignments: 5 cross-domain + 3 single-domain
        # Cross-domain: each appears in exactly 2 domains
        assignments = []
        # Entity 0: GEOINT + SIGINT
        assignments.extend([(0, GEOINT), (0, SIGINT)])
        # Entity 1: SIGINT + HUMINT
        assignments.extend([(1, SIGINT), (1, HUMINT)])
        # Entity 2: GEOINT + HUMINT
        assignments.extend([(2, GEOINT), (2, HUMINT)])
        # Entity 3: GEOINT + SIGINT
        assignments.extend([(3, GEOINT), (3, SIGINT)])
        # Entity 4: HUMINT + GEOINT
        assignments.extend([(4, HUMINT), (4, GEOINT)])
        # Now: GEOINT=4, SIGINT=4, HUMINT=4  (cross-domain)

        # Single-domain entities to fill to 10 per domain
        # GEOINT has 4 cross-domain signs, needs 6 more
        # Entity 5: GEOINT-only (6 signs)
        for _ in range(6):
            assignments.append((5, GEOINT))
        # SIGINT has 3 cross-domain signs, needs 7 more
        # Entity 6: SIGINT-only (7 signs)
        for _ in range(7):
            assignments.append((6, SIGINT))
        # HUMINT has 3 cross-domain signs, needs 7 more
        # Entity 7: HUMINT-only (7 signs)
        for _ in range(7):
            assignments.append((7, HUMINT))
        # Now: GEOINT=10, SIGINT=10, HUMINT=10 = 30 total

        for i, (ent, dom) in enumerate(assignments):
            self.sign_entity[i] = ent
            self.sign_domain[i] = dom

        # Cross-domain link state (stigmergic)
        self.link_state = np.zeros((N_SIGNS_TOTAL, N_SIGNS_TOTAL), dtype=int)

        # Ground truth
        self.same_entity = np.zeros((N_SIGNS_TOTAL, N_SIGNS_TOTAL), dtype=bool)
        for i in range(N_SIGNS_TOTAL):
            for j in range(N_SIGNS_TOTAL):
                self.same_entity[i, j] = (self.sign_entity[i] == self.sign_entity[j])

    def observe(self, sign_idx, A_feature, A_domsig, A_crosslink, A_temporal,
                rng, other_sign_idx=None):
        """Generate observations for a Sign based on its true hidden state.

        Cross-domain link observation reflects the fabric's stigmergic state:
        if ANY other Wave has asserted a link involving this Sign, the current
        Wave can see it (the fabric is shared state).  Observations are
        probabilistic to model noise in the fabric query.
        """
        true_ent = self.sign_entity[sign_idx]
        true_dom = self.sign_domain[sign_idx]

        obs_feature = rng.choice(N_FEATURE, p=A_feature[:, true_ent, true_dom])
        obs_domsig = rng.choice(N_DOMSIG, p=A_domsig[:, true_ent, true_dom])

        # Cross-domain link: check ALL established links in the fabric
        positive_links = int(np.sum(self.link_state[sign_idx] > 0))
        negative_links = int(np.sum(self.link_state[sign_idx] < 0))
        if positive_links > 0:
            # Probabilistic: high but not certain (noise in fabric query)
            obs_crosslink = rng.choice(N_CROSSLINK, p=[0.80, 0.05, 0.15])
        elif negative_links > 0:
            obs_crosslink = rng.choice(N_CROSSLINK, p=[0.05, 0.75, 0.20])
        else:
            obs_crosslink = rng.choice(N_CROSSLINK, p=A_crosslink[:, true_ent, true_dom])

        obs_temporal = rng.choice(N_TEMPORAL, p=A_temporal[:, true_ent, true_dom])

        return [obs_feature, obs_domsig, obs_crosslink, obs_temporal]

    def assert_link(self, sign_i, sign_j, same):
        val = 1 if same else -1
        self.link_state[sign_i, sign_j] = val
        self.link_state[sign_j, sign_i] = val


# ============================================================
# Generative model builders
# ============================================================
def build_A_feature(domain_expertise=None):
    """P(feature_obs | entity, domain). Shape (5, 8, 3).
    Each entity has a distinctive feature pattern.
    Expert Waves have sharper discrimination for their domain."""
    A = np.zeros((N_FEATURE, N_ENTITIES, N_DOMAIN_SOURCES))

    # Entity-specific feature patterns -- designed so each cross-domain
    # entity shares its pattern with a single-domain entity, creating
    # confusion pairs that can only be resolved through cross-domain
    # cooperation (belief propagation through stigmergic links).
    #
    # Confusion pairs and their resolution mechanism:
    #   (0,5) in GEOINT: Entity 0 unique in SIGINT → propagation resolves
    #   (3,4) in GEOINT: Entity 3 unique in SIGINT, Entity 4 unique in HUMINT
    #   (1,6) in SIGINT: temporal cross/single + mutual propagation with HUMINT
    #   (1,7) in HUMINT: temporal cross/single + mutual propagation with SIGINT
    entity_patterns = {
        0: 0, 1: 1, 2: 2, 3: 3,   # cross-domain entities
        4: 3, 5: 0, 6: 1, 7: 1,   # single-domain: shares with cross-domain
    }

    for e in range(N_ENTITIES):
        main_pattern = entity_patterns[e]
        for d in range(N_DOMAIN_SOURCES):
            if domain_expertise is not None and d == domain_expertise:
                # Expert: high precision for own-domain entities
                A[main_pattern, e, d] = 0.70
                for p in range(4):
                    if p != main_pattern:
                        A[p, e, d] = 0.06
                A[4, e, d] = 0.06  # ambiguous
            else:
                # Non-expert: moderate precision
                A[main_pattern, e, d] = 0.40
                for p in range(4):
                    if p != main_pattern:
                        A[p, e, d] = 0.11
                A[4, e, d] = 0.16

    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_domsig():
    """P(domain_sig | entity, domain). Shape (4, 8, 3)."""
    A = np.zeros((N_DOMSIG, N_ENTITIES, N_DOMAIN_SOURCES))
    for e in range(N_ENTITIES):
        for d in range(N_DOMAIN_SOURCES):
            A[d, e, d] = 0.60
            A[N_DOMSIG - 1, e, d] = 0.15
            for d2 in range(N_DOMAIN_SOURCES):
                if d2 != d:
                    A[d2, e, d] = (1.0 - 0.60 - 0.15) / (N_DOMAIN_SOURCES - 1)
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_crosslink(link_available=False):
    """P(cross_link | entity, domain). Shape (3, 8, 3).
    Cross-domain entities more likely to have links discovered.
    Moderate discrimination between cross and single-domain."""
    A = np.zeros((N_CROSSLINK, N_ENTITIES, N_DOMAIN_SOURCES))
    for e in range(N_ENTITIES):
        is_cross = e < 5
        for d in range(N_DOMAIN_SOURCES):
            if link_available and is_cross:
                A[0, e, d] = 0.45  # link_found
                A[1, e, d] = 0.15
                A[2, e, d] = 0.40  # null
            elif link_available and not is_cross:
                A[0, e, d] = 0.08
                A[1, e, d] = 0.42
                A[2, e, d] = 0.50
            else:
                A[0, e, d] = 0.05
                A[1, e, d] = 0.10
                A[2, e, d] = 0.85
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_temporal():
    """P(temporal_obs | entity, domain). Shape (4, 8, 3).
    Moderate discrimination between cross-domain and single-domain entities.
    Cross-domain entities show stronger temporal correlation (co-occurrence
    across domains). This helps distinguish confusion pairs like (0,5) where
    entity 0 is cross-domain and entity 5 is single-domain."""
    A = np.zeros((N_TEMPORAL, N_ENTITIES, N_DOMAIN_SOURCES))
    for e in range(N_ENTITIES):
        is_cross = e < 5
        for d in range(N_DOMAIN_SOURCES):
            if is_cross:
                A[0, e, d] = 0.35   # correlated
                A[1, e, d] = 0.25   # weakly_corr
                A[2, e, d] = 0.20   # uncorrelated
                A[3, e, d] = 0.20   # null
            else:
                A[0, e, d] = 0.15   # correlated
                A[1, e, d] = 0.20   # weakly_corr
                A[2, e, d] = 0.35   # uncorrelated
                A[3, e, d] = 0.30   # null
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_B_entity():
    B_entity = np.stack([np.eye(N_ENTITIES)] * N_ACTIONS, axis=-1)
    B_domain = np.expand_dims(np.eye(N_DOMAIN_SOURCES), -1)
    return [B_entity, B_domain]


def build_C_wave(domain):
    C_feature = np.array([0.5, 0.5, 0.5, 0.5, -0.5])
    C_domsig = np.zeros(N_DOMSIG)
    C_domsig[domain] = 0.5
    C_domsig[N_DOMSIG - 1] = -0.3
    C_crosslink = np.array([1.0, -0.2, -0.1])
    C_temporal = np.array([0.5, 0.2, -0.2, -0.1])
    return [C_feature, C_domsig, C_crosslink, C_temporal]


def build_D_wave(domain):
    """Domain-specific priors. Entity prior reflects domain knowledge:
    entities known to appear in this domain get higher prior mass."""
    D_entity = np.ones(N_ENTITIES) * 0.02
    for e in DOMAIN_ENTITIES[domain]:
        D_entity[e] = 0.18
    D_entity = D_entity / D_entity.sum()
    D_domain = np.full(N_DOMAIN_SOURCES, 0.1)
    D_domain[domain] = 0.8
    D_domain = D_domain / D_domain.sum()
    return [D_entity, D_domain]


# ============================================================
# Wave runner
# ============================================================
def run_waves_on_fabric(waves, fabric, n_steps, seed, coupling=True):
    """Run all Waves on fabric for n_steps.

    Key mechanisms:
      1. Per-sign belief accumulation: each sign's posterior from one visit
         becomes the empirical prior for the next visit.
      2. Confidence-gated stigmergic coupling: link assertions only happen
         when BOTH signs have confident, matching entity predictions.
         This prevents early incorrect links from cascading.

    Returns per-Wave predictions and VFE traces.
    """
    n_waves = len(waves)
    rng = np.random.RandomState(seed)

    all_preds = {}
    entropy_per_step = []
    link_count = 0

    # Per-sign accumulated posteriors (belief accumulation across visits)
    sign_posteriors = {}

    # Build observation models per wave
    wave_A = []
    for w_idx in range(n_waves):
        wave_A.append({
            'feature': waves[w_idx].A_np[0],
            'domsig': waves[w_idx].A_np[1],
            'crosslink': waves[w_idx].A_np[2],
            'temporal': waves[w_idx].A_np[3],
        })

    for step in range(n_steps):
        for w_idx, wave in enumerate(waves):
            dom = w_idx if w_idx < N_DOMAIN_SOURCES else 0
            domain_signs = np.where(fabric.sign_domain == dom)[0]
            if len(domain_signs) == 0:
                continue

            sign_idx = int(domain_signs[step % len(domain_signs)])

            # Pick another Sign from different domain for cross-link check
            # Systematic cycling ensures every cross-domain pair eventually
            # gets compared, rather than relying on random collisions.
            other_domain_signs = np.where(fabric.sign_domain != dom)[0]
            if len(other_domain_signs) > 0:
                cycle_idx = (step * n_waves + w_idx) % len(other_domain_signs)
                other_sign = other_domain_signs[cycle_idx]
            else:
                other_sign = None

            # Generate observations from the fabric
            obs = fabric.observe(
                sign_idx,
                wave_A[w_idx]['feature'],
                wave_A[w_idx]['domsig'],
                wave_A[w_idx]['crosslink'],
                wave_A[w_idx]['temporal'],
                rng,
                other_sign,
            )

            # Use accumulated posterior as empirical prior for this sign
            emp_prior = sign_posteriors.get(sign_idx, None)

            # Belief propagation through links: blend linked signs'
            # posteriors into this sign's empirical prior.  This is
            # the key mechanism that makes cooperative > isolated:
            # entity-specific information flows through the fabric.
            if coupling and emp_prior is not None:
                linked = np.where(fabric.link_state[sign_idx] > 0)[0]
                if len(linked) > 0:
                    linked_posts = []
                    for l_sign in linked[:3]:  # cap at 3 linked signs
                        l_post = sign_posteriors.get(int(l_sign), None)
                        if l_post is not None:
                            linked_posts.append(l_post[0])
                    if linked_posts:
                        avg_linked = np.mean(linked_posts, axis=0)
                        avg_linked /= avg_linked.sum() + 1e-16
                        entity_prior = 0.90 * emp_prior[0] + 0.10 * avg_linked
                        entity_prior /= entity_prior.sum() + 1e-16
                        emp_prior = [entity_prior, emp_prior[1].copy()]

            qs_np, q_pi, efe, action = wave.infer(obs, empirical_prior=emp_prior)

            # Store posterior for next visit to this sign
            sign_posteriors[sign_idx] = [q.copy() for q in qs_np]

            # Entity prediction (from accumulated posterior)
            entity_pred = int(np.argmax(qs_np[0]))
            all_preds[sign_idx] = entity_pred

            # Confidence-gated stigmergic coupling (LINK_SAME only)
            # Only assert positive links: negative links propagate too much
            # noise through the shared fabric.  Delayed until beliefs have
            # partially converged (step >= 20).
            if coupling and other_sign is not None and step >= 50:
                my_entity = entity_pred
                my_confidence = float(qs_np[0][my_entity])

                # Check if other sign has an accumulated posterior
                other_post = sign_posteriors.get(int(other_sign), None)
                if other_post is not None:
                    other_entity = int(np.argmax(other_post[0]))
                    other_confidence = float(other_post[0][other_entity])

                    # Only assert LINK_SAME when BOTH signs are confident
                    # and agree on the entity
                    if (my_confidence > LINK_CONFIDENCE_THRESHOLD
                            and other_confidence > LINK_CONFIDENCE_THRESHOLD
                            and my_entity == other_entity):
                        fabric.assert_link(sign_idx, int(other_sign),
                                           same=True)
                        link_count += 1

        # Mean posterior entropy across all visited signs (decreases as beliefs sharpen)
        if sign_posteriors:
            ents = [entropy_H(post[0]) for post in sign_posteriors.values()]
            entropy_per_step.append(float(np.mean(ents)))
        else:
            entropy_per_step.append(float(np.log(N_ENTITIES)))

    return all_preds, entropy_per_step, link_count


# ============================================================
# Run experiment
# ============================================================
N_REPLICATES = 20


def _run_single_seed(seed):
    """Run all four conditions for a single seed."""
    conditions = {}

    # --- Condition 1: Cooperative Specialized ---
    fabric = EntityFabric(seed=seed)
    waves = []
    for dom in range(N_DOMAIN_SOURCES):
        wave = InferenceWave(
            A_np=[build_A_feature(domain_expertise=dom),
                  build_A_domsig(),
                  build_A_crosslink(link_available=True),
                  build_A_temporal()],
            B_np=build_B_entity(),
            C_np=build_C_wave(dom),
            D_np=build_D_wave(dom),
            num_controls=[N_ACTIONS, 1],
            control_fac_idx=[0],
            learn=False, gamma=6.0, seed=seed + dom,
            use_states_info_gain=True,
        )
        waves.append(wave)
    preds, vfe, links = run_waves_on_fabric(waves, fabric, N_STEPS, seed,
                                            coupling=True)
    conditions['cooperative'] = {
        'accuracy': _compute_accuracy(preds, fabric),
        'fabric_vfe': vfe,
        'per_domain_acc': _per_domain_accuracy(preds, fabric),
        'cross_links': links,
        'all_preds': preds.copy(),
    }

    # --- Condition 2: Single Generalist ---
    fabric2 = EntityFabric(seed=seed)
    gen_wave = InferenceWave(
        A_np=[build_A_feature(domain_expertise=None),
              build_A_domsig(),
              build_A_crosslink(link_available=False),
              build_A_temporal()],
        B_np=build_B_entity(),
        C_np=build_C_wave(GEOINT),
        D_np=[np.ones(N_ENTITIES) / N_ENTITIES,
              np.ones(N_DOMAIN_SOURCES) / N_DOMAIN_SOURCES],
        num_controls=[N_ACTIONS, 1],
        control_fac_idx=[0],
        learn=False, gamma=6.0, seed=seed,
        use_states_info_gain=True,
    )
    preds2, vfe2, _ = run_waves_on_fabric(
        [gen_wave, gen_wave, gen_wave], fabric2, N_STEPS, seed, coupling=False
    )
    conditions['generalist'] = {
        'accuracy': _compute_accuracy(preds2, fabric2),
        'fabric_vfe': vfe2,
        'per_domain_acc': _per_domain_accuracy(preds2, fabric2),
        'cross_links': 0,
        'all_preds': preds2.copy(),
    }

    # --- Condition 3: Isolated Specialized ---
    fabric3 = EntityFabric(seed=seed)
    iso_waves = []
    for dom in range(N_DOMAIN_SOURCES):
        wave = InferenceWave(
            A_np=[build_A_feature(domain_expertise=dom),
                  build_A_domsig(),
                  build_A_crosslink(link_available=False),
                  build_A_temporal()],
            B_np=build_B_entity(),
            C_np=build_C_wave(dom),
            D_np=build_D_wave(dom),
            num_controls=[N_ACTIONS, 1],
            control_fac_idx=[0],
            learn=False, gamma=6.0, seed=seed + dom,
            use_states_info_gain=True,
        )
        iso_waves.append(wave)
    preds3, vfe3, _ = run_waves_on_fabric(
        iso_waves, fabric3, N_STEPS, seed, coupling=False
    )
    conditions['isolated'] = {
        'accuracy': _compute_accuracy(preds3, fabric3),
        'fabric_vfe': vfe3,
        'per_domain_acc': _per_domain_accuracy(preds3, fabric3),
        'cross_links': 0,
        'all_preds': preds3.copy(),
    }

    # --- Condition 4: Homogeneous Cooperative ---
    fabric4 = EntityFabric(seed=seed)
    homo_waves = []
    for dom in range(N_DOMAIN_SOURCES):
        wave = InferenceWave(
            A_np=[build_A_feature(domain_expertise=None),
                  build_A_domsig(),
                  build_A_crosslink(link_available=True),
                  build_A_temporal()],
            B_np=build_B_entity(),
            C_np=build_C_wave(dom),
            D_np=[np.ones(N_ENTITIES) / N_ENTITIES,
                  np.ones(N_DOMAIN_SOURCES) / N_DOMAIN_SOURCES],
            num_controls=[N_ACTIONS, 1],
            control_fac_idx=[0],
            learn=False, gamma=6.0, seed=seed + dom,
            use_states_info_gain=True,
        )
        homo_waves.append(wave)
    preds4, vfe4, links4 = run_waves_on_fabric(
        homo_waves, fabric4, N_STEPS, seed, coupling=True
    )
    conditions['homogeneous'] = {
        'accuracy': _compute_accuracy(preds4, fabric4),
        'fabric_vfe': vfe4,
        'per_domain_acc': _per_domain_accuracy(preds4, fabric4),
        'cross_links': links4,
        'all_preds': preds4.copy(),
    }

    return {'conditions': conditions, 'fabric': fabric}


def run_experiment_3(base_seed=42):
    """Run experiment across N_REPLICATES seeds and report averaged results."""
    print(f"=== Experiment 3: Multi-Wave Entity Resolution "
          f"({N_REPLICATES} replicates) ===")

    cond_names = ['cooperative', 'isolated', 'generalist', 'homogeneous']
    collected = {n: {'accs': [], 'per_dom': [], 'links': [], 'vfes': []}
                 for n in cond_names}

    first_result = None
    for i in range(N_REPLICATES):
        result = _run_single_seed(base_seed + i)
        if first_result is None:
            first_result = result
        for n in cond_names:
            c = result['conditions'][n]
            collected[n]['accs'].append(c['accuracy'])
            collected[n]['per_dom'].append(c['per_domain_acc'])
            collected[n]['links'].append(c['cross_links'])
            collected[n]['vfes'].append(c['fabric_vfe'])

    # Aggregate into averaged conditions dict
    conditions = {}
    for n in cond_names:
        accs = np.array(collected[n]['accs'])
        per_dom = np.array(collected[n]['per_dom'])
        links = np.array(collected[n]['links'], dtype=float)

        # Average VFE traces (pad to max length)
        vfes = collected[n]['vfes']
        max_len = max(len(v) for v in vfes) if vfes else 0
        if max_len > 0:
            vfe_arr = np.full((len(vfes), max_len), np.nan)
            for j, v in enumerate(vfes):
                vfe_arr[j, :len(v)] = v
            mean_vfe = list(np.nanmean(vfe_arr, axis=0))
        else:
            mean_vfe = []

        conditions[n] = {
            'accuracy': float(accs.mean()),
            'accuracy_se': float(accs.std() / np.sqrt(len(accs))),
            'fabric_vfe': mean_vfe,
            'per_domain_acc': list(per_dom.mean(axis=0)),
            'per_domain_se': list(per_dom.std(axis=0) / np.sqrt(len(accs))),
            'cross_links': float(links.mean()),
            'all_preds': first_result['conditions'][n]['all_preds'],
        }

        print(f"  {n}: accuracy={accs.mean():.3f} +/- {accs.std()/np.sqrt(len(accs)):.3f} (SE), "
              f"links={links.mean():.1f}, "
              f"per_domain={[f'{a:.2f}' for a in per_dom.mean(axis=0)]}")

    return {'conditions': conditions, 'fabric': first_result['fabric']}


def _compute_accuracy(preds, fabric):
    if not preds:
        return 0.0
    correct = sum(1 for s, p in preds.items() if p == fabric.sign_entity[s])
    return correct / len(preds)


def _per_domain_accuracy(preds, fabric):
    acc = []
    for dom in range(N_DOMAIN_SOURCES):
        domain_signs = np.where(fabric.sign_domain == dom)[0]
        correct = sum(1 for s in domain_signs if s in preds and preds[s] == fabric.sign_entity[s])
        total = sum(1 for s in domain_signs if s in preds)
        acc.append(correct / max(total, 1))
    return acc


# ============================================================
# Figure generation
# ============================================================
def plot_experiment_3(results):
    setup_figure_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    conditions = results['conditions']
    fabric = results['fabric']

    colors = {
        'cooperative': 'steelblue',
        'generalist': 'darkorange',
        'isolated': 'gray',
        'homogeneous': 'mediumseagreen',
    }
    labels = {
        'cooperative': 'Cooperative\nSpecialized',
        'generalist': 'Single\nGeneralist',
        'isolated': 'Isolated\nSpecialized',
        'homogeneous': 'Homogeneous\nCooperative',
    }
    names = ['cooperative', 'isolated', 'homogeneous', 'generalist']

    # (a) Mean posterior entropy over time
    ax = axes[0, 0]
    for name in names:
        ent = conditions[name]['fabric_vfe']
        if ent:
            ax.plot(range(len(ent)), ent, color=colors[name],
                    label=labels[name].replace('\n', ' '), linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'Mean $H[q(\mathrm{entity})]$ (nats)')
    ax.set_title('(a) Mean entity uncertainty over time')
    ax.legend(fontsize=7)

    # (b) Entity resolution accuracy (with SE error bars)
    ax = axes[0, 1]
    accs = [conditions[n]['accuracy'] for n in names]
    errs = [conditions[n].get('accuracy_se', 0) for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, accs, yerr=errs, capsize=4,
                  color=[colors[n] for n in names])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in names], fontsize=8)
    ax.set_ylabel('Entity Resolution Accuracy')
    ax.set_title(f'(b) Entity resolution accuracy (N={N_REPLICATES})')
    ax.set_ylim(0, 1.0)
    for bar, acc, err in zip(bars, accs, errs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + err + 0.02,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)

    # (c) Cross-domain links
    ax = axes[0, 2]
    link_counts = [conditions[n]['cross_links'] for n in names]
    bars = ax.bar(x, link_counts, color=[colors[n] for n in names])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[n] for n in names], fontsize=8)
    ax.set_ylabel('Cross-domain links asserted')
    ax.set_title('(c) Cross-domain link discovery')
    for bar, cnt in zip(bars, link_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{cnt:.1f}', ha='center', va='bottom', fontsize=9)

    # (d) Per-domain accuracy (with SE error bars)
    ax = axes[1, 0]
    x_dom = np.arange(N_DOMAIN_SOURCES)
    width = 0.2
    for i, name in enumerate(names):
        per_dom = conditions[name]['per_domain_acc']
        per_dom_se = conditions[name].get('per_domain_se', [0]*N_DOMAIN_SOURCES)
        ax.bar(x_dom + i * width, per_dom, width, yerr=per_dom_se,
               capsize=2, color=colors[name],
               label=labels[name].replace('\n', ' '))
    ax.set_xticks(x_dom + width * 1.5)
    ax.set_xticklabels(DOMAIN_NAMES)
    ax.set_ylabel('Accuracy')
    ax.set_title('(d) Per-domain accuracy')
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(0, 1.0)

    # (e) Entity confusion matrix (cooperative)
    ax = axes[1, 1]
    preds = conditions['cooperative']['all_preds']
    confusion = np.zeros((N_ENTITIES, N_ENTITIES))
    for sign_idx, pred_entity in preds.items():
        true_entity = fabric.sign_entity[sign_idx]
        confusion[true_entity, pred_entity] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    confusion_norm = confusion / row_sums
    im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('Predicted entity')
    ax.set_ylabel('True entity')
    ax.set_title('(e) Entity confusion (cooperative)')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(N_ENTITIES))
    ax.set_yticks(range(N_ENTITIES))

    # (f) Stigmergic coupling effect (smoothed)
    ax = axes[1, 2]
    coop_ent = conditions['cooperative']['fabric_vfe']
    iso_ent = conditions['isolated']['fabric_vfe']
    min_len = min(len(coop_ent), len(iso_ent))
    if min_len > 0:
        coupling_effect = np.array(iso_ent[:min_len]) - np.array(coop_ent[:min_len])
        # Rolling average for visual clarity
        window = min(10, max(1, min_len // 5))
        if min_len > window:
            kernel = np.ones(window) / window
            smooth = np.convolve(coupling_effect, kernel, mode='valid')
            x_smooth = np.arange(window - 1, min_len)
            ax.plot(x_smooth, smooth, 'steelblue', linewidth=2)
            ax.fill_between(x_smooth, smooth, alpha=0.3, color='steelblue')
        else:
            ax.plot(range(min_len), coupling_effect, 'steelblue', linewidth=1.5)
            ax.fill_between(range(min_len), coupling_effect, alpha=0.3, color='steelblue')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$\Delta H$ (isolated $-$ cooperative)')
    ax.set_title('(f) Stigmergic coupling benefit')

    fig.tight_layout()
    save_figure(fig, 'exp3_entity_resolution')
    return fig


if __name__ == '__main__':
    results = run_experiment_3()
    plot_experiment_3(results)
