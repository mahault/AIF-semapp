"""
exp2_disambiguation.py -- Experiment 2: Ambiguity Resolution (Content-Level Semiosis)
=====================================================================================

Claim: The SRF's relational structure enables active inference to perform
genuine semiosis.

A single Wave resolves polysemous Signs by accumulating contextual evidence
from Associations.
  - Hidden states: 2 factors
      true_concept (5): PLANET, ELEMENT, DEITY, CAR_BRAND, OTHER
      context_domain (4): ASTRONOMY, CHEMISTRY, MYTHOLOGY, AUTOMOTIVE
  - Observations: 3 modalities
      Signifier features (6 outcomes) -- domain-independent, concept-diagnostic
      Association context (6 outcomes) -- becomes more informative over steps
      Domain model cue (5 outcomes)
  - Progressive context: agent always accumulates association evidence across
    MAX_STEPS steps.  Commits when posterior confidence exceeds threshold.
  - N = 50 Signs, 15 polysemous
"""

import numpy as np
from common import (
    InferenceWave, MLClassifier, discrete_vfe, log_s, entropy_H,
    setup_figure_style, save_figure, FIG_DIR,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
PLANET, ELEMENT, DEITY, CAR_BRAND, OTHER = range(5)
N_CONCEPTS = 5
CONCEPT_NAMES = ['Planet', 'Element', 'Deity', 'Car Brand', 'Other']

ASTRONOMY, CHEMISTRY, MYTHOLOGY, AUTOMOTIVE = range(4)
N_DOMAINS = 4
DOMAIN_NAMES = ['Astronomy', 'Chemistry', 'Mythology', 'Automotive']

N_SIG = 6       # celestial, chemical, mythological, mechanical, generic, null
N_ASSOC = 6     # orbital, periodic_table, pantheon, vehicle, mixed, null
N_DMCUE = 5     # astro, chem, myth, auto, ambiguous

OBSERVE_MORE, COMMIT = 0, 1
N_ACTIONS = 2
MAX_STEPS = 6

# Concept -> canonical domain
CONCEPT_DOM_MAP = {
    PLANET: ASTRONOMY, ELEMENT: CHEMISTRY,
    DEITY: MYTHOLOGY, CAR_BRAND: AUTOMOTIVE,
}


# ============================================================
# Generative model builders
# ============================================================
def build_A_signifier():
    """P(signifier_obs | concept, domain). Shape (6, 5, 4).

    Signifier depends primarily on CONCEPT, not domain.  Each concept has a
    distinct primary signifier with high probability.  This means unambiguous
    terms are clearly identifiable from the signifier alone.

    For polysemous terms, the DATA GENERATOR blurs the signifier across
    candidate concepts, creating genuine ambiguity that only contextual
    evidence (Associations) can resolve.
    """
    A = np.zeros((N_SIG, N_CONCEPTS, N_DOMAINS))

    # concept -> primary signifier index
    concept_sig = {PLANET: 0, ELEMENT: 1, DEITY: 2, CAR_BRAND: 3, OTHER: 4}

    for c in range(N_CONCEPTS):
        primary = concept_sig[c]
        for d in range(N_DOMAINS):
            if c == OTHER:
                A[4, c, d] = 0.50   # generic
                A[5, c, d] = 0.20   # null
            else:
                A[primary, c, d] = 0.45  # moderate primary (domain-independent)
                A[4, c, d] = 0.12        # generic
                A[5, c, d] = 0.08        # null

    # Fill remaining probability mass uniformly
    for d in range(N_DOMAINS):
        for c in range(N_CONCEPTS):
            col = A[:, c, d]
            remainder = 1.0 - col.sum()
            if remainder > 0:
                zeros = col == 0
                if zeros.any():
                    A[zeros, c, d] = remainder / zeros.sum()
                else:
                    A[:, c, d] += remainder / N_SIG

    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_association(step=0):
    """P(assoc_obs | concept, domain). Shape (6, 5, 4).

    Becomes MUCH more informative as more Associations are examined.
    At step 0: nearly uninformative. By step 5: highly diagnostic.

    The association modality is the key to disambiguation -- it depends on
    BOTH concept and domain, so it can resolve polysemous terms where the
    signifier alone is ambiguous.
    """
    # Informativeness ramp: 0.03 at step 0 -> 0.92 at step 5
    clarity = min(0.03 + 0.178 * step, 0.92)

    A = np.zeros((N_ASSOC, N_CONCEPTS, N_DOMAINS))

    # concept-domain diagnostic associations
    # orbital -> PLANET, periodic_table -> ELEMENT, pantheon -> DEITY, vehicle -> CAR_BRAND
    concept_assoc = {PLANET: 0, ELEMENT: 1, DEITY: 2, CAR_BRAND: 3}

    for c in range(N_CONCEPTS):
        for d in range(N_DOMAINS):
            if c == OTHER:
                # OTHER concept: mostly mixed/null
                A[4, c, d] = 0.45
                A[5, c, d] = 0.45
            elif c in concept_assoc:
                diag = concept_assoc[c]
                matching_dom = CONCEPT_DOM_MAP[c]
                if d == matching_dom:
                    # Matching domain: diagnostic association is very informative
                    A[diag, c, d] = clarity * 0.90
                    A[4, c, d] = (1 - clarity) * 0.45  # mixed
                    A[5, c, d] = (1 - clarity) * 0.45  # null
                else:
                    # Mismatching domain: diagnostic association less clear
                    A[diag, c, d] = clarity * 0.35
                    # Some signal from the domain's own diagnostic assoc
                    dom_assoc = {ASTRONOMY: 0, CHEMISTRY: 1, MYTHOLOGY: 2, AUTOMOTIVE: 3}[d]
                    A[dom_assoc, c, d] = clarity * 0.20
                    A[4, c, d] = (1 - clarity) * 0.40
                    A[5, c, d] = (1 - clarity) * 0.40

    # Fill and normalize
    for d in range(N_DOMAINS):
        for c in range(N_CONCEPTS):
            col = A[:, c, d]
            if col.sum() < 0.01:
                A[:, c, d] = 1.0 / N_ASSOC
            else:
                remainder = max(0, 1.0 - col.sum())
                zeros = col == 0
                if zeros.any() and remainder > 0:
                    A[zeros, c, d] = remainder / zeros.sum()
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_domain_cue():
    """P(domain_cue | concept, domain). Shape (5, 5, 4).
    Moderately informative -- gives domain but not concept."""
    A = np.zeros((N_DMCUE, N_CONCEPTS, N_DOMAINS))
    for c in range(N_CONCEPTS):
        for d in range(N_DOMAINS):
            A[d, c, d] = 0.50
            A[N_DMCUE - 1, c, d] = 0.20  # ambiguous
            for d2 in range(N_DOMAINS):
                if d2 != d:
                    A[d2, c, d] = (1.0 - 0.50 - 0.20) / (N_DOMAINS - 1)
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_B():
    B_concept = np.stack([np.eye(N_CONCEPTS)] * N_ACTIONS, axis=-1)
    B_domain = np.expand_dims(np.eye(N_DOMAINS), -1)
    return [B_concept, B_domain]


def build_C():
    C_sig = np.zeros(N_SIG)
    C_sig[:4] = 0.3   # prefer clear features
    C_sig[4] = -0.3   # generic
    C_sig[5] = -0.5   # null

    C_assoc = np.zeros(N_ASSOC)
    C_assoc[:4] = 0.8   # strongly prefer clear associations
    C_assoc[4] = -0.3   # mixed
    C_assoc[5] = -0.5   # null

    C_dmcue = np.zeros(N_DMCUE)
    C_dmcue[:4] = 0.3
    C_dmcue[4] = -0.3

    return [C_sig, C_assoc, C_dmcue]


def build_D():
    D_concept = np.ones(N_CONCEPTS) / N_CONCEPTS
    D_domain = np.ones(N_DOMAINS) / N_DOMAINS
    return [D_concept, D_domain]


# ============================================================
# Sign generator
# ============================================================
POLYSEMOUS_TERMS = [
    'Mercury', 'Mars', 'Jupiter', 'Saturn', 'Venus',
    'Apollo', 'Titan', 'Pluto', 'Helium', 'Selenium',
    'Cadillac', 'Eclipse', 'Vega', 'Phoenix', 'Atlas',
]

POLYSEMOUS_MAPPINGS = {
    'Mercury': [(PLANET, ASTRONOMY), (ELEMENT, CHEMISTRY),
                (DEITY, MYTHOLOGY), (CAR_BRAND, AUTOMOTIVE)],
    'Mars': [(PLANET, ASTRONOMY), (DEITY, MYTHOLOGY), (CAR_BRAND, AUTOMOTIVE)],
    'Jupiter': [(PLANET, ASTRONOMY), (DEITY, MYTHOLOGY)],
    'Saturn': [(PLANET, ASTRONOMY), (DEITY, MYTHOLOGY), (CAR_BRAND, AUTOMOTIVE)],
    'Venus': [(PLANET, ASTRONOMY), (DEITY, MYTHOLOGY)],
    'Apollo': [(DEITY, MYTHOLOGY), (OTHER, ASTRONOMY)],
    'Titan': [(OTHER, ASTRONOMY), (DEITY, MYTHOLOGY), (OTHER, AUTOMOTIVE)],
    'Pluto': [(PLANET, ASTRONOMY), (DEITY, MYTHOLOGY)],
    'Helium': [(ELEMENT, CHEMISTRY), (OTHER, MYTHOLOGY)],
    'Selenium': [(ELEMENT, CHEMISTRY), (OTHER, MYTHOLOGY)],
    'Cadillac': [(CAR_BRAND, AUTOMOTIVE), (OTHER, MYTHOLOGY)],
    'Eclipse': [(OTHER, ASTRONOMY), (CAR_BRAND, AUTOMOTIVE)],
    'Vega': [(OTHER, ASTRONOMY), (CAR_BRAND, AUTOMOTIVE)],
    'Phoenix': [(OTHER, MYTHOLOGY), (CAR_BRAND, AUTOMOTIVE)],
    'Atlas': [(DEITY, MYTHOLOGY), (CAR_BRAND, AUTOMOTIVE)],
}


class SignGenerator:
    def __init__(self, seed=42, n_polysemous=15, n_total=50):
        self.rng = np.random.RandomState(seed)
        self.A_sig = build_A_signifier()
        self.A_dmcue = build_A_domain_cue()
        self.n_polysemous = n_polysemous
        self.n_total = n_total

    def generate(self):
        signs = []

        # Polysemous signs
        poly_names = self.rng.choice(POLYSEMOUS_TERMS,
                                      size=self.n_polysemous, replace=True)
        for name in poly_names:
            mappings = POLYSEMOUS_MAPPINGS[name]
            concept, domain = mappings[self.rng.choice(len(mappings))]

            # KEY: for polysemous terms, the signifier is the WORD itself
            # ("Mercury") which looks identical regardless of meaning.
            # Average over all candidate (concept, domain) distributions
            # so the signifier alone cannot determine the meaning.
            candidate_dists = np.array(
                [self.A_sig[:, c, d] for (c, d) in mappings]
            )
            avg_dist = candidate_dists.mean(axis=0)
            avg_dist = avg_dist / avg_dist.sum()
            obs_sig = self.rng.choice(N_SIG, p=avg_dist)

            obs_dmcue = self.rng.choice(N_DMCUE, p=self.A_dmcue[:, concept, domain])

            assoc_seq = []
            for step in range(MAX_STEPS):
                A_assoc = build_A_association(step)
                obs_assoc = self.rng.choice(N_ASSOC, p=A_assoc[:, concept, domain])
                assoc_seq.append(obs_assoc)

            signs.append({
                'name': name,
                'true_concept': concept,
                'true_domain': domain,
                'is_polysemous': True,
                'obs_sig': obs_sig,
                'obs_dmcue': obs_dmcue,
                'assoc_sequence': assoc_seq,
            })

        # Unambiguous signs -- concept and domain always match
        n_unambig = self.n_total - self.n_polysemous
        for _ in range(n_unambig):
            concept = self.rng.choice(N_CONCEPTS - 1)  # exclude OTHER
            domain = CONCEPT_DOM_MAP[concept]           # canonical match

            obs_sig = self.rng.choice(N_SIG, p=self.A_sig[:, concept, domain])
            obs_dmcue = self.rng.choice(N_DMCUE, p=self.A_dmcue[:, concept, domain])

            assoc_seq = []
            for step in range(MAX_STEPS):
                A_assoc = build_A_association(step)
                obs_assoc = self.rng.choice(N_ASSOC, p=A_assoc[:, concept, domain])
                assoc_seq.append(obs_assoc)

            signs.append({
                'name': f'Sign_{len(signs)}',
                'true_concept': concept,
                'true_domain': domain,
                'is_polysemous': False,
                'obs_sig': obs_sig,
                'obs_dmcue': obs_dmcue,
                'assoc_sequence': assoc_seq,
            })

        self.rng.shuffle(signs)
        return signs


# ============================================================
# Baselines
# ============================================================
class NaiveMLAssigner:
    """Assigns concept using signifier features only (no context)."""

    def __init__(self, A_sig):
        self.A_sig = A_sig

    def classify(self, obs_sig):
        # Marginalize over domains
        lik = self.A_sig[obs_sig, :, :].mean(axis=-1)
        probs = lik / (lik.sum() + 1e-16)
        pred = int(np.argmax(probs))
        return pred, float(probs[pred])


class RuleBasedContextMatcher:
    """Uses domain cue -> concept mapping (simple heuristic)."""

    def classify(self, obs_sig, obs_dmcue):
        domain_to_concept = {
            ASTRONOMY: PLANET,
            CHEMISTRY: ELEMENT,
            MYTHOLOGY: DEITY,
            AUTOMOTIVE: CAR_BRAND,
        }
        if obs_dmcue < N_DOMAINS:
            return domain_to_concept[obs_dmcue], 0.70
        # Fallback: use signifier
        sig_to_concept = {0: PLANET, 1: ELEMENT, 2: DEITY, 3: CAR_BRAND}
        if obs_sig < 4:
            return sig_to_concept[obs_sig], 0.50
        return OTHER, 0.30


# ============================================================
# Run experiment
# ============================================================
CONFIDENCE_THRESHOLD = 0.70


def run_experiment_2(seed=42):
    print("=== Experiment 2: Ambiguity Resolution ===")

    gen = SignGenerator(seed=seed, n_polysemous=15, n_total=50)
    signs = gen.generate()

    D = build_D()
    B = build_B()
    C = build_C()
    A_sig = build_A_signifier()
    A_dmcue = build_A_domain_cue()

    naive_ml = NaiveMLAssigner(A_sig)
    rule_matcher = RuleBasedContextMatcher()

    results = {
        'ai': {'predictions': [], 'correct': [], 'steps_used': [],
                'vfe_traces': [], 'belief_traces': [],
                'accuracy_at_step': [[] for _ in range(MAX_STEPS + 1)]},
        'naive_ml': {'predictions': [], 'correct': []},
        'rule': {'predictions': [], 'correct': []},
        'signs': signs,
    }

    mercury_idx = None

    for i, sign in enumerate(signs):
        true_concept = sign['true_concept']

        # --- Active inference: progressive context accumulation ---
        A_assoc_0 = build_A_association(step=0)

        ai_wave = InferenceWave(
            A_np=[A_sig, A_assoc_0, A_dmcue],
            B_np=B, C_np=C, D_np=D,
            num_controls=[N_ACTIONS, 1],
            control_fac_idx=[0],
            learn=False, gamma=6.0, seed=seed + i,
            use_states_info_gain=True,
        )

        # Step 0: signifier + null association + domain cue
        # This is the only step where signifier and domain cue are informative.
        obs = [sign['obs_sig'], N_ASSOC - 1, sign['obs_dmcue']]
        qs_np, _, _, _ = ai_wave.infer(obs)
        vfe_trace = [ai_wave.vfe_history[-1]]
        belief_trace = [qs_np[0].copy()]

        # Record accuracy at step 0
        pred_step0 = int(np.argmax(qs_np[0]))
        results['ai']['accuracy_at_step'][0].append(pred_step0 == true_concept)

        # Check confidence threshold for "commit" decision
        commit_step = 0 if float(max(qs_np[0])) >= CONFIDENCE_THRESHOLD else -1

        # Uniform A matrices for modalities already incorporated in the prior.
        # At steps 1+, only the association observation is new evidence.
        # The signifier and domain cue are already in the empirical prior
        # from step 0 and should not be re-counted.
        A_sig_uniform = np.ones_like(A_sig) / N_SIG
        A_dmcue_uniform = np.ones_like(A_dmcue) / N_DMCUE

        # Progressive context accumulation (always run all steps)
        for step in range(MAX_STEPS):
            A_assoc_new = build_A_association(step + 1)
            ai_wave.A_np[0] = A_sig_uniform    # signifier: already in prior
            ai_wave.A_np[1] = A_assoc_new      # association: new evidence
            ai_wave.A_np[2] = A_dmcue_uniform  # domain cue: already in prior

            obs = [sign['obs_sig'], sign['assoc_sequence'][step], sign['obs_dmcue']]
            qs_np, _, _, _ = ai_wave.infer(obs, empirical_prior=qs_np)
            vfe_trace.append(ai_wave.vfe_history[-1])
            belief_trace.append(qs_np[0].copy())

            step_pred = int(np.argmax(qs_np[0]))
            results['ai']['accuracy_at_step'][step + 1].append(
                step_pred == true_concept
            )

            if commit_step < 0 and float(max(qs_np[0])) >= CONFIDENCE_THRESHOLD:
                commit_step = step + 1

        if commit_step < 0:
            commit_step = MAX_STEPS

        ai_pred = int(np.argmax(qs_np[0]))
        ai_correct = (ai_pred == true_concept)
        results['ai']['predictions'].append(ai_pred)
        results['ai']['correct'].append(ai_correct)
        results['ai']['steps_used'].append(commit_step)
        results['ai']['vfe_traces'].append(vfe_trace)
        results['ai']['belief_traces'].append(belief_trace)

        if sign['name'] == 'Mercury' and mercury_idx is None:
            mercury_idx = i

        # --- Naive ML (signifier only, no context) ---
        ml_pred, _ = naive_ml.classify(sign['obs_sig'])
        results['naive_ml']['predictions'].append(ml_pred)
        results['naive_ml']['correct'].append(ml_pred == true_concept)

        # --- Rule-based (domain cue + signifier) ---
        rule_pred, _ = rule_matcher.classify(sign['obs_sig'], sign['obs_dmcue'])
        results['rule']['predictions'].append(rule_pred)
        results['rule']['correct'].append(rule_pred == true_concept)

    # Summary
    is_poly = np.array([s['is_polysemous'] for s in signs])
    for method in ['ai', 'naive_ml', 'rule']:
        correct = np.array(results[method]['correct'])
        results[method]['acc_all'] = correct.mean()
        results[method]['acc_poly'] = correct[is_poly].mean() if is_poly.any() else 0
        results[method]['acc_unambig'] = correct[~is_poly].mean() if (~is_poly).any() else 0

    print(f"  AI:       all={results['ai']['acc_all']:.3f}, "
          f"poly={results['ai']['acc_poly']:.3f}, "
          f"unambig={results['ai']['acc_unambig']:.3f}")
    print(f"  Naive ML: all={results['naive_ml']['acc_all']:.3f}, "
          f"poly={results['naive_ml']['acc_poly']:.3f}")
    print(f"  Rule:     all={results['rule']['acc_all']:.3f}, "
          f"poly={results['rule']['acc_poly']:.3f}")
    print(f"  AI mean steps: {np.mean(results['ai']['steps_used']):.1f}")

    results['is_polysemous'] = is_poly
    # Fall back to first polysemous sign if Mercury not found
    if mercury_idx is None:
        poly_indices = np.where(is_poly)[0]
        if len(poly_indices) > 0:
            mercury_idx = int(poly_indices[0])
    results['mercury_idx'] = mercury_idx

    return results


# ============================================================
# Figure generation
# ============================================================
def plot_experiment_2(results):
    setup_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    signs = results['signs']
    is_poly = results['is_polysemous']

    # (a) Accuracy vs context depth (step-by-step)
    ax = axes[0, 0]
    step_range = np.arange(MAX_STEPS + 1)

    # AI accuracy at each step
    ai_acc_per_step = []
    ai_acc_poly_per_step = []
    for s in range(MAX_STEPS + 1):
        step_correct = np.array(results['ai']['accuracy_at_step'][s])
        ai_acc_per_step.append(step_correct.mean())
        if is_poly.any():
            ai_acc_poly_per_step.append(step_correct[is_poly].mean())
        else:
            ai_acc_poly_per_step.append(0)

    ax.plot(step_range, ai_acc_per_step, 'b-o', label='AI (all)',
            linewidth=1.5, markersize=4)
    ax.plot(step_range, ai_acc_poly_per_step, 'r-s',
            label='AI (polysemous)', linewidth=1.5, markersize=4)
    ax.axhline(y=results['naive_ml']['acc_all'], color='darkorange',
               linestyle='--', label='Naive ML (signifier only)', linewidth=1.5)
    ax.axhline(y=results['rule']['acc_all'], color='gray',
               linestyle=':', label='Rule-based (domain cue)', linewidth=1.5)
    ax.set_xlabel('Context depth (associations examined)')
    ax.set_ylabel('Accuracy')
    ax.set_title('(a) Accuracy vs context depth')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)

    # (b) Belief evolution for Mercury
    ax = axes[0, 1]
    mercury_idx = results['mercury_idx']
    if mercury_idx is not None:
        beliefs = results['ai']['belief_traces'][mercury_idx]
        steps = range(len(beliefs))
        for c in range(N_CONCEPTS):
            vals = [b[c] for b in beliefs]
            ax.plot(steps, vals, marker='o', markersize=4,
                    label=CONCEPT_NAMES[c], linewidth=1.5)
        true_c = signs[mercury_idx]['true_concept']
        ax.set_xlabel('Inference step')
        ax.set_ylabel('Posterior P(concept)')
        ax.set_title(f'(b) Belief evolution: "{signs[mercury_idx]["name"]}" '
                      f'(true: {CONCEPT_NAMES[true_c]})')
        ax.legend(fontsize=7)
        ax.set_ylim(-0.02, 1.02)
    else:
        ax.text(0.5, 0.5, 'No polysemous example found',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(b) Belief evolution')

    # (c) VFE over steps
    ax = axes[1, 0]
    max_len = max(len(t) for t in results['ai']['vfe_traces'])
    vfe_padded = np.full((len(results['ai']['vfe_traces']), max_len), np.nan)
    for i, trace in enumerate(results['ai']['vfe_traces']):
        vfe_padded[i, :len(trace)] = trace

    if is_poly.any():
        mean_vfe_poly = np.nanmean(vfe_padded[is_poly], axis=0)
        ax.plot(range(max_len), mean_vfe_poly, 'r--', label='Polysemous', linewidth=1.5)
    if (~is_poly).any():
        mean_vfe_unambig = np.nanmean(vfe_padded[~is_poly], axis=0)
        ax.plot(range(max_len), mean_vfe_unambig, 'b:', label='Unambiguous', linewidth=1.5)
    mean_vfe_all = np.nanmean(vfe_padded, axis=0)
    ax.plot(range(max_len), mean_vfe_all, 'k-', label='All Signs', linewidth=2)

    ax.set_xlabel('Inference step')
    ax.set_ylabel(r'Mean VFE ($\mathcal{F}$)')
    ax.set_title('(c) VFE over progressive context accumulation')
    ax.legend(fontsize=8)

    # (d) Accuracy by ambiguity level
    ax = axes[1, 1]
    methods = ['AI', 'Naive ML', 'Rule-based']
    x = np.arange(len(methods))
    width = 0.3

    acc_poly = [results['ai']['acc_poly'], results['naive_ml']['acc_poly'],
                results['rule']['acc_poly']]
    acc_unambig = [results['ai']['acc_unambig'], results['naive_ml']['acc_unambig'],
                   results['rule']['acc_unambig']]

    ax.bar(x - width/2, acc_poly, width, label='Polysemous', color='coral')
    ax.bar(x + width/2, acc_unambig, width, label='Unambiguous', color='mediumseagreen')
    ax.set_ylabel('Accuracy')
    ax.set_title('(d) Accuracy by ambiguity level')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    save_figure(fig, 'exp2_disambiguation')
    return fig


if __name__ == '__main__':
    results = run_experiment_2()
    plot_experiment_2(results)
