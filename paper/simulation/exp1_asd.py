"""
exp1_asd.py — Experiment 1: Artifact Standard Detection (ASD)
=============================================================

Claim: Even the simplest SemApp operation benefits from active inference.

A single Wave receives Artifacts and infers file standards from noisy cues.
  - Hidden states: 2 factors
    - Factor 0: parse_status (2 states: not_parsed, parsed) — controlled by action
    - Factor 1: standard (6 states: NITF, JPEG2000, GeoTIFF, PDF, XML, UNKNOWN)
  - Observations: 4 modalities (magic number, file extension, parse result, file size)
  - Actions: 2 — classify (keep not_parsed) vs request_parse (transition to parsed)
  - The parse modality is uninformative when not_parsed, diagnostic when parsed.
    This lets EFE correctly value the epistemic benefit of requesting a parse.
  - Comparison: active inference vs rule-based priority chain vs maximum-likelihood
  - N = 500 artifacts, ~30% ambiguous (conflicting cues)
"""

import numpy as np
from common import (
    InferenceWave, RuleBasedClassifier, MLClassifier,
    confidence_calibration, discrete_vfe, log_s,
    setup_figure_style, save_figure, FIG_DIR,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Constants
# ============================================================
# File standards
NITF, JPEG2000, GEOTIFF, PDF, XML, UNKNOWN = range(6)
N_STANDARDS = 6
STANDARD_NAMES = ['NITF', 'JPEG2000', 'GeoTIFF', 'PDF', 'XML', 'UNKNOWN']

# Parse status (hidden factor 0)
NOT_PARSED, PARSED = 0, 1
N_PARSE_STATUS = 2

# Observation modalities
N_MAGIC = 7   # 6 standards + null
N_EXT = 8     # .ntf, .jp2, .tif, .pdf, .xml, .other, .missing, null
N_PARSE = 7   # nitf_match, jp2_match, geotiff_match, pdf_match, xml_match, fail, null
N_SIZE = 5    # tiny, small, medium, large, null

# Actions (control factor 0: parse_status)
CLASSIFY, REQUEST_PARSE = 0, 1
N_ACTIONS = 2


# ============================================================
# Generative model builders (1-factor versions for baselines)
# ============================================================
def build_A_magic_1f():
    """P(magic_obs | standard). Shape (7, 6)."""
    A = np.array([
        [0.85, 0.01, 0.01, 0.01, 0.01, 0.02],  # magic_NITF
        [0.01, 0.82, 0.01, 0.01, 0.01, 0.02],  # magic_JP2
        [0.02, 0.01, 0.65, 0.01, 0.01, 0.02],  # magic_TIFF (shared w/ GeoTIFF)
        [0.01, 0.01, 0.01, 0.85, 0.01, 0.02],  # magic_PDF
        [0.01, 0.01, 0.01, 0.01, 0.80, 0.02],  # magic_XML
        [0.05, 0.09, 0.26, 0.06, 0.11, 0.80],  # other/ambiguous
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.10],  # null
    ])
    return A / A.sum(axis=0, keepdims=True)


def build_A_ext_1f():
    """P(ext_obs | standard). Shape (8, 6)."""
    A = np.array([
        [0.80, 0.01, 0.01, 0.01, 0.01, 0.03],  # .ntf
        [0.01, 0.78, 0.01, 0.01, 0.01, 0.03],  # .jp2
        [0.02, 0.02, 0.70, 0.01, 0.01, 0.03],  # .tif
        [0.01, 0.01, 0.01, 0.82, 0.01, 0.03],  # .pdf
        [0.01, 0.01, 0.01, 0.01, 0.80, 0.03],  # .xml
        [0.08, 0.10, 0.15, 0.08, 0.08, 0.50],  # .other
        [0.05, 0.05, 0.09, 0.04, 0.06, 0.30],  # .missing
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.05],  # null
    ])
    return A / A.sum(axis=0, keepdims=True)


def build_A_parse_diagnostic():
    """P(parse_obs | standard) when parsed. Shape (7, 6).
    Standard-specific parser matching: try each format's parser, see which
    succeeds. This is how real ASD works — each parser validates headers,
    metadata, and structure specific to its format.
      Rows: nitf_match, jp2_match, geotiff_match, pdf_match, xml_match, fail, null
      Cols: NITF, JP2, GeoTIFF, PDF, XML, UNKNOWN"""
    A = np.array([
        [0.85, 0.02, 0.03, 0.02, 0.01, 0.01],  # nitf_match
        [0.02, 0.82, 0.02, 0.02, 0.01, 0.01],  # jp2_match
        [0.03, 0.02, 0.80, 0.02, 0.01, 0.01],  # geotiff_match
        [0.02, 0.02, 0.02, 0.82, 0.01, 0.01],  # pdf_match
        [0.01, 0.01, 0.01, 0.01, 0.90, 0.01],  # xml_match
        [0.05, 0.09, 0.10, 0.09, 0.04, 0.87],  # fail
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.08],  # null
    ])
    return A / A.sum(axis=0, keepdims=True)


def build_A_size_1f():
    """P(size_obs | standard). Shape (5, 6)."""
    A = np.array([
        [0.02, 0.02, 0.02, 0.05, 0.15, 0.15],  # tiny
        [0.08, 0.10, 0.05, 0.20, 0.40, 0.20],  # small
        [0.35, 0.45, 0.35, 0.45, 0.30, 0.30],  # medium
        [0.50, 0.38, 0.53, 0.25, 0.10, 0.15],  # large
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.20],  # null
    ])
    return A / A.sum(axis=0, keepdims=True)


# ============================================================
# 2-factor A matrices for AI agent: A(obs | parse_status, standard)
# ============================================================
def _expand_A(A_1f):
    """Expand (obs, standard) → (obs, parse_status, standard).
    Non-parse modalities: same for both parse states."""
    n_obs, n_std = A_1f.shape
    A_2f = np.zeros((n_obs, N_PARSE_STATUS, n_std))
    A_2f[:, NOT_PARSED, :] = A_1f
    A_2f[:, PARSED, :] = A_1f
    return A_2f


def build_A_parse_2f():
    """Parse observation model with parse_status factor. Shape (5, 2, 6).
    When not_parsed: uniform (uninformative).
    When parsed: highly diagnostic per standard."""
    A = np.zeros((N_PARSE, N_PARSE_STATUS, N_STANDARDS))
    A[:, NOT_PARSED, :] = 1.0 / N_PARSE          # uniform
    A[:, PARSED, :] = build_A_parse_diagnostic()  # diagnostic
    return A / A.sum(axis=0, keepdims=True)


def build_B_2f():
    """Transition matrices for 2 factors.
    Factor 0 (parse_status): controlled by action.
    Factor 1 (standard): identity (doesn't change)."""
    B_parse = np.zeros((N_PARSE_STATUS, N_PARSE_STATUS, N_ACTIONS))
    # CLASSIFY: parse status unchanged
    B_parse[:, :, CLASSIFY] = np.eye(N_PARSE_STATUS)
    # REQUEST_PARSE: always transition to parsed
    B_parse[PARSED, :, REQUEST_PARSE] = 1.0
    # Factor 1: standard doesn't change (identity, 1 dummy action)
    B_standard = np.eye(N_STANDARDS).reshape(N_STANDARDS, N_STANDARDS, 1)
    return [B_parse, B_standard]


def build_C():
    """Preference vectors for each modality."""
    C_magic = np.array([1.0, 1.0, 0.8, 1.0, 1.0, -0.3, -0.5])
    C_ext = np.array([0.5, 0.5, 0.5, 0.5, 0.5, -0.2, -0.3, -0.1])
    C_parse = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.5, -0.1])  # match=good, fail=bad
    C_size = np.array([0.0, 0.0, 0.0, 0.0, -0.1])
    return [C_magic, C_ext, C_parse, C_size]


def build_D_2f():
    """Priors for 2 factors.
    Factor 0: parse_status — start not_parsed.
    Factor 1: standard — empirical prior."""
    D_parse = np.array([0.99, 0.01])
    D_standard = np.array([0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
    D_standard = D_standard / D_standard.sum()
    return [D_parse, D_standard]


def build_D_standard():
    """1-factor prior over standards (for baselines)."""
    D = np.array([0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
    return D / D.sum()


# ============================================================
# Artifact generator
# ============================================================
class ArtifactGenerator:
    """Generates synthetic artifacts with noisy observations."""

    def __init__(self, seed=42, ambiguity_rate=0.30):
        self.rng = np.random.RandomState(seed)
        self.A_magic = build_A_magic_1f()
        self.A_ext = build_A_ext_1f()
        self.A_parse = build_A_parse_diagnostic()
        self.A_size = build_A_size_1f()
        self.D = build_D_standard()
        self.ambiguity_rate = ambiguity_rate

    def generate(self, n=500):
        artifacts = []
        for i in range(n):
            true_std = self.rng.choice(N_STANDARDS, p=self.D)
            is_ambiguous = self.rng.random() < self.ambiguity_rate

            if is_ambiguous:
                obs_magic = self._sample_confusing_magic(true_std)
                obs_ext = self._sample_confusing_ext(true_std)
            else:
                obs_magic = self.rng.choice(N_MAGIC, p=self.A_magic[:, true_std])
                obs_ext = self.rng.choice(N_EXT, p=self.A_ext[:, true_std])

            obs_size = self.rng.choice(N_SIZE, p=self.A_size[:, true_std])
            obs_parse = self.rng.choice(N_PARSE, p=self.A_parse[:, true_std])

            artifacts.append({
                'true_standard': true_std,
                'is_ambiguous': is_ambiguous,
                'obs_magic': obs_magic,
                'obs_ext': obs_ext,
                'obs_size': obs_size,
                'obs_parse_if_requested': obs_parse,
            })
        return artifacts

    def _sample_confusing_magic(self, true_std):
        """Ambiguous magic: corrupted, missing, or genuinely overlapping bytes.
        No wrong-standard sampling — ambiguity means uncertainty, not deception."""
        r = self.rng.random()
        if r < 0.30:
            return self.rng.choice(N_MAGIC, p=self.A_magic[:, true_std])
        else:
            return self.rng.choice([5, 6])  # other (corrupted) or null (missing)

    def _sample_confusing_ext(self, true_std):
        """Ambiguous extension: missing, wrong, or generic."""
        r = self.rng.random()
        if r < 0.25:
            return self.rng.choice(N_EXT, p=self.A_ext[:, true_std])
        else:
            return self.rng.choice([5, 6, 7])  # other, missing, null


# ============================================================
# Rule-based classifier (SemApp's current ASD approach)
# ============================================================
def build_rule_based():
    """Priority chain: magic > extension > default to UNKNOWN."""
    magic_map = {0: NITF, 1: JPEG2000, 2: GEOTIFF, 3: PDF, 4: XML}
    ext_map = {0: NITF, 1: JPEG2000, 2: GEOTIFF, 3: PDF, 4: XML}

    def rule_magic(obs):
        m = obs[0]
        if m in magic_map:
            return (magic_map[m], 0.90)
        return None

    def rule_ext(obs):
        e = obs[1]
        if e in ext_map:
            return (ext_map[e], 0.70)
        return None

    def rule_default(obs):
        return (UNKNOWN, 0.30)

    return RuleBasedClassifier([rule_magic, rule_ext, rule_default])


# ============================================================
# Run experiment
# ============================================================
def run_experiment_1(seed=42):
    """Run Experiment 1: Artifact Standard Detection."""
    print("=== Experiment 1: Artifact Standard Detection ===")

    # Generate artifacts
    gen = ArtifactGenerator(seed=seed, ambiguity_rate=0.30)
    artifacts = gen.generate(n=500)

    # Build 2-factor generative model for AI agent
    A_magic_1f = build_A_magic_1f()
    A_ext_1f = build_A_ext_1f()
    A_size_1f = build_A_size_1f()

    A_magic_2f = _expand_A(A_magic_1f)
    A_ext_2f = _expand_A(A_ext_1f)
    A_parse_2f = build_A_parse_2f()
    A_size_2f = _expand_A(A_size_1f)

    B_2f = build_B_2f()
    C = build_C()
    D_2f = build_D_2f()

    # --- Active Inference agent (2-factor: parse_status × standard) ---
    ai_wave = InferenceWave(
        A_np=[A_magic_2f, A_ext_2f, A_parse_2f, A_size_2f],
        B_np=B_2f,
        C_np=C,
        D_np=D_2f,
        num_controls=[N_ACTIONS, 1],
        control_fac_idx=[0],
        learn=False,
        gamma=16.0,
        seed=seed,
        use_states_info_gain=True,
    )

    # --- ML baseline (1-factor, no parse access) ---
    # ML sees null parse = uniform, effectively no parse info
    A_parse_null_1f = np.full((N_PARSE, N_STANDARDS), 1.0 / N_PARSE)
    ml_clf = MLClassifier([A_magic_1f, A_ext_1f, A_parse_null_1f, A_size_1f])

    # --- Rule-based baseline ---
    rule_clf = build_rule_based()

    results = {
        'ai': {'predictions': [], 'confidences': [], 'correct': [],
                'vfe': [], 'actions': [], 'parse_rate': 0},
        'rule': {'predictions': [], 'confidences': [], 'correct': []},
        'ml': {'predictions': [], 'confidences': [], 'correct': []},
    }

    n_parsed = 0

    for i, art in enumerate(artifacts):
        true_std = art['true_standard']
        # Initial observation: null parse (index 4)
        obs_initial = [art['obs_magic'], art['obs_ext'], N_PARSE - 1, art['obs_size']]

        # --- Active inference: two-phase ---
        ai_wave.reset_history()

        # Phase 1: observe cheap cues (parse_status = not_parsed)
        qs_np, q_pi, efe, action = ai_wave.infer(obs_initial)

        # Phase 2: if agent chose request_parse, get parse result
        if action == REQUEST_PARSE:
            n_parsed += 1
            obs_with_parse = [art['obs_magic'], art['obs_ext'],
                              art['obs_parse_if_requested'], art['obs_size']]
            # Empirical prior: parse_status → parsed, standard from Phase 1
            emp_prior = [np.array([0.0, 1.0]), qs_np[1].copy()]
            qs_np, _, _, _ = ai_wave.infer(obs_with_parse, empirical_prior=emp_prior)

        # Prediction from standard factor (factor 1)
        ai_pred = int(np.argmax(qs_np[1]))
        ai_conf = float(qs_np[1][ai_pred])
        ai_correct = (ai_pred == true_std)

        results['ai']['predictions'].append(ai_pred)
        results['ai']['confidences'].append(ai_conf)
        results['ai']['correct'].append(ai_correct)
        results['ai']['vfe'].append(ai_wave.vfe_history[-1])
        results['ai']['actions'].append(action)

        # --- Rule-based ---
        obs_for_rule = [art['obs_magic'], art['obs_ext'],
                        N_PARSE - 1, art['obs_size']]
        rule_pred, rule_conf = rule_clf.classify(obs_for_rule)
        rule_correct = (rule_pred == true_std)
        results['rule']['predictions'].append(rule_pred)
        results['rule']['confidences'].append(rule_conf)
        results['rule']['correct'].append(rule_correct)

        # --- ML baseline ---
        obs_for_ml = [art['obs_magic'], art['obs_ext'],
                      N_PARSE - 1, art['obs_size']]
        ml_pred, ml_conf = ml_clf.classify(obs_for_ml)
        ml_correct = (ml_pred == true_std)
        results['ml']['predictions'].append(ml_pred)
        results['ml']['confidences'].append(ml_conf)
        results['ml']['correct'].append(ml_correct)

    # Summary metrics
    is_ambiguous = np.array([a['is_ambiguous'] for a in artifacts])
    results['ai']['parse_rate'] = n_parsed / len(artifacts)

    for method in ['ai', 'rule', 'ml']:
        correct = np.array(results[method]['correct'])
        results[method]['acc_all'] = correct.mean()
        results[method]['acc_ambiguous'] = correct[is_ambiguous].mean()
        results[method]['acc_clear'] = correct[~is_ambiguous].mean()

    print(f"  AI:   all={results['ai']['acc_all']:.3f}, "
          f"ambig={results['ai']['acc_ambiguous']:.3f}, "
          f"clear={results['ai']['acc_clear']:.3f}, "
          f"parse_rate={results['ai']['parse_rate']:.2f}")
    print(f"  Rule: all={results['rule']['acc_all']:.3f}, "
          f"ambig={results['rule']['acc_ambiguous']:.3f}, "
          f"clear={results['rule']['acc_clear']:.3f}")
    print(f"  ML:   all={results['ml']['acc_all']:.3f}, "
          f"ambig={results['ml']['acc_ambiguous']:.3f}, "
          f"clear={results['ml']['acc_clear']:.3f}")

    results['artifacts'] = artifacts
    results['is_ambiguous'] = is_ambiguous

    return results


# ============================================================
# Figure generation
# ============================================================
def plot_experiment_1(results):
    """Generate 2x2 figure for Experiment 1."""
    setup_figure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Accuracy bar chart
    ax = axes[0, 0]
    methods = ['AI (Active\nInference)', 'Rule-Based', 'Maximum\nLikelihood']
    x = np.arange(3)
    width = 0.25

    acc_all = [results[m]['acc_all'] for m in ['ai', 'rule', 'ml']]
    acc_amb = [results[m]['acc_ambiguous'] for m in ['ai', 'rule', 'ml']]
    acc_clr = [results[m]['acc_clear'] for m in ['ai', 'rule', 'ml']]

    ax.bar(x - width, acc_all, width, label='All', color='steelblue')
    ax.bar(x, acc_amb, width, label='Ambiguous', color='coral')
    ax.bar(x + width, acc_clr, width, label='Clear', color='mediumseagreen')

    ax.set_ylabel('Accuracy')
    ax.set_title('(a) Classification accuracy by method and ambiguity')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # (b) Per-standard accuracy comparison
    ax = axes[0, 1]
    true_stds = np.array([a['true_standard'] for a in results['artifacts']])
    std_names_short = [STANDARD_NAMES[s][:6] for s in range(N_STANDARDS)]
    x_std = np.arange(N_STANDARDS)
    width_s = 0.28

    for j, (method, color, label) in enumerate([
        ('ai', 'steelblue', 'AI'),
        ('rule', 'gray', 'Rule'),
        ('ml', 'darkorange', 'ML'),
    ]):
        correct = np.array(results[method]['correct'])
        per_std_acc = []
        for s in range(N_STANDARDS):
            mask = true_stds == s
            per_std_acc.append(correct[mask].mean() if mask.any() else 0)
        ax.bar(x_std + (j - 1) * width_s, per_std_acc, width_s,
               color=color, label=label, alpha=0.8)

    ax.set_xticks(x_std)
    ax.set_xticklabels(std_names_short, fontsize=8, rotation=30)
    ax.set_ylabel('Accuracy')
    ax.set_title('(b) Accuracy by file standard')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # (c) Confidence calibration
    ax = axes[1, 0]
    for method, color, label in [('ai', 'steelblue', 'AI'),
                                  ('ml', 'darkorange', 'ML'),
                                  ('rule', 'gray', 'Rule-based')]:
        centers, accs, counts = confidence_calibration(
            results[method]['confidences'], results[method]['correct'], n_bins=8
        )
        mask = counts > 0
        ax.plot(centers[mask], accs[mask], 'o-', color=color, label=label,
                linewidth=1.5, markersize=5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Predicted confidence')
    ax.set_ylabel('Observed accuracy')
    ax.set_title('(c) Confidence calibration')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # (d) VFE distribution colored by correct/incorrect
    ax = axes[1, 1]
    vfe_arr = np.array(results['ai']['vfe'])
    correct_arr = np.array(results['ai']['correct'])

    ax.hist(vfe_arr[correct_arr], bins=25, alpha=0.6, color='steelblue',
            label='Correct', density=True)
    ax.hist(vfe_arr[~correct_arr], bins=25, alpha=0.6, color='coral',
            label='Incorrect', density=True)
    ax.set_xlabel('Variational Free Energy')
    ax.set_ylabel('Density')
    ax.set_title(r'(d) VFE distribution ($\mathcal{F}$) by outcome')
    ax.legend()

    fig.tight_layout()
    save_figure(fig, 'exp1_asd')
    return fig


if __name__ == '__main__':
    results = run_experiment_1()
    plot_experiment_1(results)
