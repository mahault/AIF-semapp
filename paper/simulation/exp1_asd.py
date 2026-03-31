"""
exp1_asd.py — Experiment 1: Artifact Standard Detection (ASD)
=============================================================

Claim: Even the simplest SemApp operation benefits from active inference.

A single Wave receives Artifacts and infers file standards from noisy cues.
  - Hidden state: 1 factor, 6 states (NITF, JPEG2000, GeoTIFF, PDF, XML, UNKNOWN)
  - Observations: 4 modalities (magic number, file extension, parse result, file size)
  - Actions: 2 — classify vs request_parse
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

# Observation modalities
# Magic number: 7 outcomes (6 standards + null)
N_MAGIC = 7
# Extension: 8 outcomes (.ntf, .jp2, .tif, .pdf, .xml, .other, .missing, null)
N_EXT = 8
# Parse result: 5 outcomes (success, partial, fail, timeout, null)
N_PARSE = 5
# File size: 5 outcomes (tiny, small, medium, large, null)
N_SIZE = 5

# Actions
CLASSIFY, REQUEST_PARSE = 0, 1
N_ACTIONS = 2


# ============================================================
# Generative model builders
# ============================================================
def build_A_magic():
    """P(magic_obs | standard). Shape (7, 6).
    Magic bytes are diagnostic but can overlap (GeoTIFF/TIFF share bytes)."""
    # Rows: magic_NITF, magic_JP2, magic_TIFF, magic_PDF, magic_XML, magic_other, null
    # Cols: NITF, JPEG2000, GeoTIFF, PDF, XML, UNKNOWN
    A = np.array([
        [0.85, 0.01, 0.01, 0.01, 0.01, 0.02],  # magic_NITF
        [0.01, 0.82, 0.01, 0.01, 0.01, 0.02],  # magic_JP2
        [0.02, 0.01, 0.65, 0.01, 0.01, 0.02],  # magic_TIFF (shared with GeoTIFF)
        [0.01, 0.01, 0.01, 0.85, 0.01, 0.02],  # magic_PDF
        [0.01, 0.01, 0.01, 0.01, 0.80, 0.02],  # magic_XML
        [0.05, 0.09, 0.26, 0.06, 0.11, 0.80],  # other/ambiguous
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.10],  # null (no magic found)
    ])
    # Normalize columns
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_ext():
    """P(ext_obs | standard). Shape (8, 6).
    Extensions can be wrong (e.g., .tif for GeoTIFF vs plain TIFF)."""
    # Rows: .ntf, .jp2, .tif, .pdf, .xml, .other, .missing, null
    # Cols: NITF, JPEG2000, GeoTIFF, PDF, XML, UNKNOWN
    A = np.array([
        [0.80, 0.01, 0.01, 0.01, 0.01, 0.03],  # .ntf
        [0.01, 0.78, 0.01, 0.01, 0.01, 0.03],  # .jp2
        [0.02, 0.02, 0.70, 0.01, 0.01, 0.03],  # .tif (ambiguous for GeoTIFF)
        [0.01, 0.01, 0.01, 0.82, 0.01, 0.03],  # .pdf
        [0.01, 0.01, 0.01, 0.01, 0.80, 0.03],  # .xml
        [0.08, 0.10, 0.15, 0.08, 0.08, 0.50],  # .other (wrong extension)
        [0.05, 0.05, 0.09, 0.04, 0.06, 0.30],  # .missing
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.05],  # null
    ])
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_parse():
    """P(parse_obs | standard). Shape (5, 6).
    Parse attempts succeed differently for each standard.
    Note: 'null' outcome dominates before parse is requested."""
    # Rows: success, partial, fail, timeout, null
    # Cols: NITF, JPEG2000, GeoTIFF, PDF, XML, UNKNOWN
    A = np.array([
        [0.80, 0.75, 0.70, 0.82, 0.78, 0.05],  # success
        [0.10, 0.12, 0.15, 0.08, 0.10, 0.15],  # partial
        [0.05, 0.06, 0.08, 0.05, 0.07, 0.50],  # fail
        [0.03, 0.05, 0.05, 0.03, 0.03, 0.20],  # timeout
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.10],  # null (not yet parsed)
    ])
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_parse_null():
    """Parse observation matrix when no parse has been requested.
    Almost always returns null (uninformative)."""
    A = np.array([
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # success
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # partial
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # fail
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],  # timeout
        [0.92, 0.92, 0.92, 0.92, 0.92, 0.92],  # null
    ])
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_A_size():
    """P(size_obs | standard). Shape (5, 6).
    File size ranges give weak evidence about format."""
    # Rows: tiny(<1K), small(1K-100K), medium(100K-10M), large(>10M), null
    # Cols: NITF, JPEG2000, GeoTIFF, PDF, XML, UNKNOWN
    A = np.array([
        [0.02, 0.02, 0.02, 0.05, 0.15, 0.15],  # tiny
        [0.08, 0.10, 0.05, 0.20, 0.40, 0.20],  # small
        [0.35, 0.45, 0.35, 0.45, 0.30, 0.30],  # medium
        [0.50, 0.38, 0.53, 0.25, 0.10, 0.15],  # large
        [0.05, 0.05, 0.05, 0.05, 0.05, 0.20],  # null
    ])
    A = A / A.sum(axis=0, keepdims=True)
    return A


def build_B():
    """Transition matrix: identity (standard doesn't change). Shape (6, 6, 2).
    Two actions but neither changes the hidden state."""
    B = np.stack([np.eye(N_STANDARDS)] * N_ACTIONS, axis=-1)
    return B


def build_C():
    """Preference vectors for each modality."""
    # Magic: prefer clear identifications
    C_magic = np.array([1.0, 1.0, 0.8, 1.0, 1.0, -0.3, -0.5])
    # Extension: prefer clear matches
    C_ext = np.array([0.5, 0.5, 0.5, 0.5, 0.5, -0.2, -0.3, -0.1])
    # Parse: prefer success, penalize failure/timeout
    C_parse = np.array([1.0, 0.3, -1.5, -1.0, -0.1])
    # Size: neutral
    C_size = np.array([0.0, 0.0, 0.0, 0.0, -0.1])
    return [C_magic, C_ext, C_parse, C_size]


def build_D():
    """Empirical prior over file standards."""
    # NITF and GeoTIFF are common in IC context
    D = np.array([0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
    return D / D.sum()


# ============================================================
# Artifact generator
# ============================================================
class ArtifactGenerator:
    """Generates synthetic artifacts with noisy observations."""

    def __init__(self, seed=42, ambiguity_rate=0.30):
        self.rng = np.random.RandomState(seed)
        self.A_magic = build_A_magic()
        self.A_ext = build_A_ext()
        self.A_parse = build_A_parse()
        self.A_size = build_A_size()
        self.D = build_D()
        self.ambiguity_rate = ambiguity_rate

    def generate(self, n=500):
        """Generate n artifacts with observations.

        Returns
        -------
        artifacts : list of dict
            Each with 'true_standard', 'is_ambiguous', 'obs_magic', 'obs_ext',
            'obs_size', 'obs_parse_if_requested'.
        """
        artifacts = []
        for i in range(n):
            true_std = self.rng.choice(N_STANDARDS, p=self.D)

            # Decide if ambiguous
            is_ambiguous = self.rng.random() < self.ambiguity_rate

            if is_ambiguous:
                # Create conflicting cues
                obs_magic = self._sample_confusing_magic(true_std)
                obs_ext = self._sample_confusing_ext(true_std)
            else:
                obs_magic = self.rng.choice(N_MAGIC, p=self.A_magic[:, true_std])
                obs_ext = self.rng.choice(N_EXT, p=self.A_ext[:, true_std])

            obs_size = self.rng.choice(N_SIZE, p=self.A_size[:, true_std])

            # Parse result is only available if requested
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
        """Sample magic bytes that may point to wrong standard."""
        # 40% chance of correct, 40% wrong standard, 20% other/null
        r = self.rng.random()
        if r < 0.40:
            return self.rng.choice(N_MAGIC, p=self.A_magic[:, true_std])
        elif r < 0.80:
            wrong = self.rng.choice([s for s in range(N_STANDARDS) if s != true_std])
            return self.rng.choice(N_MAGIC, p=self.A_magic[:, wrong])
        else:
            return self.rng.choice([5, 6])  # other or null

    def _sample_confusing_ext(self, true_std):
        """Sample extension that may conflict."""
        r = self.rng.random()
        if r < 0.35:
            return self.rng.choice(N_EXT, p=self.A_ext[:, true_std])
        elif r < 0.75:
            wrong = self.rng.choice([s for s in range(N_STANDARDS) if s != true_std])
            return self.rng.choice(N_EXT, p=self.A_ext[:, wrong])
        else:
            return self.rng.choice([5, 6, 7])  # other, missing, null


# ============================================================
# Rule-based classifier (SemApp's current ASD approach)
# ============================================================
def build_rule_based():
    """Priority chain: magic > extension > default to UNKNOWN."""

    # Maps magic obs index to standard
    magic_map = {0: NITF, 1: JPEG2000, 2: GEOTIFF, 3: PDF, 4: XML}
    # Maps ext obs index to standard
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

    def rule_parse(obs):
        """If parse was done and succeeded, use size heuristic."""
        p = obs[2]
        if p == 0:  # success
            return (UNKNOWN, 0.50)  # parse succeeded but doesn't tell us which standard
        return None

    def rule_default(obs):
        return (UNKNOWN, 0.30)

    return RuleBasedClassifier([rule_magic, rule_ext, rule_parse, rule_default])


# ============================================================
# Run experiment
# ============================================================
def run_experiment_1(seed=42):
    """Run Experiment 1: Artifact Standard Detection.

    Returns
    -------
    results : dict with keys 'ai', 'rule', 'ml', each containing metrics.
    """
    print("=== Experiment 1: Artifact Standard Detection ===")
    rng = np.random.RandomState(seed)

    # Generate artifacts
    gen = ArtifactGenerator(seed=seed, ambiguity_rate=0.30)
    artifacts = gen.generate(n=500)

    # Build generative model
    A_magic = build_A_magic()
    A_ext = build_A_ext()
    A_parse = build_A_parse()
    A_parse_null = build_A_parse_null()
    A_size = build_A_size()
    C = build_C()
    D = build_D()
    B = build_B()

    # --- Active Inference agent ---
    # Use parse_null initially; parse becomes informative after request_parse
    ai_wave = InferenceWave(
        A_np=[A_magic, A_ext, A_parse_null, A_size],
        B_np=[B],
        C_np=C,
        D_np=[D],
        num_controls=[N_ACTIONS],
        control_fac_idx=[0],
        learn=False,
        gamma=8.0,
        seed=seed,
        use_states_info_gain=True,
    )

    # --- ML baseline (uses null-parse model like AI Phase 1) ---
    ml_clf = MLClassifier([A_magic, A_ext, A_parse_null, A_size])

    # --- Rule-based baseline ---
    rule_clf = build_rule_based()

    results = {
        'ai': {'predictions': [], 'confidences': [], 'correct': [],
                'vfe': [], 'actions': [], 'belief_examples': []},
        'rule': {'predictions': [], 'confidences': [], 'correct': []},
        'ml': {'predictions': [], 'confidences': [], 'correct': []},
    }

    # Track a specific ambiguous case for belief evolution plot
    geotiff_example_idx = None

    for i, art in enumerate(artifacts):
        true_std = art['true_standard']
        obs_initial = [art['obs_magic'], art['obs_ext'], N_PARSE - 1, art['obs_size']]
        # obs_initial has null parse (index 4)

        # --- Active inference: two-phase ---
        # Phase 1: observe cheap cues
        ai_wave.reset_history()
        qs_np, q_pi, efe, action = ai_wave.infer(obs_initial)
        vfe1 = ai_wave.vfe_history[-1]

        # Phase 2: if agent chose request_parse, update with parse result
        if action == REQUEST_PARSE:
            obs_with_parse = [art['obs_magic'], art['obs_ext'],
                              art['obs_parse_if_requested'], art['obs_size']]
            # Update the Wave's A matrix view for this step
            old_A = ai_wave.A_np[2]
            ai_wave.A_np[2] = A_parse
            qs_np, _, _, _ = ai_wave.infer(obs_with_parse, empirical_prior=[qs_np[0]])
            ai_wave.A_np[2] = old_A  # restore
        else:
            # Classify based on Phase 1 only
            pass

        ai_pred = int(np.argmax(qs_np[0]))
        ai_conf = float(qs_np[0][ai_pred])
        ai_correct = (ai_pred == true_std)

        results['ai']['predictions'].append(ai_pred)
        results['ai']['confidences'].append(ai_conf)
        results['ai']['correct'].append(ai_correct)
        results['ai']['vfe'].append(ai_wave.vfe_history[-1])
        results['ai']['actions'].append(action)

        # Track GeoTIFF ambiguous example
        if (true_std == GEOTIFF and art['is_ambiguous']
                and geotiff_example_idx is None):
            geotiff_example_idx = i
            results['ai']['belief_examples'] = list(ai_wave.belief_history)

        # --- Rule-based (no parse access — uses cheap cues only) ---
        obs_for_rule = [art['obs_magic'], art['obs_ext'],
                        N_PARSE - 1, art['obs_size']]  # null parse
        rule_pred, rule_conf = rule_clf.classify(obs_for_rule)
        rule_correct = (rule_pred == true_std)
        results['rule']['predictions'].append(rule_pred)
        results['rule']['confidences'].append(rule_conf)
        results['rule']['correct'].append(rule_correct)

        # --- ML baseline (no parse access — uses cheap cues only) ---
        obs_for_ml = [art['obs_magic'], art['obs_ext'],
                      N_PARSE - 1, art['obs_size']]  # null parse
        ml_pred, ml_conf = ml_clf.classify(obs_for_ml)
        ml_correct = (ml_pred == true_std)
        results['ml']['predictions'].append(ml_pred)
        results['ml']['confidences'].append(ml_conf)
        results['ml']['correct'].append(ml_correct)

    # Compute summary metrics
    artifacts_arr = artifacts
    is_ambiguous = np.array([a['is_ambiguous'] for a in artifacts_arr])

    for method in ['ai', 'rule', 'ml']:
        correct = np.array(results[method]['correct'])
        results[method]['acc_all'] = correct.mean()
        results[method]['acc_ambiguous'] = correct[is_ambiguous].mean()
        results[method]['acc_clear'] = correct[~is_ambiguous].mean()

    print(f"  AI:   all={results['ai']['acc_all']:.3f}, "
          f"ambig={results['ai']['acc_ambiguous']:.3f}, "
          f"clear={results['ai']['acc_clear']:.3f}")
    print(f"  Rule: all={results['rule']['acc_all']:.3f}, "
          f"ambig={results['rule']['acc_ambiguous']:.3f}, "
          f"clear={results['rule']['acc_clear']:.3f}")
    print(f"  ML:   all={results['ml']['acc_all']:.3f}, "
          f"ambig={results['ml']['acc_ambiguous']:.3f}, "
          f"clear={results['ml']['acc_clear']:.3f}")

    # Store metadata
    results['artifacts'] = artifacts
    results['is_ambiguous'] = is_ambiguous
    results['geotiff_example_idx'] = geotiff_example_idx

    return results


# ============================================================
# Figure generation
# ============================================================
def plot_experiment_1(results):
    """Generate 2×2 figure for Experiment 1."""
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

    bars1 = ax.bar(x - width, acc_all, width, label='All', color='steelblue')
    bars2 = ax.bar(x, acc_amb, width, label='Ambiguous', color='coral')
    bars3 = ax.bar(x + width, acc_clr, width, label='Clear', color='mediumseagreen')

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
