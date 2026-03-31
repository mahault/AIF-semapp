"""
common.py -- Shared infrastructure for the new SemApp experiment suite
=====================================================================

Implements principled discrete active inference following the pymdp
generative model specification (Heins et al., 2022):
  - Bayesian state inference via variational mean-field coordinate ascent
  - Expected Free Energy (EFE) policy evaluation with pragmatic + epistemic
  - Softmax action selection with policy precision gamma
  - Variational Free Energy (VFE) computation
  - Optional Dirichlet parameter learning

The math matches pymdp exactly; we use numpy for per-item inference
efficiency (avoiding JAX JIT recompilation overhead for small state spaces).

Provides:
  - InferenceWave: active inference agent with pymdp-compatible generative model
  - discrete_vfe(): compute VFE from posterior, likelihood, prior
  - RuleBasedClassifier / MLClassifier: baselines
  - Figure utilities
"""

import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================
FIG_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Numeric helpers
# ============================================================
def log_s(x):
    return np.log(np.maximum(x, 1e-16))


def softmax(x, temp=1.0):
    e = np.exp(temp * (x - x.max()))
    return e / e.sum()


def entropy_H(p):
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + 1e-16)
    return -np.sum(p * log_s(p))


def kl_div(q, p):
    q = np.asarray(q, dtype=np.float64)
    q = q / (q.sum() + 1e-16)
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + 1e-16)
    return np.sum(q * (log_s(q) - log_s(p)))


# ============================================================
# VFE computation: F = E_q[ln q(s)] - E_q[ln p(o,s)]
# ============================================================
def discrete_vfe(qs_list, obs_indices, A_list, D_list):
    """Compute variational free energy under mean-field factorization.

    F = sum_f E_{q_f}[ln q_f(s_f)]        (negative entropy)
      - sum_m E_q[ln A_m(o_m | s)]         (expected log-likelihood)
      - sum_f E_{q_f}[ln D_f(s_f)]         (expected log-prior)
    """
    n_factors = len(qs_list)

    log_likelihood = 0.0
    for m, (A_m, o_m) in enumerate(zip(A_list, obs_indices)):
        logA = log_s(A_m[o_m])
        val = logA
        for f in range(n_factors):
            val = np.tensordot(qs_list[f], val, axes=([0], [0]))
        log_likelihood += float(val)

    log_prior = sum(np.dot(qs_list[f], log_s(D_list[f])) for f in range(n_factors))
    neg_entropy = sum(np.dot(qs_list[f], log_s(qs_list[f])) for f in range(n_factors))

    return float(neg_entropy - log_likelihood - log_prior)


# ============================================================
# InferenceWave -- numpy-based active inference agent
# ============================================================
class InferenceWave:
    """Discrete active inference agent implementing the pymdp generative
    model specification in numpy.

    Generative model: p(o, s) = prod_m A_m(o_m | s) * prod_f D_f(s_f)
    with action-conditioned transitions B_f(s'_f | s_f, a)

    Inference:
      1. State inference: variational mean-field coordinate ascent
         q_f(s_f) \\propto exp{ E_{q_{\\f}}[ln p(o, s)] }
      2. Policy evaluation: Expected Free Energy
         G(a) = -E_{q_a}[ln p(o)] + E_{q_a}[H[p(o|s)]]
         where q_a(s') = B(s'|s,a) q(s) and p(o) = exp(C)
      3. Action selection: pi(a) = sigma(-gamma * G(a))
    """

    def __init__(self, A_np, B_np, C_np, D_np,
                 num_controls, control_fac_idx,
                 learn=False, gamma=8.0, seed=0,
                 use_states_info_gain=True):
        self.rng = np.random.RandomState(seed)

        # Generative model matrices (pymdp-compatible shapes)
        self.A_np = [a.copy() for a in A_np]
        self.B_np = [b.copy() for b in B_np]
        self.C_np = [c.copy() for c in C_np]
        self.D_np = [d.copy() for d in D_np]

        self.num_controls = num_controls
        self.control_fac_idx = control_fac_idx
        self.n_factors = len(D_np)
        self.n_modalities = len(A_np)
        self.gamma = gamma
        self.use_info_gain = use_states_info_gain
        self.learn = learn

        # Dirichlet concentration parameters for learning
        if learn:
            self.pA = [a.copy() * 1.0 for a in A_np]

        # History tracking
        self.belief_history = []
        self.vfe_history = []
        self.action_history = []
        self.obs_history = []

    def infer(self, obs, empirical_prior=None):
        """Run one perception-action cycle.

        Implements the same inference as pymdp.agent.Agent:
          1. infer_states: mean-field variational inference
          2. infer_policies: EFE evaluation
          3. sample_action: softmax selection

        Parameters
        ----------
        obs : list of int
            Observed outcome index per modality.
        empirical_prior : list of numpy arrays or None
            If provided, used in place of D for state inference.

        Returns
        -------
        qs_np : list of numpy arrays
            Posterior marginals per factor.
        q_pi : numpy array
            Policy distribution.
        efe : numpy array
            Expected free energy per action.
        action : int
            Selected action.
        """
        # --- 1. State inference: coordinate ascent ---
        prior = empirical_prior if empirical_prior is not None else self.D_np
        qs = [p.copy() for p in prior]

        for _ in range(8):  # convergence iterations
            for f in range(self.n_factors):
                log_q = log_s(prior[f]).copy()

                for m in range(self.n_modalities):
                    logA = log_s(self.A_np[m][obs[m]])

                    if self.n_factors == 1:
                        log_q += logA
                    elif self.n_factors == 2:
                        if f == 0:
                            log_q += logA @ qs[1]
                        else:
                            log_q += logA.T @ qs[0]
                    else:
                        # General case: contract all factors except f
                        val = logA
                        for f2 in reversed(range(self.n_factors)):
                            if f2 != f:
                                val = np.tensordot(val, qs[f2], axes=([f2 if f2 < f else f2], [0]))
                        log_q += val.flatten() if hasattr(val, 'flatten') else val

                log_q -= log_q.max()
                qs[f] = np.exp(log_q)
                qs[f] /= qs[f].sum() + 1e-16

        # --- 2. Policy evaluation: EFE ---
        n_actions = self.num_controls[0] if self.num_controls else 1
        efe = np.zeros(n_actions)

        for a in range(n_actions):
            # Predicted next-state under action a
            qs_next = [q.copy() for q in qs]
            if len(self.B_np) > 0 and self.B_np[0].ndim == 3:
                B = self.B_np[0]
                if a < B.shape[2]:
                    qs_next[0] = B[:, :, a] @ qs[0]
                    qs_next[0] /= qs_next[0].sum() + 1e-16

            # Pragmatic: E_{q_a}[C_m^T * q_a(o_m)]
            pragmatic = 0.0
            for m in range(self.n_modalities):
                qo = self._predicted_obs(self.A_np[m], qs_next)
                qo /= qo.sum() + 1e-16
                pragmatic += np.dot(qo, self.C_np[m])

            # Epistemic: I(o;s) = H[q(o)] - E_q[H[p(o|s)]]
            epistemic = 0.0
            if self.use_info_gain:
                for m in range(self.n_modalities):
                    qo = self._predicted_obs(self.A_np[m], qs_next)
                    qo /= qo.sum() + 1e-16
                    H_qo = entropy_H(qo)
                    H_cond = self._expected_cond_entropy(self.A_np[m], qs_next)
                    epistemic += H_qo - H_cond

            efe[a] = -pragmatic - epistemic

        # --- 3. Action selection: softmax ---
        q_pi = softmax(-efe, temp=self.gamma)
        action = int(self.rng.choice(n_actions, p=q_pi))

        # Compute VFE for tracking
        vfe = self.compute_vfe(qs, obs)

        # Dirichlet learning update
        if self.learn:
            self._dirichlet_update(qs, obs)

        # Record
        self.belief_history.append([q.copy() for q in qs])
        self.vfe_history.append(vfe)
        self.action_history.append(action)
        self.obs_history.append(obs)

        return qs, q_pi, efe, action

    def _predicted_obs(self, A_m, qs):
        """q(o) = sum_s A(o|s) q(s) — predicted observation distribution."""
        result = A_m.copy()
        for f in range(self.n_factors - 1, -1, -1):
            result = np.tensordot(result, qs[f], axes=([-1], [0]))
        return result

    def _expected_cond_entropy(self, A_m, qs):
        """E_q[H[p(o|s)]] — expected conditional entropy."""
        if self.n_factors == 1:
            H_cols = np.array([entropy_H(A_m[:, s]) for s in range(A_m.shape[1])])
            return np.dot(qs[0], H_cols)
        elif self.n_factors == 2:
            H_cond = 0.0
            for s1 in range(A_m.shape[1]):
                for s2 in range(A_m.shape[2]):
                    H_cond += qs[0][s1] * qs[1][s2] * entropy_H(A_m[:, s1, s2])
            return H_cond
        else:
            from itertools import product as iterproduct
            shapes = [A_m.shape[f + 1] for f in range(self.n_factors)]
            H_cond = 0.0
            for cfg in iterproduct(*[range(s) for s in shapes]):
                col = A_m[(slice(None),) + cfg]
                prob = np.prod([qs[f][cfg[f]] for f in range(self.n_factors)])
                H_cond += prob * entropy_H(col)
            return H_cond

    def _dirichlet_update(self, qs, obs):
        """Dirichlet parameter update (online learning)."""
        for m in range(self.n_modalities):
            o_m = obs[m]
            if self.n_factors == 1:
                self.pA[m][o_m, :] += qs[0] * 0.1
            elif self.n_factors == 2:
                self.pA[m][o_m] += np.outer(qs[0], qs[1]) * 0.1
            self.A_np[m] = self.pA[m] / self.pA[m].sum(axis=0, keepdims=True)

    def compute_vfe(self, qs_np, obs):
        return discrete_vfe(qs_np, obs, self.A_np, self.D_np)

    def reset_history(self):
        self.belief_history = []
        self.vfe_history = []
        self.action_history = []
        self.obs_history = []


# ============================================================
# Baselines
# ============================================================
class RuleBasedClassifier:
    def __init__(self, priority_rules):
        self.rules = priority_rules

    def classify(self, obs):
        for rule in self.rules:
            result = rule(obs)
            if result is not None:
                return result
        return (0, 0.5)


class MLClassifier:
    def __init__(self, A_list):
        self.A_list = A_list

    def classify(self, obs, prior=None):
        n_primary = self.A_list[0].shape[1]
        log_lik = np.zeros(n_primary)
        for m, (A_m, o_m) in enumerate(zip(self.A_list, obs)):
            lik_slice = A_m[o_m]
            while lik_slice.ndim > 1:
                lik_slice = lik_slice.mean(axis=-1)
            log_lik += log_s(lik_slice)
        if prior is not None:
            log_lik += log_s(prior)
        probs = np.exp(log_lik - log_lik.max())
        probs = probs / probs.sum()
        label = int(np.argmax(probs))
        return label, float(probs[label])


# ============================================================
# Calibration
# ============================================================
def confidence_calibration(confidences, correct, n_bins=10):
    confidences = np.asarray(confidences)
    correct = np.asarray(correct, dtype=bool)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if i == n_bins - 1:
            mask = mask | (confidences == bins[i + 1])
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_accuracies[i] = correct[mask].mean()
    return bin_centers, bin_accuracies, bin_counts


# ============================================================
# Figure utilities
# ============================================================
def setup_figure_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 100,
        'savefig.dpi': 200,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def save_figure(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(FIG_DIR, f'{name}.{fmt}'),
            dpi=200, bbox_inches='tight'
        )
    plt.close(fig)
    print(f"  Figure saved: {name}")
