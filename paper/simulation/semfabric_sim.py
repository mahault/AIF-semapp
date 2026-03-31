"""
semfabric_sim.py — Semiotic Fabric Active Inference Simulation
==============================================================

Computational demonstration for:
"The Semiotic Fabric as Cognitive Substrate: Decentralized Active Inference
Through Nested Markov Blankets in Ultra-Large-Scale Data Architecture"

Uses pymdp (inferactively-pymdp v1.0, JAX backend) for discrete active
inference with parameter learning.  Process Waves navigate a 5x5 semiotic
fabric, sense semantic and domain signals, and enrich cells — driven by
expected free energy minimization.

Agent design:
  - State factors : semantic_state(3) × domain(3)
  - Observations  : semantic_obs(4), assoc_obs(3), domain_cue(4)
  - Control       : enrich_action(2)  [OBSERVE / ENRICH]
  - Movement      : spatial EFE evaluation over candidate cells (external)
  - Learning      : learn_A (observation model), learn_B (transitions)
"""

import numpy as np
from itertools import product
import jax.numpy as jnp
import jax.random as jr
from pymdp.agent import Agent

# ============================================================
# Constants
# ============================================================
GRID_N = 5
N_CELLS = GRID_N * GRID_N

UNPROCESSED, PARTIAL, ENRICHED = 0, 1, 2
N_SEM = 3

GEOINT, SIGINT, OSINT = 0, 1, 2
N_DOM = 3
DOMAIN_NAMES = ["GEOINT", "SIGINT", "OSINT"]

N_OBS_SEM = 4       # none / hint / match / contradiction
N_OBS_ASSOC = 3     # none / consistent / inconsistent
N_OBS_DOM = 4       # dom0 / dom1 / dom2 / ambiguous

OBS_SEM_NONE, OBS_SEM_HINT, OBS_SEM_MATCH, OBS_SEM_CONTRA = 0, 1, 2, 3
OBS_ASSOC_NONE, OBS_ASSOC_CONSISTENT, OBS_ASSOC_INCONSISTENT = 0, 1, 2
OBS_DOM_AMB = 3

UP, DOWN, LEFT, RIGHT, STAY = range(5)
OBSERVE, ENRICH_ACT = 0, 1
MOVE_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
N_MOVE = 5


# ============================================================
# Grid helpers
# ============================================================
def idx_to_rc(k):
    return divmod(k, GRID_N)

def rc_to_idx(r, c):
    return r * GRID_N + c

def step_cell(idx, move):
    r, c = idx_to_rc(idx)
    if move == UP:    r = max(r - 1, 0)
    elif move == DOWN:  r = min(r + 1, GRID_N - 1)
    elif move == LEFT:  c = max(c - 1, 0)
    elif move == RIGHT: c = min(c + 1, GRID_N - 1)
    return rc_to_idx(r, c)

def neighbors(idx):
    r, c = idx_to_rc(idx)
    nbs = []
    if r > 0:            nbs.append(rc_to_idx(r - 1, c))
    if r < GRID_N - 1:   nbs.append(rc_to_idx(r + 1, c))
    if c > 0:            nbs.append(rc_to_idx(r, c - 1))
    if c < GRID_N - 1:   nbs.append(rc_to_idx(r, c + 1))
    return nbs


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
    p = p / p.sum()
    return -np.sum(p * log_s(p))

def kl_div(q, p):
    q = np.asarray(q, dtype=np.float64); q = q / q.sum()
    p = np.asarray(p, dtype=np.float64); p = p / p.sum()
    return np.sum(q * (log_s(q) - log_s(p)))


# ============================================================
# Environment
# ============================================================
class SemFabricEnv:
    """5×5 semiotic fabric with semantic states and domain assignments."""

    def __init__(self, heterogeneous=False, true_domain=GEOINT, seed=None):
        self.rng = np.random.RandomState(seed)
        self.sem = np.zeros(N_CELLS, dtype=int)        # semantic state per cell
        self.quality = np.zeros(N_CELLS, dtype=float)   # enrichment quality
        self.cell_domains = np.full(N_CELLS, true_domain, dtype=int)

        if heterogeneous:
            for k in range(N_CELLS):
                r, c = idx_to_rc(k)
                if r + c < GRID_N - 1:
                    self.cell_domains[k] = GEOINT
                elif r + c > GRID_N - 1:
                    self.cell_domains[k] = SIGINT
                else:
                    self.cell_domains[k] = self.rng.choice([GEOINT, SIGINT])

    def observe(self, pos):
        """Generate noisy observations at *pos*."""
        sem_state = self.sem[pos]
        # Semantic obs
        if sem_state == UNPROCESSED:
            probs = [0.80, 0.12, 0.03, 0.05]
        elif sem_state == PARTIAL:
            probs = [0.10, 0.55, 0.25, 0.10]
        else:
            probs = [0.02, 0.08, 0.85, 0.05]
        obs_sem = self.rng.choice(N_OBS_SEM, p=probs)

        # Association obs (based on neighbors)
        nbs = neighbors(pos)
        has_enriched = any(self.sem[n] >= PARTIAL for n in nbs)
        if has_enriched:
            obs_assoc = self.rng.choice(N_OBS_ASSOC, p=[0.10, 0.75, 0.15])
        else:
            obs_assoc = self.rng.choice(N_OBS_ASSOC, p=[0.85, 0.10, 0.05])

        # Domain obs
        true_dom = self.cell_domains[pos]
        r = self.rng.random()
        if r < 0.55:
            obs_dom = true_dom
        elif r < 0.80:
            obs_dom = OBS_DOM_AMB
        else:
            wrong = [d for d in range(N_DOM) if d != true_dom]
            obs_dom = self.rng.choice(wrong)

        return [obs_sem, obs_assoc, obs_dom]

    def step(self, pos, move, enrich, wave_dom_belief=None):
        """Execute movement + enrichment, return (new_pos, obs, did_enrich)."""
        new_pos = step_cell(pos, move)
        did_enrich = False

        if enrich == ENRICH_ACT and self.sem[new_pos] < ENRICHED:
            true_dom = self.cell_domains[new_pos]
            if wave_dom_belief is not None:
                accuracy = float(wave_dom_belief[true_dom])
            else:
                accuracy = 0.85
            p_success = 0.15 + 0.80 * accuracy
            if self.rng.random() < p_success:
                self.sem[new_pos] = min(self.sem[new_pos] + 1, ENRICHED)
                self.quality[new_pos] = max(self.quality[new_pos], accuracy)
                did_enrich = True

        obs = self.observe(new_pos)
        return new_pos, obs, did_enrich

    def n_enriched(self):
        return int(np.sum(self.sem == ENRICHED))

    def n_processed(self):
        return int(np.sum(self.sem >= PARTIAL))

    def grid(self):
        return self.sem.reshape(GRID_N, GRID_N).copy()


# ============================================================
# Generative model builders
# ============================================================
def build_A_sem(expertise_dom=None, kappa=0.50):
    """P(obs_semantic | semantic_state, domain).  Shape (4, 3, 3).
    Depends primarily on semantic_state; domain has negligible effect."""
    base = np.array([
        [0.80, 0.10, 0.02],   # none
        [0.12, 0.55, 0.08],   # hint
        [0.03, 0.25, 0.85],   # match
        [0.05, 0.10, 0.05],   # contradiction
    ])  # shape (4, 3)
    # Tile across domain dimension (domain-independent)
    A = np.stack([base] * N_DOM, axis=-1)  # (4, 3, 3)
    return A

def build_A_assoc():
    """P(obs_assoc | semantic_state, domain).  Shape (3, 3, 3).
    Uses semantic_state as proxy for neighborhood enrichment."""
    base = np.array([
        [0.85, 0.30, 0.10],   # none
        [0.10, 0.50, 0.75],   # consistent
        [0.05, 0.20, 0.15],   # inconsistent
    ])  # (3, 3)
    A = np.stack([base] * N_DOM, axis=-1)  # (3, 3, 3)
    return A

def build_A_dom(expertise_dom=None, kappa=0.50):
    """P(obs_domain | semantic_state, domain).  Shape (4, 3, 3).
    Depends primarily on domain factor."""
    base = np.zeros((N_OBS_DOM, N_DOM))
    for d in range(N_DOM):
        for o in range(N_DOM):
            base[o, d] = kappa if o == d else (1.0 - kappa - 0.25) / (N_DOM - 1)
        base[OBS_DOM_AMB, d] = 0.25
    if expertise_dom is not None:
        base[expertise_dom, expertise_dom] = 0.70
        base[OBS_DOM_AMB, expertise_dom] = 0.10
        # renormalise column
        col = base[:, expertise_dom]
        others = [i for i in range(N_OBS_DOM) if i != expertise_dom and i != OBS_DOM_AMB]
        remainder = 1.0 - base[expertise_dom, expertise_dom] - base[OBS_DOM_AMB, expertise_dom]
        for i in others:
            base[i, expertise_dom] = remainder / len(others)
    # Tile across semantic (semantic-independent)
    A = np.stack([base] * N_SEM, axis=1)  # (4, 3, 3) — obs × sem × dom
    return A

def build_B_sem():
    """P(sem' | sem, action).  Shape (3, 3, 2).
    Action 0 = OBSERVE (identity), Action 1 = ENRICH (upgrade)."""
    B_obs = np.eye(N_SEM)
    B_enr = np.array([
        [0.20, 0.00, 0.00],
        [0.80, 0.20, 0.00],
        [0.00, 0.80, 1.00],
    ])
    return np.stack([B_obs, B_enr], axis=-1)  # (3, 3, 2)

def build_B_dom():
    """P(dom' | dom, action).  Shape (3, 3, 1). Always identity (domain constant)."""
    return np.expand_dims(np.eye(N_DOM), -1)  # (3, 3, 1)

def build_C_sem():
    return np.array([-0.3, 0.5, 0.0, -1.0])

def build_C_assoc():
    return np.array([-0.2, 0.5, -0.3])

def build_C_dom(preference_dom=None):
    C = np.array([0.0, 0.0, 0.0, -0.5])
    if preference_dom is not None:
        C[preference_dom] = 1.0
    return C

def build_D_sem():
    D = np.array([0.7, 0.2, 0.1])
    return D / D.sum()

def build_D_dom(prior_dom=None):
    if prior_dom is not None:
        D = np.full(N_DOM, 0.1)
        D[prior_dom] = 0.8
    else:
        D = np.ones(N_DOM) / N_DOM
    return D / D.sum()


# ============================================================
# Wave (Active Inference Agent wrapping pymdp)
# ============================================================
class Wave:
    """Process Wave agent using pymdp for local inference + learning,
    with spatial EFE-based movement policy."""

    def __init__(self, epistemic=True, gamma=8.0, seed=None,
                 expertise_dom=None, preference_dom=None, prior_dom=None,
                 learn=True, dirichlet_scale=1.0):
        self.rng = np.random.RandomState(seed)
        self.jax_key = jr.PRNGKey(seed if seed is not None else 0)
        self.epistemic = epistemic
        self.gamma = gamma
        self.learn = learn

        # Build generative model arrays (numpy)
        A_sem_np = build_A_sem(expertise_dom)
        A_assoc_np = build_A_assoc()
        A_dom_np = build_A_dom(expertise_dom)
        B_sem_np = build_B_sem()
        B_dom_np = build_B_dom()
        C_sem_np = build_C_sem()
        C_assoc_np = build_C_assoc()
        C_dom_np = build_C_dom(preference_dom)
        D_sem_np = build_D_sem()
        D_dom_np = build_D_dom(prior_dom)

        # Store numpy copies for spatial EFE evaluation
        self.A_sem_np = A_sem_np
        self.A_assoc_np = A_assoc_np
        self.A_dom_np = A_dom_np
        self.C_sem_np = C_sem_np
        self.C_assoc_np = C_assoc_np
        self.C_dom_np = C_dom_np
        self.D_sem_np = D_sem_np
        self.D_dom_np = D_dom_np

        # Convert to JAX
        A = [jnp.array(A_sem_np), jnp.array(A_assoc_np), jnp.array(A_dom_np)]
        B = [jnp.array(B_sem_np), jnp.array(B_dom_np)]
        C = [jnp.array(C_sem_np), jnp.array(C_assoc_np), jnp.array(C_dom_np)]
        D = [jnp.array(D_sem_np), jnp.array(D_dom_np)]

        # Dirichlet priors for learning
        pA = [a * dirichlet_scale for a in A] if learn else None
        pB = [b * dirichlet_scale for b in B] if learn else None

        # Create pymdp Agent
        self.agent = Agent(
            A=A, B=B, C=C, D=D,
            pA=pA, pB=pB,
            num_controls=[2, 1],      # enrich(2) for semantic, none(1) for domain
            control_fac_idx=[0],       # only semantic factor is controllable
            policy_len=1,
            use_utility=True,
            use_states_info_gain=epistemic,
            learn_A=learn,
            learn_B=learn,
            action_selection='stochastic',
            gamma=gamma,
            batch_size=1,
        )

        # Per-cell belief stores
        self.cell_beliefs_sem = np.tile(D_sem_np, (N_CELLS, 1))   # (25, 3)
        self.cell_beliefs_dom = np.tile(D_dom_np, (N_CELLS, 1))   # (25, 3)

        # Visit counts for exploration bonus
        self.visit_counts = np.zeros(N_CELLS, dtype=int)

        # Tracking
        self.qs_history = []
        self.obs_history = []
        self.action_history = []

    def _get_empirical_prior(self, pos):
        """Load per-cell beliefs as empirical prior for pymdp."""
        sem = jnp.expand_dims(jnp.array(self.cell_beliefs_sem[pos]), 0)  # (1, 3)
        dom = jnp.expand_dims(jnp.array(self.cell_beliefs_dom[pos]), 0)  # (1, 3)
        return [sem, dom]

    def _store_posterior(self, pos, qs):
        """Store updated posterior beliefs back to per-cell store."""
        self.cell_beliefs_sem[pos] = np.array(qs[0][0, 0, :])  # (3,)
        self.cell_beliefs_dom[pos] = np.array(qs[1][0, 0, :])  # (3,)

    def _obs_to_jax(self, obs):
        """Convert integer observations to JAX format for pymdp."""
        return [jnp.array([[o]]) for o in obs]  # list of (1, 1) arrays

    def infer_and_act(self, pos, obs):
        """Full perception–action cycle at current position.
        Returns: (enrich_action, qs, q_pi, efe)"""
        # Perception: state inference using per-cell beliefs as prior
        emp_prior = self._get_empirical_prior(pos)
        jax_obs = self._obs_to_jax(obs)
        qs = self.agent.infer_states(jax_obs, emp_prior)

        # Store updated beliefs
        self._store_posterior(pos, qs)

        # Policy inference (observe vs enrich)
        q_pi, efe = self.agent.infer_policies(qs)

        # Sample action
        self.jax_key, subkey = jr.split(self.jax_key)
        action = self.agent.sample_action(q_pi, rng_key=jr.split(subkey, 1))
        enrich_action = int(action[0, 0])  # first factor = enrichment

        # Track for learning
        self.qs_history.append(qs)
        self.obs_history.append(jax_obs)
        self.action_history.append(action)

        return enrich_action, qs, q_pi, efe

    def select_movement(self, current_pos):
        """Spatial EFE: evaluate candidate positions via softmax selection.
        Returns: (move_direction, target_pos)"""
        self.visit_counts[current_pos] += 1

        candidates = [(STAY, current_pos)]
        for move in [UP, DOWN, LEFT, RIGHT]:
            target = step_cell(current_pos, move)
            if target != current_pos:
                candidates.append((move, target))

        values = np.array([self._cell_efe(t) for _, t in candidates])
        probs = softmax(values, temp=4.0)
        idx = self.rng.choice(len(candidates), p=probs)
        return candidates[idx]

    def _cell_efe(self, cell_idx):
        """Compute negative EFE (value) for visiting a cell.
        Higher = more valuable to visit."""
        q_sem = self.cell_beliefs_sem[cell_idx]
        q_dom = self.cell_beliefs_dom[cell_idx]

        pragmatic = 0.0
        epistemic_val = 0.0

        # Semantic modality
        A_sem_marginal = np.einsum('ijk,k->ij', self.A_sem_np, q_dom)  # (4, 3)
        qo_sem = A_sem_marginal @ q_sem
        qo_sem = qo_sem / (qo_sem.sum() + 1e-16)
        pragmatic += np.dot(qo_sem, self.C_sem_np)
        if self.epistemic:
            H_qo = entropy_H(qo_sem)
            H_cols = np.array([entropy_H(A_sem_marginal[:, s]) for s in range(N_SEM)])
            epistemic_val += H_qo - np.dot(q_sem, H_cols)

        # Domain modality
        A_dom_marginal = np.einsum('ijk,j->ik', self.A_dom_np, q_sem)
        qo_dom = A_dom_marginal @ q_dom
        qo_dom = qo_dom / (qo_dom.sum() + 1e-16)
        pragmatic += np.dot(qo_dom, self.C_dom_np)
        if self.epistemic:
            H_qo_d = entropy_H(qo_dom)
            H_cols_d = np.array([entropy_H(A_dom_marginal[:, d]) for d in range(N_DOM)])
            epistemic_val += H_qo_d - np.dot(q_dom, H_cols_d)

        # Association modality
        A_assoc_marginal = np.einsum('ijk,k->ij', self.A_assoc_np, q_dom)
        qo_assoc = A_assoc_marginal @ q_sem
        qo_assoc = qo_assoc / (qo_assoc.sum() + 1e-16)
        pragmatic += np.dot(qo_assoc, self.C_assoc_np)
        if self.epistemic:
            H_qo_a = entropy_H(qo_assoc)
            H_cols_a = np.array([entropy_H(A_assoc_marginal[:, s]) for s in range(N_SEM)])
            epistemic_val += H_qo_a - np.dot(q_sem, H_cols_a)

        # Novelty bonus: prefer less-visited cells (1/sqrt(n+1))
        novelty = 1.0 / np.sqrt(self.visit_counts[cell_idx] + 1)

        # Unprocessed cells are more valuable to explore
        unprocessed_bonus = q_sem[UNPROCESSED] * 0.3

        value = pragmatic + epistemic_val + novelty + unprocessed_bonus
        return value

    def update_neighbor_beliefs(self, pos, obs_assoc):
        """Gentle belief propagation from observation to neighbors."""
        nbs = neighbors(pos)
        if obs_assoc == OBS_ASSOC_CONSISTENT:
            for n in nbs:
                self.cell_beliefs_sem[n, PARTIAL:] *= 1.2
                self.cell_beliefs_sem[n] /= self.cell_beliefs_sem[n].sum()
        elif obs_assoc == OBS_ASSOC_INCONSISTENT:
            for n in nbs:
                self.cell_beliefs_sem[n, UNPROCESSED] *= 1.15
                self.cell_beliefs_sem[n] /= self.cell_beliefs_sem[n].sum()

    def propagate_domain_belief(self, pos, obs_dom):
        """Gentle domain belief propagation to neighbors."""
        if obs_dom < N_DOM:  # actual cue, not ambiguous
            nbs = neighbors(pos)
            likelihood = np.zeros(N_DOM)
            likelihood[obs_dom] = 0.55
            others = (1.0 - 0.55) / (N_DOM - 1)
            for d in range(N_DOM):
                if d != obs_dom:
                    likelihood[d] = others
            for n in nbs:
                tempered = 0.1 * likelihood + 0.9 * np.ones(N_DOM) / N_DOM
                posterior = tempered * self.cell_beliefs_dom[n]
                self.cell_beliefs_dom[n] = posterior / posterior.sum()

    def record_enrichment(self, pos, success):
        """Update beliefs after an enrichment attempt."""
        if success:
            boost = np.array([0.1, 0.5, 2.0])
            self.cell_beliefs_sem[pos] *= boost
            self.cell_beliefs_sem[pos] /= self.cell_beliefs_sem[pos].sum()

    def do_learning_update(self):
        """Run parameter learning on accumulated experience."""
        if not self.learn or len(self.qs_history) < 2:
            return

        T = len(self.qs_history)
        n_factors = 2

        # Build beliefs sequence: list of (1, T, Ns_f)
        beliefs = []
        for f in range(n_factors):
            seq = jnp.concatenate(
                [self.qs_history[t][f][:, 0, :] for t in range(T)],
                axis=0
            ).reshape(1, T, -1)
            beliefs.append(seq)

        # Build outcomes: list of (1, T, 1)
        n_modalities = 3
        outcomes = []
        for m in range(n_modalities):
            seq = jnp.concatenate(
                [self.obs_history[t][m] for t in range(T)],
                axis=1
            )  # (1, T)
            outcomes.append(seq)

        # Build actions: (1, T, n_action_dims)
        actions = jnp.concatenate(
            [self.action_history[t] for t in range(T)],
            axis=0
        ).reshape(1, T, -1)

        # Update
        self.agent = self.agent.infer_parameters(beliefs, outcomes, actions)

    def reset_history(self):
        """Clear history after learning update."""
        self.qs_history = []
        self.obs_history = []
        self.action_history = []


# ============================================================
# Free energy metrics (objective, fabric-level)
# ============================================================
def sign_free_energy(sem_state, quality=0.0):
    """VFE of a single Sign (cell) — based on objective enrichment state."""
    F_max = np.log(N_SEM)
    if sem_state == UNPROCESSED:
        return F_max
    effective = (sem_state / 2.0) * max(quality, 0.01)
    return (1.0 - effective) * F_max

def regional_free_energy(env, r0, c0, size=2):
    """Mean VFE over a size×size region."""
    vals = []
    for dr in range(size):
        for dc in range(size):
            r, c = r0 + dr, c0 + dc
            if r < GRID_N and c < GRID_N:
                idx = rc_to_idx(r, c)
                vals.append(sign_free_energy(env.sem[idx], env.quality[idx]))
    return np.mean(vals) if vals else 0.0

def fabric_free_energy(env):
    """Fabric-level VFE = mean sign-level VFE across all cells."""
    return np.mean([
        sign_free_energy(env.sem[k], env.quality[k]) for k in range(N_CELLS)
    ])
