"""
semfabric_sim.py — Semiotic Fabric Active Inference Simulation
==============================================================

Computational demonstration for:
"The Semiotic Fabric as Cognitive Substrate: Decentralized Active Inference
Through Nested Markov Blankets in Ultra-Large-Scale Data Architecture"

Self-contained discrete active inference using numpy, following
Parr, Pezzulo & Friston (2022).

Process Waves navigate a 5x5 semiotic fabric, sense semantic and domain
signals, and enrich cells — driven by expected free energy minimization
with proper 2-step policy evaluation.
"""

import numpy as np
from itertools import product

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

# Observation sizes
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
N_ENRICH = 2
COMBINED = list(product(range(N_MOVE), range(N_ENRICH)))
N_ACTIONS = len(COMBINED)


def log_s(x):
    return np.log(np.maximum(x, 1e-16))

def softmax(x, temp=1.0):
    e = np.exp(temp * (x - np.max(x)))
    return e / e.sum()

def entropy_H(p):
    p = np.asarray(p, dtype=float)
    return -np.sum(p * log_s(p))

def kl_div(q, p):
    return np.sum(q * (log_s(q) - log_s(p)))


# ============================================================
# Grid helpers
# ============================================================
def idx_to_rc(k):
    return divmod(k, GRID_N)

def rc_to_idx(r, c):
    return r * GRID_N + c

def step_cell(idx, move):
    r, c = idx_to_rc(idx)
    if move == UP:    r = max(0, r - 1)
    elif move == DOWN:  r = min(GRID_N - 1, r + 1)
    elif move == LEFT:  c = max(0, c - 1)
    elif move == RIGHT: c = min(GRID_N - 1, c + 1)
    return rc_to_idx(r, c)

def get_neighbors(idx):
    r, c = idx_to_rc(idx)
    nb = {}
    if r > 0:         nb['N'] = rc_to_idx(r-1, c)
    if r < GRID_N-1:  nb['S'] = rc_to_idx(r+1, c)
    if c < GRID_N-1:  nb['E'] = rc_to_idx(r, c+1)
    if c > 0:         nb['W'] = rc_to_idx(r, c-1)
    return nb


# ============================================================
# Semiotic Fabric Environment (Generative Process)
# ============================================================
class SemFabricEnv:
    """5x5 semiotic fabric with heterogeneous domain context."""

    def __init__(self, heterogeneous=False, true_domain=GEOINT, seed=None):
        self.rng = np.random.default_rng(seed)
        self.sem = np.zeros(N_CELLS, dtype=int)
        self.quality = np.zeros(N_CELLS)
        self.heterogeneous = heterogeneous

        if heterogeneous:
            self.cell_domains = np.zeros(N_CELLS, dtype=int)
            for i in range(N_CELLS):
                r, c = idx_to_rc(i)
                if r + c < GRID_N - 1:
                    self.cell_domains[i] = GEOINT
                elif r + c > GRID_N - 1:
                    self.cell_domains[i] = SIGINT
                else:
                    self.cell_domains[i] = self.rng.choice([GEOINT, SIGINT])
        else:
            self.cell_domains = np.full(N_CELLS, true_domain, dtype=int)

    def observe(self, pos):
        obs = [0, 0, 0, 0]
        obs[0] = pos

        s = self.sem[pos]
        if s == UNPROCESSED:
            obs[1] = OBS_SEM_NONE
        elif s == PARTIAL:
            obs[1] = self.rng.choice([OBS_SEM_HINT, OBS_SEM_MATCH], p=[0.7, 0.3])
        else:
            obs[1] = OBS_SEM_MATCH

        nb = get_neighbors(pos)
        has_enriched_nb = any(self.sem[n] >= PARTIAL for n in nb.values())
        if has_enriched_nb:
            obs[2] = OBS_ASSOC_CONSISTENT if self.rng.random() < 0.8 else OBS_ASSOC_INCONSISTENT
        else:
            obs[2] = OBS_ASSOC_NONE

        local_dom = self.cell_domains[pos]
        p_correct = 0.55
        r = self.rng.random()
        if r < p_correct:
            obs[3] = local_dom
        elif r < p_correct + 0.25:
            obs[3] = OBS_DOM_AMB
        else:
            obs[3] = (local_dom + self.rng.integers(1, N_DOM)) % N_DOM
        return obs

    def step(self, pos, move, enrich, wave_dom_belief=None):
        """
        Enrichment success depends on Wave's domain belief accuracy.
        p_success = 0.15 + 0.80 * q(correct_domain)
        """
        new_pos = step_cell(pos, move)
        did_enrich = False
        if enrich == ENRICH_ACT and self.sem[new_pos] < ENRICHED:
            cell_dom = self.cell_domains[new_pos]
            if wave_dom_belief is not None:
                accuracy = float(wave_dom_belief[cell_dom])
                p_success = 0.15 + 0.80 * accuracy
            else:
                accuracy = 0.5
                p_success = 0.85
            if self.rng.random() < p_success:
                self.sem[new_pos] = min(self.sem[new_pos] + 1, ENRICHED)
                self.quality[new_pos] = max(self.quality[new_pos], accuracy)
                did_enrich = True
        return new_pos, self.observe(new_pos), did_enrich

    def grid(self):
        return self.sem.copy().reshape(GRID_N, GRID_N)

    def n_enriched(self):
        return int(np.sum(self.sem >= ENRICHED))

    def n_processed(self):
        return int(np.sum(self.sem >= PARTIAL))


# ============================================================
# Active Inference Wave
# ============================================================
class Wave:
    """
    Process Wave as discrete active inference agent with 2-step
    policy evaluation (proper expected free energy).

    Configurable generative model components:
    - A_dom: expertise_dom increases observation precision for one domain
    - C_dom: preference_dom creates pragmatic drive toward one domain
    - D (domain_prior): initial domain belief
    """

    def __init__(self, domain_prior=None, epistemic=True, gamma=8.0, seed=None,
                 expertise_dom=None, preference_dom=None):
        self.rng = np.random.default_rng(seed)
        self.epistemic = epistemic
        self.gamma = gamma

        # Per-cell semantic belief
        self.sem_belief = np.ones((N_CELLS, N_SEM)) / N_SEM
        self.sem_belief[:, UNPROCESSED] = 0.7
        self.sem_belief[:, PARTIAL] = 0.2
        self.sem_belief[:, ENRICHED] = 0.1

        # Per-cell domain belief (D) — enables position-dependent predictions
        if domain_prior is not None:
            dp = np.array(domain_prior, dtype=float)
            dp /= dp.sum()
        else:
            dp = np.ones(N_DOM) / N_DOM
        self.cell_dom_belief = np.tile(dp, (N_CELLS, 1))

        # --- Likelihood matrices (A) ---

        # P(obs_sem | sem_state), shape (N_SEM, N_OBS_SEM)
        self.A_sem = np.array([
            [0.80, 0.12, 0.03, 0.05],   # UNPROCESSED
            [0.10, 0.55, 0.25, 0.10],   # PARTIAL
            [0.02, 0.08, 0.85, 0.05],   # ENRICHED
        ])

        # P(obs_assoc | neighbor_state), shape (2, N_OBS_ASSOC)
        self.A_assoc = np.array([
            [0.85, 0.10, 0.05],  # no enriched neighbor
            [0.10, 0.75, 0.15],  # has enriched neighbor
        ])

        # P(obs_dom | domain), shape (N_OBS_DOM, N_DOM)
        self.A_dom = np.array([
            [0.50, 0.10, 0.10],  # obs=dom0
            [0.10, 0.50, 0.10],  # obs=dom1
            [0.10, 0.10, 0.50],  # obs=dom2
            [0.30, 0.30, 0.30],  # obs=ambiguous
        ])

        # Customize A_dom for domain expertise
        if expertise_dom is not None:
            self.A_dom[expertise_dom, expertise_dom] = 0.70
            self.A_dom[OBS_DOM_AMB, expertise_dom] = 0.10
            # Renormalize affected column
            self.A_dom[:, expertise_dom] /= self.A_dom[:, expertise_dom].sum()

        # --- Transition models (B) ---
        self.B_sem_observe = np.eye(N_SEM)
        self.B_sem_enrich = np.array([
            # B[s', s] = P(s' | s, enrich)
            # Rows indexed by (from_state): UNPROC, PARTIAL, ENRICHED
            # Values: [P(stay), P(upgrade), P(skip_to_enriched)]
            [0.20, 0.80, 0.00],  # from UNPROCESSED
            [0.00, 0.20, 0.80],  # from PARTIAL
            [0.00, 0.00, 1.00],  # from ENRICHED
        ])

        # --- Preferences (C, log-scale) ---
        # Key design: match=0 so enriched cells don't trap the agent.
        # hint>0 rewards discovering partially-enriched cells.
        # none is mildly negative, contradiction is penalized.
        self.C_sem = np.array([-0.3, 0.5, 0.0, -1.0])
        self.C_assoc = np.array([-0.2, 0.5, -0.3])

        # Customize C_dom for domain preferences
        if preference_dom is not None:
            self.C_dom = np.zeros(N_OBS_DOM)
            self.C_dom[preference_dom] = 1.0
            self.C_dom[OBS_DOM_AMB] = -0.5
        else:
            self.C_dom = np.array([0.0, 0.0, 0.0, -0.5])  # neutral

        # Precompute constant entropies for EFE (performance)
        self._H_A_sem = np.array([entropy_H(self.A_sem[s, :]) for s in range(N_SEM)])
        self._H_A_dom = np.array([entropy_H(self.A_dom[:, d]) for d in range(N_DOM)])
        self._H_A_assoc = np.array([entropy_H(self.A_assoc[i]) for i in range(2)])

    def update_beliefs(self, pos, obs):
        """Bayesian belief update given observation at position pos."""
        # Semantic belief for current cell
        obs_sem = obs[1]
        likelihood_sem = self.A_sem[:, obs_sem]
        posterior_sem = likelihood_sem * self.sem_belief[pos]
        s = posterior_sem.sum()
        self.sem_belief[pos] = posterior_sem / s if s > 1e-16 else np.ones(N_SEM) / N_SEM

        # Neighbor awareness from association signal
        obs_assoc = obs[2]
        nb = get_neighbors(pos)
        if obs_assoc == OBS_ASSOC_CONSISTENT:
            for nidx in nb.values():
                self.sem_belief[nidx, PARTIAL:] *= 1.3
                self.sem_belief[nidx] /= self.sem_belief[nidx].sum()
        elif obs_assoc == OBS_ASSOC_INCONSISTENT:
            for nidx in nb.values():
                self.sem_belief[nidx, UNPROCESSED] *= 1.2
                self.sem_belief[nidx] /= self.sem_belief[nidx].sum()

        # Per-cell domain belief update (moderate tempering for local learning)
        obs_dom = obs[3]
        likelihood_dom = self.A_dom[obs_dom, :]
        tempered_likelihood = 0.3 * likelihood_dom + 0.7 * np.ones(N_DOM) / N_DOM
        posterior_dom = tempered_likelihood * self.cell_dom_belief[pos]
        s = posterior_dom.sum()
        self.cell_dom_belief[pos] = posterior_dom / s if s > 1e-16 else np.ones(N_DOM) / N_DOM

        # Gently propagate domain knowledge to neighbors (spatial smoothing)
        if obs_dom < N_DOM:  # actual domain cue (not ambiguous)
            for nidx in nb.values():
                nb_tempered = 0.1 * likelihood_dom + 0.9 * np.ones(N_DOM) / N_DOM
                nb_post = nb_tempered * self.cell_dom_belief[nidx]
                s = nb_post.sum()
                if s > 1e-16:
                    self.cell_dom_belief[nidx] = nb_post / s

    def _compute_neighbor_enrichment_prob(self, cell_idx):
        """P(at least one neighbor is enriched/partial)."""
        nb = get_neighbors(cell_idx)
        if not nb:
            return 0.0
        p_all_unproc = 1.0
        for nidx in nb.values():
            p_all_unproc *= self.sem_belief[nidx, UNPROCESSED]
        return 1.0 - p_all_unproc

    def _step_value(self, pos, pred_sem):
        """
        Compute value (negative EFE) for one step at position with
        predicted semantic state.

        G = -pragmatic - epistemic  (minimize G)
        value = pragmatic + epistemic  (maximize value)
        """
        # --- Expected observations per modality ---
        qo_sem = self.A_sem.T @ pred_sem
        qo_sem = np.maximum(qo_sem, 1e-16)
        qo_sem /= qo_sem.sum()

        p_enr_nb = self._compute_neighbor_enrichment_prob(pos)
        qo_assoc = (1 - p_enr_nb) * self.A_assoc[0] + p_enr_nb * self.A_assoc[1]
        qo_assoc /= qo_assoc.sum()

        dom_belief = self.cell_dom_belief[pos]
        qo_dom = self.A_dom @ dom_belief
        qo_dom = np.maximum(qo_dom, 1e-16)
        qo_dom /= qo_dom.sum()

        # --- Pragmatic value: E_q(o)[C(o)] ---
        pragmatic = (np.dot(qo_sem, self.C_sem) +
                     np.dot(qo_assoc, self.C_assoc) +
                     np.dot(qo_dom, self.C_dom))

        # --- Epistemic value: mutual information I(o;s) ---
        epistemic = 0.0
        if self.epistemic:
            # Semantic IG (using precomputed A_sem row entropies)
            epistemic += entropy_H(qo_sem) - np.dot(pred_sem, self._H_A_sem)

            # Domain IG (using precomputed A_dom column entropies)
            epistemic += entropy_H(qo_dom) - np.dot(dom_belief, self._H_A_dom)

            # Association IG (using precomputed A_assoc entropies)
            H_qo = entropy_H(qo_assoc)
            E_H = (1 - p_enr_nb) * self._H_A_assoc[0] + p_enr_nb * self._H_A_assoc[1]
            epistemic += H_qo - E_H

        return pragmatic + epistemic

    def evaluate_policies(self, pos):
        """
        Evaluate all 2-step policies pi = (a1, a2).

        G(pi) = G1 + G2 (expected free energy summed over both steps)
        Returns value = -G for each policy (higher = better).
        """
        # Cache step 1 results for each first action (10 unique)
        step1 = {}
        for a1 in range(N_ACTIONS):
            move1, enrich1 = COMBINED[a1]
            pos1 = step_cell(pos, move1)
            if enrich1 == ENRICH_ACT:
                pred_sem1 = self.B_sem_enrich.T @ self.sem_belief[pos1]
            else:
                pred_sem1 = self.sem_belief[pos1].copy()
            v1 = self._step_value(pos1, pred_sem1)
            step1[a1] = (pos1, pred_sem1, v1)

        # Evaluate all 100 two-step policies
        policy_values = np.zeros(N_ACTIONS * N_ACTIONS)
        for a1 in range(N_ACTIONS):
            pos1, pred_sem1, v1 = step1[a1]
            for a2 in range(N_ACTIONS):
                move2, enrich2 = COMBINED[a2]
                pos2 = step_cell(pos1, move2)

                # Semantic belief at pos2 accounting for step 1 effects
                if pos2 == pos1:
                    pre_sem2 = pred_sem1
                else:
                    pre_sem2 = self.sem_belief[pos2]

                if enrich2 == ENRICH_ACT:
                    pred_sem2 = self.B_sem_enrich.T @ pre_sem2
                else:
                    pred_sem2 = pre_sem2

                v2 = self._step_value(pos2, pred_sem2)
                policy_values[a1 * N_ACTIONS + a2] = v1 + v2

        return policy_values

    def select_action(self, pos):
        """Select action by marginalizing over second action of 2-step policies."""
        policy_values = self.evaluate_policies(pos)
        q_pi = softmax(policy_values, self.gamma)

        # Marginalize: P(a1) = sum_{a2} P(pi = (a1, a2))
        action_probs = np.zeros(N_ACTIONS)
        for a1 in range(N_ACTIONS):
            action_probs[a1] = q_pi[a1 * N_ACTIONS:(a1 + 1) * N_ACTIONS].sum()

        action_idx = self.rng.choice(N_ACTIONS, p=action_probs)
        move, enrich = COMBINED[action_idx]
        return move, enrich, policy_values, action_probs

    def record_enrichment(self, pos, success):
        """Update beliefs after enrichment attempt."""
        if success:
            cur = self.sem_belief[pos].copy()
            cur[UNPROCESSED] *= 0.1
            cur[PARTIAL] *= 0.5
            cur[ENRICHED] *= 2.0
            self.sem_belief[pos] = cur / cur.sum()


# ============================================================
# Wave Factory
# ============================================================
def make_wave(expertise_dom=None, preference_dom=None, prior_dom=None,
              epistemic=True, gamma=8.0, seed=None):
    """
    Create a Wave with specific generative model configuration.

    Parameters
    ----------
    expertise_dom : int or None
        Domain for which A_dom has higher observation precision.
    preference_dom : int or None
        Domain whose observations are preferred (C_dom).
    prior_dom : int or None
        Domain with strongest prior belief (D_dom).
    """
    if prior_dom is not None:
        dp = np.ones(N_DOM) * 0.1
        dp[prior_dom] = 0.8
        dp /= dp.sum()
    else:
        dp = None
    return Wave(domain_prior=dp, expertise_dom=expertise_dom,
                preference_dom=preference_dom, epistemic=epistemic,
                gamma=gamma, seed=seed)


# ============================================================
# Free Energy Computation (objective — no agent beliefs)
# ============================================================
def sign_free_energy(sem_state, quality=0.0):
    """
    F_sign based on objective fabric state.

    sem_state: 0=UNPROCESSED, 1=PARTIAL, 2=ENRICHED
    quality: accuracy of domain belief at enrichment time, in [0, 1]
    """
    F_max = np.log(N_DOM)
    if sem_state == UNPROCESSED:
        return F_max
    effective = (sem_state / 2.0) * quality
    return (1.0 - effective) * F_max


def regional_free_energy(env, r0, c0, size=2):
    """Mean F_sign over a size x size region starting at (r0, c0)."""
    g = env.grid()
    F, n = 0.0, 0
    for dr in range(size):
        for dc in range(size):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < GRID_N and 0 <= c < GRID_N:
                idx = rc_to_idx(r, c)
                F += sign_free_energy(g[r, c], env.quality[idx])
                n += 1
    return F / max(n, 1)


def fabric_free_energy(env):
    """F_material = (1/N) sum_i F_sign(i)"""
    g = env.grid()
    return np.mean([sign_free_energy(g[idx_to_rc(i)], env.quality[i])
                     for i in range(N_CELLS)])


# ============================================================
# Experiment 1: Single Wave Epistemic Foraging
# ============================================================
def run_experiment_1(T=100, seed=42):
    """Compare epistemic vs pragmatic-only Wave on homogeneous grid."""
    results = {}
    for label, epist in [('epistemic', True), ('pragmatic', False)]:
        print(f"  [{label}]", end="", flush=True)
        env = SemFabricEnv(true_domain=GEOINT, seed=seed)
        wave = Wave(epistemic=epist, gamma=8.0, seed=seed+1)

        pos = rc_to_idx(GRID_N // 2, GRID_N // 2)
        obs = env.observe(pos)

        hist = {
            'positions': [pos], 'fabric_F': [], 'domain_entropy': [],
            'enrichment_count': [0], 'fabric_states': [env.grid()],
        }

        for t in range(T):
            wave.update_beliefs(pos, obs)
            move, enrich, _, _ = wave.select_action(pos)
            target = step_cell(pos, move)
            pos, obs, did_enrich = env.step(pos, move, enrich, wave.cell_dom_belief[target])
            wave.record_enrichment(pos, did_enrich)

            F = fabric_free_energy(env)
            hist['positions'].append(pos)
            hist['fabric_F'].append(F)
            # Mean domain entropy across cells (how uncertain the Wave is about domains)
            hist['domain_entropy'].append(
                np.mean([entropy_H(wave.cell_dom_belief[i]) for i in range(N_CELLS)]))
            hist['enrichment_count'].append(env.n_enriched())
            hist['fabric_states'].append(env.grid())

            if (t+1) % 25 == 0:
                print(f" t={t+1}:F={F:.2f},E={env.n_enriched()}", end="", flush=True)

        print()
        results[label] = hist
    return results


# ============================================================
# Experiment 2: Multi-Wave Cooperation — Isolating A, C, D
# ============================================================
def run_experiment_2(T=100, seed=42):
    """
    Five conditions isolating each generative model component's contribution.

    All use heterogeneous domain grid (GEOINT upper-left, SIGINT lower-right).
    Two Waves start at opposite corners.

    Conditions:
    1. shared:      identical (standard A, neutral C, uniform D)
    2. diff_A:      only A matrices differ (domain expertise)
    3. diff_C:      only C vectors differ (domain preferences)
    4. diff_D:      only D priors differ (domain prior beliefs)
    5. adversarial: Wave B has fully misspecified model (OSINT A + C + D)
    """
    conditions = {
        'shared': {
            'a': {'expertise_dom': None, 'preference_dom': None, 'prior_dom': None},
            'b': {'expertise_dom': None, 'preference_dom': None, 'prior_dom': None},
        },
        'diff_A': {
            'a': {'expertise_dom': GEOINT, 'preference_dom': None, 'prior_dom': None},
            'b': {'expertise_dom': SIGINT, 'preference_dom': None, 'prior_dom': None},
        },
        'diff_C': {
            'a': {'expertise_dom': None, 'preference_dom': GEOINT, 'prior_dom': None},
            'b': {'expertise_dom': None, 'preference_dom': SIGINT, 'prior_dom': None},
        },
        'diff_D': {
            'a': {'expertise_dom': None, 'preference_dom': None, 'prior_dom': GEOINT},
            'b': {'expertise_dom': None, 'preference_dom': None, 'prior_dom': SIGINT},
        },
        'adversarial': {
            'a': {'expertise_dom': None, 'preference_dom': None, 'prior_dom': None},
            'b': {'expertise_dom': OSINT, 'preference_dom': OSINT, 'prior_dom': OSINT},
        },
    }

    results = {}
    for cond_name, wave_configs in conditions.items():
        print(f"  [{cond_name}]", end="", flush=True)
        env = SemFabricEnv(heterogeneous=True, seed=seed)

        wa = make_wave(**wave_configs['a'], epistemic=True, gamma=8.0, seed=seed+10)
        wb = make_wave(**wave_configs['b'], epistemic=True, gamma=8.0, seed=seed+20)

        # Opposite starting corners
        pos_a = rc_to_idx(0, 0)                    # GEOINT territory
        pos_b = rc_to_idx(GRID_N-1, GRID_N-1)      # SIGINT territory
        obs_a, obs_b = env.observe(pos_a), env.observe(pos_b)

        hist = {
            'fabric_F': [], 'enrichment_count': [0],
            'fabric_states': [env.grid()],
            'positions_a': [pos_a], 'positions_b': [pos_b],
            'coverage': [],
        }

        for t in range(T):
            # Wave A
            wa.update_beliefs(pos_a, obs_a)
            mv_a, en_a, _, _ = wa.select_action(pos_a)
            tgt_a = step_cell(pos_a, mv_a)
            pos_a, obs_a, did_a = env.step(pos_a, mv_a, en_a, wa.cell_dom_belief[tgt_a])
            wa.record_enrichment(pos_a, did_a)

            # Wave B (sees changes from A through shared environment)
            wb.update_beliefs(pos_b, obs_b)
            mv_b, en_b, _, _ = wb.select_action(pos_b)
            tgt_b = step_cell(pos_b, mv_b)
            pos_b, obs_b, did_b = env.step(pos_b, mv_b, en_b, wb.cell_dom_belief[tgt_b])
            wb.record_enrichment(pos_b, did_b)

            # Cross-Wave belief propagation through fabric
            if did_a:
                for nidx in get_neighbors(pos_a).values():
                    wb.sem_belief[nidx, PARTIAL:] *= 1.1
                    wb.sem_belief[nidx] /= wb.sem_belief[nidx].sum()
            if did_b:
                for nidx in get_neighbors(pos_b).values():
                    wa.sem_belief[nidx, PARTIAL:] *= 1.1
                    wa.sem_belief[nidx] /= wa.sem_belief[nidx].sum()

            F = fabric_free_energy(env)
            hist['fabric_F'].append(F)
            hist['enrichment_count'].append(env.n_enriched())
            hist['fabric_states'].append(env.grid())
            hist['positions_a'].append(pos_a)
            hist['positions_b'].append(pos_b)
            hist['coverage'].append(env.n_processed() / N_CELLS)

            if (t+1) % 25 == 0:
                print(f" t={t+1}:F={F:.2f},E={env.n_enriched()}", end="", flush=True)

        print()
        hist['env'] = env
        results[cond_name] = hist
    return results


# ============================================================
# Experiment 3: Nested Free Energy
# ============================================================
def run_experiment_3(T=100, seed=42):
    """Track F at Sign, regional, and fabric level."""
    print("  [nested]", end="", flush=True)
    env = SemFabricEnv(true_domain=GEOINT, seed=seed)
    wave = Wave(domain_prior=[0.5, 0.25, 0.25], epistemic=True, gamma=8.0, seed=seed+1)

    pos = rc_to_idx(GRID_N // 2, GRID_N // 2)
    obs = env.observe(pos)

    tracked = [rc_to_idx(0,0), rc_to_idx(1,2), rc_to_idx(2,4),
               rc_to_idx(3,1), rc_to_idx(4,3)]
    regions = [(0,0), (0,2), (2,0), (2,2)]

    hist = {
        'sign_F': {s: [] for s in tracked},
        'regional_F': {r: [] for r in regions},
        'fabric_F': [], 'enrichment_count': [0],
    }

    for t in range(T):
        wave.update_beliefs(pos, obs)
        mv, en, _, _ = wave.select_action(pos)
        tgt = step_cell(pos, mv)
        pos, obs, did = env.step(pos, mv, en, wave.cell_dom_belief[tgt])
        wave.record_enrichment(pos, did)

        g = env.grid()
        for s in tracked:
            r, c = idx_to_rc(s)
            hist['sign_F'][s].append(sign_free_energy(g[r,c], env.quality[s]))
        for (r0, c0) in regions:
            hist['regional_F'][(r0,c0)].append(regional_free_energy(env, r0, c0))
        F = fabric_free_energy(env)
        hist['fabric_F'].append(F)
        hist['enrichment_count'].append(env.n_enriched())

        if (t+1) % 25 == 0:
            print(f" t={t+1}:F={F:.2f},E={env.n_enriched()}", end="", flush=True)

    print()
    hist['tracked'] = tracked
    hist['regions'] = regions
    return hist


if __name__ == '__main__':
    print("=== Semiotic Fabric Active Inference Simulation ===\n")

    print("Experiment 1: Single Wave Epistemic Foraging")
    r1 = run_experiment_1(T=100, seed=42)
    print(f"  Epistemic: F={r1['epistemic']['fabric_F'][-1]:.3f}, "
          f"enriched={r1['epistemic']['enrichment_count'][-1]}")
    print(f"  Pragmatic: F={r1['pragmatic']['fabric_F'][-1]:.3f}, "
          f"enriched={r1['pragmatic']['enrichment_count'][-1]}\n")

    print("Experiment 2: Multi-Wave Cooperation")
    r2 = run_experiment_2(T=100, seed=42)
    for c in ['shared', 'diff_A', 'diff_C', 'diff_D', 'adversarial']:
        print(f"  {c}: F={r2[c]['fabric_F'][-1]:.3f}, "
              f"enriched={r2[c]['enrichment_count'][-1]}")
    print()

    print("Experiment 3: Nested Free Energy")
    r3 = run_experiment_3(T=100, seed=42)
    print(f"  Final fabric F: {r3['fabric_F'][-1]:.3f}")
    print("\nDone.")
