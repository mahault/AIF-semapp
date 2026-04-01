# The Semiotic Fabric as Cognitive Substrate

**Decentralized Active Inference Through Nested Markov Blankets in Ultra-Large-Scale Data Architecture**

Mahault Albarracin, S. Yoakum-Stover, M. A. Eick

## Overview

This repository contains the paper, simulation code, and figures for a study arguing that the SemApp semiotic data fabric — an operational, petabyte-scale data architecture deployed within the U.S. Intelligence Community — constitutes a cognitive substrate in the formal sense defined by the Free Energy Principle.

The paper establishes:
- A structural isomorphism between Peirce's triadic semiotics and variational inference
- That SemApp's three-layer architecture (unstructured data, structured Signs, domain models) induces nested Markov blankets
- That Process Waves can be formalized as active inference agents
- A fabric-level variational free energy metric whose decrease under consistent processing constitutes learning

## Repository Structure

```
paper/
  semapp-cognitive-substrate.tex     Main paper (LaTeX)
  figures/                           Generated figures (PDF + PNG)
  simulation/
    common.py                        Shared infrastructure (InferenceWave, baselines, utilities)
    exp1_asd.py                      Experiment 1: Artifact Standard Detection
    exp2_disambiguation.py           Experiment 2: Ambiguity Resolution
    exp3_entity_resolution.py        Experiment 3: Multi-Wave Entity Resolution
    exp4_nested_vfe.py               Experiment 4: Nested Free Energy
    run_new_experiments.py           Run all 4 experiments and generate figures
    semfabric_sim.py                 Legacy simulation (kept for reference)
    run_experiments.py               Legacy figure generation (kept for reference)
```

## Experiments

The computational demonstration validates the theoretical framework through four experiments of increasing scope:

| Experiment | Claim | Result |
|---|---|---|
| **1. Artifact Standard Detection** | Even the simplest SemApp operation benefits from active inference | AI 87.8% vs Rule-based 80.4% vs ML 79.0% (advantage on ambiguous artifacts: 68.3% vs 51.0%/48.3%) |
| **2. Ambiguity Resolution** | The SRF's relational structure enables genuine semiosis | AI 88.0% (100% on polysemous Signs) vs Rule 58.0% vs ML 48.0% |
| **3. Multi-Wave Entity Resolution** | The shared fabric enables cooperative intelligence through overlapping Markov blankets | Cooperative 94.0% > Isolated 85.3% > Homogeneous 61.2% > Generalist 56.2% (N=20 replicates) |
| **4. Nested Free Energy** | The fabric exhibits multi-scale self-organization | Fabric uncertainty 1.79 &rarr; 0.29 nats, monotonic decrease |

## Running the Simulations

### Requirements

- Python 3.9+
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

### Generate all figures

```bash
cd paper/simulation
python run_new_experiments.py
```

This runs all four experiments and saves figures to `paper/figures/`. Total runtime is approximately 2-3 minutes.

### Run individual experiments

```bash
cd paper/simulation
python exp1_asd.py            # ~3s
python exp2_disambiguation.py  # ~2s
python exp3_entity_resolution.py  # ~2 min (20 replicates)
python exp4_nested_vfe.py      # ~3s
```

### Building the paper

The paper is a standard LaTeX document. Compile with:

```bash
cd paper
pdflatex semapp-cognitive-substrate.tex
bibtex semapp-cognitive-substrate
pdflatex semapp-cognitive-substrate.tex
pdflatex semapp-cognitive-substrate.tex
```

Or upload `paper/` to [Overleaf](https://www.overleaf.com).

## Technical Notes

The simulation implements discrete active inference following the [pymdp](https://github.com/infer-actively/pymdp) generative model specification (Heins et al., 2022) in pure NumPy. This avoids JAX JIT compilation overhead for the small state spaces used here while maintaining mathematical equivalence with pymdp's inference algorithms:

- **State inference**: Variational mean-field coordinate ascent
- **Policy evaluation**: Expected Free Energy (EFE) with pragmatic + epistemic components
- **Action selection**: Softmax with policy precision parameter gamma

## License

All rights reserved.
