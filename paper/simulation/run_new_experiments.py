"""
run_new_experiments.py — Run all 4 new experiments and generate figures
======================================================================

Experiments:
  1. Artifact Standard Detection (ASD)
  2. Ambiguity Resolution (Content-Level Semiosis)
  3. Multi-Wave Entity Resolution + Cross-Domain Inference
  4. Nested Free Energy (Cognitive Substrate)

Output: PDF + PNG figures in paper/figures/
"""

import os
import sys
import time

# Ensure simulation directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp1_asd import run_experiment_1, plot_experiment_1
from exp2_disambiguation import run_experiment_2, plot_experiment_2
from exp3_entity_resolution import run_experiment_3, plot_experiment_3
from exp4_nested_vfe import run_experiment_4, plot_experiment_4
from common import FIG_DIR


def main():
    print("=" * 60)
    print("SemApp Cognitive Substrate — New Experiment Suite")
    print("=" * 60)
    print()

    total_start = time.time()

    # Experiment 1
    t0 = time.time()
    results_1 = run_experiment_1(seed=42)
    plot_experiment_1(results_1)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    # Experiment 2
    t0 = time.time()
    results_2 = run_experiment_2(seed=42)
    plot_experiment_2(results_2)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    # Experiment 3
    t0 = time.time()
    results_3 = run_experiment_3(seed=42)
    plot_experiment_3(results_3)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    # Experiment 4
    t0 = time.time()
    results_4 = run_experiment_4(seed=42)
    plot_experiment_4(results_4)
    print(f"  Time: {time.time() - t0:.1f}s\n")

    total_time = time.time() - total_start
    print("=" * 60)
    print(f"All experiments complete. Total time: {total_time:.1f}s")
    print(f"Figures saved to: {os.path.abspath(FIG_DIR)}")
    print("=" * 60)


if __name__ == '__main__':
    main()
