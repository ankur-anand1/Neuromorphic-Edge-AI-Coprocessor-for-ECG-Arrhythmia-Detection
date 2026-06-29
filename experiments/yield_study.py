#!/usr/bin/env python3
"""
Yield study experiment: plot accuracy vs. defect rate.

Run: python experiments/yield_study.py
Output: experiments/yield_curve.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.system.coprocessor import NeuromorphicCoprocessor


def main() -> None:
    print("Running Monte Carlo yield study (this may take ~30 seconds)...")
    coprocessor = NeuromorphicCoprocessor()
    results = coprocessor.fault_tolerance_study(n_beats=200, n_trials=10)

    rates = [r["defect_rate_pct"] for r in results]
    accs = [r["mean_accuracy"] * 100 for r in results]
    stds = [r["std_accuracy"] * 100 for r in results]
    threshold = results[0]["threshold"] * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(rates, accs, yerr=stds, fmt="o-", capsize=4, color="#2563eb", label="Mean accuracy")
    ax.axhline(threshold, color="#dc2626", linestyle="--", label=f"Threshold ({threshold:.0f}%)")
    ax.axvline(2.0, color="#16a34a", linestyle=":", alpha=0.7, label="2% defect rate (pitch claim)")

    ax.set_xlabel("Open-Circuit Defect Rate (%)", fontsize=12)
    ax.set_ylabel("Arrhythmia Classification Accuracy (%)", fontsize=12)
    ax.set_title("Fault Tolerance: CiM Crossbar Yield vs. Detection Accuracy", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 105)

    out = Path(__file__).parent / "yield_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved plot to {out}")

    # Print table
    print(f"\n{'Defect %':>10} {'Accuracy %':>12} {'Pass Rate':>12}")
    for r in results:
        print(f"{r['defect_rate_pct']:>10.1f} {r['mean_accuracy']*100:>12.1f} {r['pass_rate']*100:>11.0f}%")


if __name__ == "__main__":
    main()
