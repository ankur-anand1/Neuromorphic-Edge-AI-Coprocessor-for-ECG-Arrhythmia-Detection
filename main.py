#!/usr/bin/env python3
"""
Fault-Tolerant Mixed-Signal Neuromorphic Coprocessor — Main Demo

Demonstrates cycle-accurate emulation of a CiM edge processor for
cardiac arrhythmia detection with manufacturing defect tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.system.coprocessor import NeuromorphicCoprocessor
from src.signal.ecg_loader import ARRHYTHMIA_CLASSES
from src.signal.preprocessing import extract_features, bandpass_filter
from src.signal.ecg_loader import load_ecg_dataset


def demo_inference(coprocessor: NeuromorphicCoprocessor) -> None:
    print("\n" + "=" * 60)
    print("  DEMO 1: Single-Beat Inference (Analog CiM Path)")
    print("=" * 60)

    beats, labels, fs = load_ecg_dataset(n_beats=10)
    beats = bandpass_filter(beats, fs)
    features = extract_features(beats[:1])

    result = coprocessor.run_inference(features[0])
    print(f"  Predicted class : {result['class_name']}")
    print(f"  Probabilities   : {[f'{p:.3f}' for p in result['probability']]}")
    print(f"  RISC cycles     : {result['risc_stats']['total_cycles']}")
    print(f"  CPI             : {result['risc_stats']['cpi']}")
    print(f"  Crossbar defects: {result['crossbar_summary']['defect_rate_pct']}%")


def demo_classification(coprocessor: NeuromorphicCoprocessor, n_beats: int) -> None:
    print("\n" + "=" * 60)
    print("  DEMO 2: Full Dataset Classification")
    print("=" * 60)

    result = coprocessor.evaluate_dataset(n_beats=n_beats)
    print(f"  Beats classified : {result['n_beats']}")
    print(f"  Accuracy         : {result['accuracy'] * 100:.1f}%")
    print(f"  Defect rate      : {result['defect_rate'] * 100:.2f}%")

    preds = result["predictions"]
    labels = result["labels"]
    for cls_id, cls_name in ARRHYTHMIA_CLASSES.items():
        mask = labels == cls_id
        if mask.sum() > 0:
            cls_acc = (preds[mask] == labels[mask]).mean()
            print(f"  {cls_name:40s}: {cls_acc * 100:.1f}% ({mask.sum()} beats)")


def demo_fault_tolerance(coprocessor: NeuromorphicCoprocessor, n_beats: int) -> None:
    print("\n" + "=" * 60)
    print("  DEMO 3: Fault-Tolerance / Yield Study")
    print("  (Proving arrhythmia detection survives hardware defects)")
    print("=" * 60)

    results = coprocessor.fault_tolerance_study(n_beats=n_beats, n_trials=5)
    print(f"\n  {'Defect Rate':>12}  {'Mean Accuracy':>14}  {'Pass Rate':>10}  {'Status'}")
    print("  " + "-" * 52)

    for r in results:
        status = "PASS" if r["pass_rate"] >= 0.8 else "FAIL"
        print(
            f"  {r['defect_rate_pct']:>10.1f}%  "
            f"{r['mean_accuracy'] * 100:>12.1f}%  "
            f"{r['pass_rate'] * 100:>8.0f}%  "
            f"{status}"
        )

    # Highlight the 2% claim from the elevator pitch
    r2 = next((r for r in results if abs(r["defect_rate_pct"] - 2.0) < 0.01), None)
    if r2:
        print(f"\n  >> At 2% defect rate: {r2['mean_accuracy'] * 100:.1f}% accuracy")
        print(f"     Pass rate (>=90% threshold): {r2['pass_rate'] * 100:.0f}%")


def demo_system_info(coprocessor: NeuromorphicCoprocessor) -> None:
    print("\n" + "=" * 60)
    print("  SYSTEM ARCHITECTURE SUMMARY")
    print("=" * 60)
    summary = coprocessor.system_summary()
    for key, val in summary.items():
        print(f"  {key:20s}: {val}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neuromorphic CiM Coprocessor Emulator for Edge Arrhythmia Detection"
    )
    parser.add_argument("--defect-rate", type=float, default=0.0,
                        help="Initial defect fraction (0.0–0.05)")
    parser.add_argument("--n-beats", type=int, default=200,
                        help="Number of ECG beats to classify")
    parser.add_argument("--demo", choices=["all", "inference", "classify", "yield", "info"],
                        default="all", help="Which demo to run")
    parser.add_argument("--json", action="store_true", help="Output yield study as JSON")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  FAULT-TOLERANT MIXED-SIGNAL NEUROMORPHIC COPROCESSOR")
    print("  Edge Arrhythmia Detection - Cycle-Accurate Emulator")
    print("=" * 60)

    coprocessor = NeuromorphicCoprocessor(defect_rate=args.defect_rate)

    if args.demo in ("all", "info"):
        demo_system_info(coprocessor)
    if args.demo in ("all", "inference"):
        demo_inference(coprocessor)
    if args.demo in ("all", "classify"):
        demo_classification(coprocessor, args.n_beats)
    if args.demo in ("all", "yield"):
        if args.json:
            results = coprocessor.fault_tolerance_study(n_beats=args.n_beats)
            print(json.dumps(results, indent=2))
        else:
            demo_fault_tolerance(coprocessor, args.n_beats)

    print("\n" + "=" * 60)
    print("  Demo complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
