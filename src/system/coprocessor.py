"""
Full neuromorphic coprocessor system integration.

Ties together: 16-bit RISC + CiM crossbar + SAR ADC + yield model + ECG pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..adc.sar_adc import SARADC
from ..classifier.arrhythmia_detector import (
    ArrhythmiaDetector,
    CrossbarClassifier,
    INPUT_FEATURES,
    HIDDEN_SIZE,
)
from ..crossbar.memristor_array import MemristorCrossbar
from ..crossbar.ohm_law_engine import OhmLawEngine
from ..risc.emulator import RISCEmulator
from ..risc.isa import Opcode, encode_instruction
from ..signal.ecg_loader import load_ecg_dataset, ARRHYTHMIA_CLASSES
from ..classifier.trainer import train_classifier
from ..signal.preprocessing import extract_features, bandpass_filter
from ..yield_model.defect_injector import YieldModel


def _build_inference_program() -> list[int]:
    """Compile a minimal inference microprogram for the RISC coprocessor."""
    return [
        encode_instruction(Opcode.LDI, rd=1, imm=0x80),   # Load input scale
        encode_instruction(Opcode.CIM_START),              # Trigger analog MVM
        encode_instruction(Opcode.ADC_START),              # Trigger SAR ADC
        encode_instruction(Opcode.HALT),                   # Done
    ]


@dataclass
class NeuromorphicCoprocessor:
    """
    Top-level fault-tolerant mixed-signal neuromorphic coprocessor emulator.

    Components
    ----------
    - 16×8 memristor crossbar (Compute-in-Memory)
    - 10-bit SAR ADC (mixed-signal boundary)
    - 16-bit RISC control core (event-driven)
    - Manufacturing yield / defect injection model
    - ECG arrhythmia detection pipeline
    """

    defect_rate: float = 0.0
    seed: int = 42

    crossbar: MemristorCrossbar = field(init=False)
    ohm_engine: OhmLawEngine = field(init=False)
    sar_adc: SARADC = field(init=False)
    classifier: CrossbarClassifier = field(init=False)
    detector: ArrhythmiaDetector = field(init=False)
    risc: RISCEmulator = field(init=False)
    yield_model: YieldModel = field(init=False)

    def __post_init__(self) -> None:
        self.crossbar = MemristorCrossbar(rows=INPUT_FEATURES, cols=HIDDEN_SIZE)
        self.ohm_engine = OhmLawEngine(crossbar=self.crossbar)
        self.sar_adc = SARADC(resolution_bits=10, v_ref=1.0, conversion_cycles=12)
        self.classifier = CrossbarClassifier(
            crossbar=self.crossbar,
            ohm_engine=self.ohm_engine,
            sar_adc=self.sar_adc,
        )
        self.detector = ArrhythmiaDetector(classifier=self.classifier)
        self.risc = RISCEmulator(
            ohm_engine=self.ohm_engine,
            sar_adc=self.sar_adc,
            program=_build_inference_program(),
        )
        self.yield_model = YieldModel(crossbar=self.crossbar)

        # Train classifier weights on synthetic ECG data
        beats, labels, fs = load_ecg_dataset(n_beats=500, seed=42)
        beats_filtered = bandpass_filter(beats, fs)
        features = extract_features(beats_filtered, n_features=INPUT_FEATURES)
        train_acc = train_classifier(self.classifier, features, labels, epochs=800, lr=0.15)
        self.yield_model.save_ideal_state()
        self._train_accuracy = train_acc

        if self.defect_rate > 0:
            self.yield_model.inject_defects(self.defect_rate)

    def run_inference(self, features: np.ndarray) -> dict:
        """Run single-beat inference through full analog+digital path."""
        self.risc.regs.pc = 0
        self.risc.halted = False
        self.risc.pipeline.total_cycles = 0
        self.risc.pipeline.instructions_retired = 0
        self.risc.regs.write(1, int(features[0] * 1000) & 0xFFFF)
        stats = self.risc.run(max_cycles=150)

        logits = self.classifier.forward_analog(features)
        pred = int(np.argmax(logits))
        proba = self.classifier.predict_proba(features, analog=True)

        return {
            "prediction": pred,
            "class_name": ARRHYTHMIA_CLASSES[pred],
            "probability": proba,
            "risc_stats": stats,
            "crossbar_summary": self.crossbar.summary(),
        }

    def evaluate_dataset(self, n_beats: int = 200) -> dict:
        """Evaluate arrhythmia detection on full dataset."""
        beats, labels, fs = load_ecg_dataset(n_beats=n_beats, seed=99)
        beats_filtered = bandpass_filter(beats, fs)
        result = self.detector.detect(beats_filtered, labels)
        result["fs"] = fs
        result["defect_rate"] = self.crossbar.defect_rate
        return result

    def fault_tolerance_study(
        self,
        n_beats: int = 200,
        defect_rates: Optional[list[float]] = None,
        n_trials: int = 5,
    ) -> list[dict]:
        """Run yield Monte Carlo study across defect rates."""
        beats, labels, fs = load_ecg_dataset(n_beats=n_beats, seed=99)
        beats_filtered = bandpass_filter(beats, fs)
        features = extract_features(beats_filtered, n_features=INPUT_FEATURES)

        if defect_rates is None:
            defect_rates = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]

        def predict_fn(inputs):
            return self.classifier.predict(inputs)

        return self.yield_model.monte_carlo_study(
            predict_fn, features, labels, defect_rates, n_trials
        )

    def system_summary(self) -> dict:
        return {
            "architecture": "16-bit RISC + CiM Crossbar + SAR ADC",
            "crossbar": f"{INPUT_FEATURES}×{HIDDEN_SIZE} memristor array",
            "adc": f"{self.sar_adc.resolution_bits}-bit SAR, {self.sar_adc.conversion_cycles} cycles/conv",
            "classes": list(ARRHYTHMIA_CLASSES.values()),
            "defect_rate_pct": round(self.crossbar.defect_rate * 100, 2),
            "train_accuracy_pct": round(getattr(self, "_train_accuracy", 0) * 100, 1),
            "risc_pc": self.risc.regs.pc,
        }
