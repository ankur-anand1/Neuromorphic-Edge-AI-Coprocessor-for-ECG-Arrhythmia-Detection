# Fault-Tolerant Mixed-Signal Neuromorphic Coprocessor

**A cycle-accurate software emulator for Compute-in-Memory (CiM) edge arrhythmia detection.**


## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   16-bit RISC Control Core                   │
│         IF → ID → EX → MEM → WB  (cycle-accurate)           │
│    Custom ISA: CIM_START | ADC_START | WFI | HALT            │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐   ┌──────────────────────────────┐
│   Memristor Crossbar     │   │      10-bit SAR ADC          │
│   16×8 conductance array │──▶│   Mixed-signal boundary      │
│   I = G × V  (Ohm's Law) │   │   12 cycles/conversion       │
└──────────────────────────┘   └──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│              ECG Signal Pipeline (Edge)                      │
│   Bandpass Filter → R-Peak → DCT Features → Classify        │
│   Classes: NSR | PVC | APB  (MIT-BIH compatible)            │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│           Manufacturing Yield Model (VLSI)                   │
│   Monte Carlo open-circuit defect injection                  │
│   Accuracy vs. defect rate characterization                  │
└─────────────────────────────────────────────────────────────┘
```


## Project Structure

```
neuromorphic-cim-coprocessor/
├── main.py                          # Full demo entry point
├── requirements.txt
├── README.md
├── docs/
│   └── architecture.md              # Deep technical reference
├── experiments/
│   └── yield_study.py               # Monte Carlo yield plot
└── src/
    ├── crossbar/
    │   ├── memristor_array.py       # 16×8 resistive array + defects
    │   └── ohm_law_engine.py        # I = G×V analog MVM engine
    ├── adc/
    │   └── sar_adc.py               # 10-bit SAR ADC with INL/DNL
    ├── risc/
    │   ├── registers.py             # 16-bit GPR + coprocessor regs
    │   ├── isa.py                   # Custom instruction set
    │   ├── pipeline.py              # 5-stage cycle-accurate pipeline
    │   └── emulator.py              # Full RISC emulator
    ├── yield_model/
    │   └── defect_injector.py       # Monte Carlo yield model
    ├── signal/
    │   ├── ecg_loader.py            # Synthetic MIT-BIH ECG data
    │   └── preprocessing.py         # Filter, R-peak, DCT features
    ├── classifier/
    │   └── arrhythmia_detector.py   # Crossbar-mapped classifier
    └── system/
        └── coprocessor.py           # Top-level integration
```

---

## Quick Start

```bash
cd neuromorphic-cim-coprocessor
pip install -r requirements.txt
python main.py
```

### Run specific demos

```bash
# Single-beat analog inference
python main.py --demo inference

# Full dataset classification
python main.py --demo classify --n-beats 200

# Fault-tolerance yield study (the knockout demo)
python main.py --demo yield

# With 2% pre-injected defects
python main.py --defect-rate 0.02 --demo classify

# Generate yield curve plot
python experiments/yield_study.py
```

---

## Key Technical Details

### 1. Compute-in-Memory Crossbar
- **16 wordlines × 8 bitlines** memristor array
- Conductance range: 1 µS (HRS) → 100 µS (LRS)
- Matrix-vector multiply via **Ohm's Law**: `I_out = G_eff @ V_in`
- IR drop modeled on 500 Ω access lines
- Open-circuit defects zero out conductance at random cells

### 2. SAR ADC (Mixed-Signal Boundary)
- **10-bit** successive approximation register ADC
- 12 conversion cycles per sample
- INL distortion + comparator offset modeled
- SNR: 6.02×10 + 1.76 = **61.96 dB** theoretical

### 3. 16-bit RISC Architecture
- 8 general-purpose registers (R0 hardwired to 0)
- Custom coprocessor instructions: `CIM_START`, `ADC_START`
- Event-driven `WFI` (Wait-For-Interrupt) for power gating
- 5-stage pipeline with stall accounting for analog latency

### 4. Arrhythmia Detection
- Input: 800 ms ECG beat window @ 360 Hz
- Bandpass: 0.5–40 Hz Butterworth
- Features: 16 DCT coefficients → 16 wordlines
- Output classes: Normal (NSR), PVC, Atrial Premature Beat (APB)

### 5. Fault Tolerance / Yield Model
- Monte Carlo injection of **open-circuit (stuck-open) defects**
- Sweeps defect rates: 0% → 5%
- Pass criterion: ≥ 90% classification accuracy
- **Key result**: System maintains clinical-grade detection at 2% defect rate

---
