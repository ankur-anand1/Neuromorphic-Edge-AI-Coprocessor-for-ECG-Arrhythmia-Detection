# Architecture Deep Dive

## 1. System Block Diagram

```
ECG Sensor (360 Hz)
       │
       ▼
┌──────────────┐     ┌─────────────────────────────────────────┐
│  Bandpass    │     │         16-bit RISC Control Core         │
│  0.5-40 Hz   │     │                                          │
└──────┬───────┘     │  ┌────┐   ┌────┐   ┌────┐   ┌────┐      │
       │             │  │ IF │ → │ ID │ → │ EX │ → │ MEM│ → WB│
       ▼             │  └────┘   └────┘   └────┘   └────┘      │
┌──────────────┐     │                    │                     │
│  R-Peak Det  │     │              CIM_START                   │
│  + Beat Seg  │     │              ADC_START                   │
└──────┬───────┘     │              WFI (sleep)                 │
       │             └──────────┬──────────────┬────────────────┘
       ▼                        │              │
┌──────────────┐                ▼              ▼
│  DCT Feature │     ┌──────────────┐  ┌──────────────┐
│  16 coeffs   │────▶│  Crossbar    │─▶│  SAR ADC     │
└──────────────┘     │  16×8 G array│  │  10-bit      │
                     │  I = G × V    │  │  12 cyc/conv │
                     └──────────────┘  └──────┬───────┘
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │ Digital Head │
                                     │ 8 → 3 logits │
                                     └──────┬───────┘
                                            │
                                            ▼
                                     NSR / PVC / APB
```

## 2. Memristor Crossbar Physics

Each cell at row `i`, column `j` stores conductance `G_ij` (Siemens).

**Programming:**
```
G_ij = G_HRS + |w_ij| × (G_LRS - G_HRS)
```
where `w_ij ∈ [-1, 1]` is the normalized neural weight.

**Read operation (one MAC cycle):**
```
V_cell[i,j] = V_in[i] × V_read / (1 + R_line × Σ_j G_ij)
I_out[j] = Σ_i G_ij × V_cell[i,j]        (KCL at bitline j)
V_bl[j] = I_out[j] × T_pulse / C_bl       (integration)
```

**Defect model:**
```
G_ij → 0  if cell is stuck-open (probability p_defect)
```

## 3. SAR ADC Conversion

Binary search algorithm modeled with:
- **Resolution:** N = 10 bits → 1024 levels
- **LSB:** V_ref / 2^N = 976.6 µV
- **INL:** Sinusoidal distortion ±0.5 LSB
- **Latency:** N + 2 = 12 clock cycles
- **SNR:** 6.02N + 1.76 = 61.96 dB (ideal)

Conversion pseudocode:
```
for bit = N-1 down to 0:
    V_dac = V_ref × code / 2^N
    if V_in > V_dac + V_offset:
        code[bit] = 1
    cycles += 1
```

## 4. RISC ISA Encoding (16-bit)

```
┌─────────┬───────┬───────┬───────┬──────────┐
│ opcode  │  rd   │  rs1  │  rs2  │   imm    │
│ 4 bits  │ 3 bit │ 3 bit │ 3 bit │  3+ bits │
└─────────┴───────┴───────┴───────┴──────────┘
```

| Opcode | Mnemonic | Description |
|--------|----------|-------------|
| 0x0 | ADD | R[rd] = R[rs1] + R[rs2] |
| 0x8 | LD | R[rd] = MEM[R[rs1] + imm] |
| 0x9 | ST | MEM[R[rs1] + imm] = R[rd] |
| 0xA | LDI | R[rd] = imm (8-bit) |
| 0xE | CIM_START | Trigger crossbar MVM |
| 0xF | ADC_START | Trigger SAR conversion |
| 0x12 | WFI | Sleep until event |
| 0x11 | HALT | Stop processor |

## 5. Inference Microprogram

```asm
; Address 0x0000: Arrhythmia inference kernel
    LDI  R1, #0x80      ; Input scale factor
    CIM_START            ; Analog MVM (1 + integration cycle)
    ADC_START            ; 8 × 12 = 96 ADC cycles
    HALT                 ; Total ≈ 100 cycles
```

## 6. Yield Model Methodology

For each defect rate `p` ∈ {0%, 0.5%, 1%, 1.5%, 2%, 3%, 5%}:

1. Reset crossbar to ideal state
2. Randomly select `⌊p × M × N⌋` cells
3. Set `G_ij = 0` (open-circuit) at selected cells
4. Run full arrhythmia classification on 200 ECG beats
5. Repeat for 10 Monte Carlo trials
6. Record mean accuracy and pass rate (≥ 90% threshold)

**Expected result at 2%:**
- ~3 defective cells out of 128
- MVM relative error ≈ 2–5%
- Classification accuracy remains ≥ 90%

## 7. Power & Latency Budget (Estimated)

| Component | Cycles | Power (est.) |
|-----------|--------|-------------|
| Crossbar MVM | 1 | 0.5 µW (analog) |
| SAR ADC (×8) | 96 | 8 × 50 µW |
| RISC overhead | 4 | 4 × 10 µW |
| **Total** | **~100** | **~450 µW** |

At 10 MHz clock → **10 µs inference latency** — well within the 800 ms ECG beat window.

## 8. Fault Tolerance Mechanisms

1. **Differential encoding:** Positive/negative weights on paired cells; common-mode defects cancel
2. **Digital output head:** 8→3 linear layer after ADC provides graceful degradation
3. **Over-provisioned crossbar:** 16×8 array for 16×3 effective mapping (redundancy)
4. **Threshold-based decision:** Softmax over 3 classes tolerates ±5% analog error
