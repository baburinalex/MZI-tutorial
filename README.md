# Mach–Zehnder Interferometer Tutorial

An educational walkthrough of **Mach–Zehnder interferometer (MZI)** physics on the silicon-photonics (SOI) platform — from the 2×2 coupler matrix all the way to picking the device geometry. Every figure is produced by a single, self-contained Python script you can run yourself.

> Companion to the `ring-resonator-tutorial`. Same platform, same five-level structure — the natural next device after the ring. Where the ring is a resonant, multi-beam filter, the MZI is a two-beam interferometer: the workhorse modulator and switch of integrated photonics.

![Transmission spectrum](images/fig2_transmission.png)

## What's inside

The guide is built in five levels, each going one step further — and each with a figure and runnable code:

1. **Building blocks** — the 2×2 coupler matrix `t, κ`, energy conservation, per-arm phase and loss, and the arm phase difference `Δφ`.
2. **Transmission spectrum** — the master formulas `T_bar(λ)`, `T_cross(λ)` and the complementary-output condition.
3. **Key metrics** — Free Spectral Range (FSR), extinction ratio (ER), insertion loss (IL), and the balanced-arm ("critical") condition for perfect extinction.
4. **Geometry → physics** — how arm imbalance `ΔL` maps to FSR, how coupler length `Lc` maps to the 50:50 split, and how a heater maps to phase via the thermo-optic effect.
5. **Inverse design** — given a target FSR and ER, solve for `ΔL`, the splitter length, and the arm-balance budget; a design map in the (ΔL, imbalance) plane.

The full tutorial text is in [`mzi_interferometer_tutorial.md`](mzi_interferometer_tutorial.md) (in Russian).

## Quick start

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python mzi_tutorial.py
```

This regenerates all six figures into `images/`.

## Repository layout

```
mach-zehnder-tutorial/
├── README.md
├── LICENSE
├── requirements.txt
├── mzi_tutorial.py                  # physics functions + figure generation
├── mzi_interferometer_tutorial.md   # the teaching guide (5 levels + bonus)
└── images/                          # generated figures used in the guide
```

## The physics, in one paragraph

A Mach–Zehnder interferometer splits light into two waveguide arms with a 3-dB coupler and recombines it with a second coupler. Each arm imparts a phase; their difference `Δφ` decides which output the light leaves from. Make one arm longer by `ΔL` and the phase difference becomes wavelength-dependent, producing a periodic spectrum whose period (FSR) is set by `ΔL`. The two outputs are complementary (`T_bar + T_cross = 1` when lossless). A perfect null at the cross port requires the two arms to deliver equal amplitudes — the MZI's analogue of the ring's critical coupling. Unlike the ring, the MZI interferes only two beams once, so its fringes are broad sinusoids (finesse ≈ 2) rather than sharp resonances; that, plus its predictable transfer function, is exactly why it dominates modulator and switch design.

## How it relates to the ring tutorial

| | Ring resonator | Mach–Zehnder interferometer |
|---|---|---|
| Phase quantity | round-trip phase `φ = 2π n_eff L / λ` | arm phase difference `Δφ = 2π n_eff ΔL / λ` |
| FSR | `λ² / (n_g · L)` | `λ² / (n_g · ΔL)` |
| Perfect extinction | critical coupling `t = A` | balanced arms `a₁ = a₂` |
| Lineshape | Lorentzian dip, tunable `Q` | sinusoidal fringe, fixed finesse ≈ 2 |
| Beams interfering | many (resonant build-up) | two (single pass) |
| Typical use | narrowband filter, sensor | modulator, switch, (de)mux |

## Platform constants (used in examples)

SOI strip waveguide, 220 × 500 nm cross-section, TE mode, λ ≈ 1550 nm: `n_eff ≈ 2.45`, `n_g ≈ 4.20`, propagation loss ≈ 2 dB/cm, thermo-optic `dn/dT ≈ 1.86e-4 /K`. These are typical teaching values; a real design takes them from a mode solver.

## License

MIT — see [`LICENSE`](LICENSE).
